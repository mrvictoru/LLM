import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
from helper import LLMAPI

import re
import polars as pl
import json
import numpy as np
import os

from prompt import graph_extraction_prompt, json_formatting_prompt, example_1_prompt, example_2_prompt

from neo4j import GraphDatabase

class PDFDocumentHandler:
    def __init__(self, pdf_path: str, dict_prompt: dict = None, chunk_size: int = 10, lang=English()):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.lang = lang

        if dict_prompt is None:
            dict_prompt = {
                "graph_extraction_prompt": graph_extraction_prompt,
                "json_formatting_prompt": json_formatting_prompt,
                "example_1_prompt": example_1_prompt,
                "example_2_prompt": example_2_prompt,
            }
        self.dict_prompt = dict_prompt
        self.pdf_document = None
        self.pdf_content = None
        self.pages_and_chunks = None
        self.chunks_and_graphs = None


    def open_pdf(self):
        self.pdf_document = fitz.open(self.pdf_path)


    def close_pdf(self):
        if self.pdf_document:
            self.pdf_document.close()


    def __chunk_sentences(self, sentences: list[str]) -> list[list[str]]:
        """
        Chunks a list of sentences into groups of a specified size.

        Parameters:
            sentences (list[str]): A list of sentences to be chunked.

        Returns:
            list[list[str]]: A list of lists, each containing a chunk of sentences.
        """
        return [sentences[i:i + self.chunk_size] for i in range(0, len(sentences), self.chunk_size)]

    def read_pdf(self) -> pl.DataFrame:
        """
        Reads the PDF document, extracts text content page by page, chunks sentences, and collects statistics.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the page number
            (adjusted), character count, word count, sentence count, token count, the extracted text, the sentences, and the chunked groups of sentences
            for each page.
        """
        if self.pdf_document is None:
            print("Opening PDF document...")
            self.open_pdf()

        pdf_content = []
        # add sentencizer to the language object
        self.lang.add_pipe("sentencizer")
        # loop through the pdf page using enumerate
        print("Loading PDF document...")
        for iter_page in tqdm(enumerate(self.pdf_document), total=len(self.pdf_document), desc="Reading PDF"):
            page_num = iter_page[0]
            text = iter_page[1].get_text()
            clean_text = text.replace("\n", " ").strip()
            # get the sentences from the text using sentencizer
            sentencizer = self.lang(clean_text).sents
            # loop through the sentences and store them in a list
            sentences = [str(sent.text) for sent in sentencizer]
            sentence_count = len(sentences)
            # chunk the sentences into groups of 10
            chunked_sentences = self.__chunk_sentences(sentences)
            pdf_content.append(
                {
                    "page": page_num + 1,
                    "char_count": len(clean_text),
                    "word_count": len(clean_text.split(" ")),
                    "sentence_spacy_count": sentence_count,
                    "chunk_count": self.chunk_size,
                    "token_count": len(clean_text.split()) / 4,  # rough estimate of tokens
                    "text": clean_text,
                    "sentences": sentences,
                    "sentence_chunks": chunked_sentences,
                }
            )
        self.pdf_content = pdf_content
        return pl.DataFrame(pdf_content)


    def embed_chunks(self, embedding: LLMAPI) -> pl.DataFrame:
        """
        Embeds the chunks of sentences using the specified embedding model.

        Parameters:
            embedding (LLMAPI): The embedding model to use for embedding the chunks.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the page number, sentence chunk, chunk statistics, and the embedded chunk.
        """
        if self.pdf_content is None:
            self.read_pdf()
        pages_and_chunks = []
        for page in tqdm(self.pdf_content, desc="Embedding sentence chunks"):
            for sentence_chunk in page["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page"] = page["page"]
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" -> ". A"
                chunk_dict["sentence_chunk"] = joined_sentence_chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
                chunk_dict["chunk_token_count"] = chunk_dict["chunk_char_count"] / 4  # 1 token = ~4 characters
                chunk_dict["embedding"] = np.array(embedding.embedding_text(joined_sentence_chunk))

                pages_and_chunks.append(chunk_dict)

        self.pages_and_chunks = pages_and_chunks
        return pl.DataFrame(pages_and_chunks)
    
    def __get_graph(self, text:str, nlp: LLMAPI):
        main_prompt = self.dict_prompt["graph_extraction_prompt"]
        formatted_prompt = main_prompt.format(json_formatting_prompt=self.dict_prompt["json_formatting_prompt"], example_1_prompt=self.dict_prompt["example_1_prompt"], example_2_prompt=self.dict_prompt["example_2_prompt"], text=text)
        output = nlp.invoke(formatted_prompt)
        # check if the output is valid JSON
        try:
            json_output = json.loads(output)
        except json.JSONDecodeError:
            print(output)
            raise ValueError("Invalid JSON output from the NLP model.")
        return json_output

    def graph_extraction(self, nlp: LLMAPI) -> pl.DataFrame:
        """
        Extracts entities and relationships and store it in a new column in self.pages_and_chunks from the sentences chunks from self.pages_and_chunks by prompting the NLP model.

        Parameters:
            nlp (LLMAPI): The NLP model to use for entity and relationship extraction.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the page number, entity, relationship, and the extracted text.
        """

        if self.pages_and_chunks is None:
            self.embed_chunks(nlp)
        chunks_and_graphs = []
        # loop through the sentence_chunk and create graph using __get_graph
        for chunk in tqdm(self.pages_and_chunks, desc="Extracting entities and relationships"):
            graph = {}
            graph["page"] = chunk["page"]
            graph["sentence_chunk"] = chunk["sentence_chunk"]
            graph["graph_extraction"] = self.__get_graph(chunk["sentence_chunk"], nlp)
            chunks_and_graphs.append(graph)

        self.chunks_and_graphs = chunks_and_graphs

        return pl.DataFrame(chunks_and_graphs)

    def save_graphs(self, path: str):
        if self.chunks_and_graphs is None:
            raise ValueError("No graphs to save. Please extract the graphs first.")
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        df = pl.DataFrame(self.chunks_and_graphs)
        df.write_json(path)
        print(f"Graphs saved to {path}.")



def find_similar_chunk_np(prompt_embedding: np.ndarray, pages_and_chunks, threshold=0.5):
    """
    Find the cosine similarity between a prompt embedding and all the embeddings from embedded sentence chunks,
    and filter the results to only include rows with a cosine similarity above a certain threshold.


    Parameters:
        prompt_embedding: The prompt_embedding to compare.
        pages_and_chunks_df (pl.DataFrame): The Polars DataFrame containing page information and embeddings.
        threshold (float): The cosine similarity threshold for filtering the results.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the page number, sentence chunk, and cosine similarity score.
    """
    # Get the embeddings as a NumPy array
    embeddings_np = np.stack(pages_and_chunks["embedding"].to_numpy())
    # Stack the prompt as a NumPy array
    prompt_np = np.stack([prompt_embedding] * embeddings_np.shape[0])

    # Normalize the text embeddings
    norm_text_embeddings = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    # Normalize the prompt embeddings
    norm_prompt_embeddings = prompt_np / np.linalg.norm(prompt_np, axis=1, keepdims=True)

    # Calculate the cosine similarity
    cosine_similarity = np.diag(np.dot(norm_text_embeddings, norm_prompt_embeddings.T))

    # take each element in the array and add it as a row in a new column to the dataframe
    df = pages_and_chunks.with_columns(pl.Series("cosine_similarity", cosine_similarity))

    # Filter rows based on the threshold
    df = df.filter(pl.col("cosine_similarity") > threshold)
    sorted_df = df.sort("cosine_similarity", descending=True)

    return sorted_df.select(["page", "sentence_chunk", "cosine_similarity"])

class GraphDatabaseConnection:
    def __init__(self, uri, user, password):
        if not uri or not user or not password:
            raise ValueError(
                "URI, user, and password must be provided to initialize the DatabaseConnection.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_session(self):
        return self.driver.session()

    def clear_database(self):
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")

class GraphDataLoader:
    def __init__(self, db_connection: GraphDatabaseConnection):
        self.db_connection = db_connection

    def create_entity(self, entity_name, entity_type, entity_description):
        with self.db_connection.get_session() as session:
            session.write_transaction(self._create_entity, entity_name, entity_type, entity_description)

    @staticmethod
    def _create_entity(tx, entity_name, entity_type, entity_description):
        query = (
            "CREATE (e:Entity {name: $entity_name, type: $entity_type, description: $entity_description})"
        )
        tx.run(query, entity_name=entity_name, entity_type=entity_type, entity_description=entity_description)

    def create_relationship(self, source_entity, target_entity, relationship_description, relationship_strength):
        with self.db_connection.get_session() as session:
            session.write_transaction(self._create_relationship, source_entity, target_entity, relationship_description, relationship_strength)

    @staticmethod
    def _create_relationship(tx, source_entity, target_entity, relationship_description, relationship_strength):
        query = (
            "MATCH (a:Entity {name: $source_entity}), (b:Entity {name: $target_entity}) "
            "CREATE (a)-[r:RELATED_TO {description: $relationship_description, strength: $relationship_strength}]->(b)"
        )
        tx.run(query, source_entity=source_entity, target_entity=target_entity, relationship_description=relationship_description, relationship_strength=relationship_strength)

    def load_data(self, data):
        for entity in data['entities']:
            self.create_entity(entity['entity_name'], entity['entity_type'], entity['entity_description'])
        
        for relationship in data['relationships']:
            self.create_relationship(relationship['source_entity'], relationship['target_entity'], relationship['relationship_description'], relationship['relationship_strength'])