import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
from helper import LLMAPI

import re
import polars as pl
import json
import numpy as np
import os

from prompt import graph_extraction_prompt, json_formatting_prompt, example_1_prompt, example_2_prompt, check_duplicate_entities_prompt, summarize_descriptions_prompt

import networkx as nx

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

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray):
    """
    Calculate the cosine similarity between two embeddings.

    Parameters:
        embedding1 (np.ndarray): The first embedding.
        embedding2 (np.ndarray): The second embedding.

    Returns:
        float: The cosine similarity between the two embeddings.
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

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


# the following helper function use llm api to summarize the two descriptions
def summarize_descriptions_llm(description1, description2, llm: LLMAPI):
    prompt = summarize_descriptions_prompt.format(description1=description1, description2=description2)
    response = llm.invoke(prompt)
    return response

def normalize_text(text):
    return text.strip().lower()

def is_duplicate(entity1, entity2):
    return normalize_text(entity1["entity_name"]) == normalize_text(entity2["entity_name"]) and entity1["entity_type"] == entity2["entity_type"]

# the following helper function use llm api to check if the entities are duplicate
def is_duplicate_llm(entity1, entity2, llm: LLMAPI):
    prompt = check_duplicate_entities_prompt.format(entity1_name=entity1["entity_name"], 
                                                     entity1_type=entity1["entity_type"], 
                                                     entity1_description=entity1["entity_description"],
                                                     entity2_name=entity2["entity_name"], 
                                                     entity2_type=entity2["entity_type"], 
                                                     entity2_description=entity2["entity_description"])
    response = llm.invoke(prompt)
    return response == 'yes'

def is_duplicate_emb(entity1, entity2, embedding: LLMAPI, threshold=0.8):
    embedding1 = np.array(embedding.embedding_text(entity1["entity_description"]))
    embedding2 = np.array(embedding.embedding_text(entity2["entity_description"]))
    similarity = calculate_cosine_similarity(embedding1, embedding2)
    return similarity > threshold or is_duplicate(entity1, entity2)

def resolve_entities(combined_dict, llm):
    unique_entities = []
    entity_map = {}  # Maps old entity names to new entity names

    for entity in tqdm(combined_dict['entities'], desc="Resolving entities"):
        found_duplicate = False
        for unique_entity in unique_entities:
            if is_duplicate(entity, unique_entity):
                # Merge attributes if needed
                found_duplicate = True
                entity_map[entity['entity_name']] = unique_entity['entity_name']
                # Summarize descriptions if they differ
                if entity['entity_description'] != unique_entity['entity_description']:
                    unique_entity['entity_description'] = summarize_descriptions_llm(
                        unique_entity['entity_description'],
                        entity['entity_description'],
                        llm
                    )
                break
        if not found_duplicate:
            unique_entities.append(entity)
            entity_map[entity['entity_name']] = entity['entity_name']


    # Update relationships to point to resolved entities
    for relationship in combined_dict['relationships']:
        relationship['source_entity'] = entity_map.get(relationship['source_entity'], relationship['source_entity'])
        relationship['target_entity'] = entity_map.get(relationship['target_entity'], relationship['target_entity'])

    return {
        'entities': unique_entities,
        'relationships': combined_dict['relationships']
    }, entity_map, unique_entities

# this function is used to resolve the entities by using llm to check whether the entities are duplicate
def resolve_entities_v2(combined_dict, llm, llm_2):
    unique_entities = []
    entity_map = {}  # Maps old entity names to new entity names

    for entity in tqdm(combined_dict['entities'], desc="Resolving entities"):
        found_duplicate = False
        for unique_entity in unique_entities:
            if is_duplicate_llm(entity, unique_entity, llm_2):
                # Merge attributes if needed
                found_duplicate = True
                entity_map[entity['entity_name']] = unique_entity['entity_name']
                # Summarize descriptions if they differ
                if entity['entity_description'] != unique_entity['entity_description']:
                    unique_entity['entity_description'] = summarize_descriptions_llm(
                        unique_entity['entity_description'],
                        entity['entity_description'],
                        llm
                    )
                break
        if not found_duplicate:
            unique_entities.append(entity)
            entity_map[entity['entity_name']] = entity['entity_name']


    # Update relationships to point to resolved entities
    for relationship in combined_dict['relationships']:
        relationship['source_entity'] = entity_map.get(relationship['source_entity'], relationship['source_entity'])
        relationship['target_entity'] = entity_map.get(relationship['target_entity'], relationship['target_entity'])

    return {
        'entities': unique_entities,
        'relationships': combined_dict['relationships']
    }, entity_map, unique_entities

# this function is used to resolve the entities by using embedding similarity to check whether the entities are duplicate
def resolve_entities_v3(combined_dict, llm):
    unique_entities = []
    entity_map = {}  # Maps old entity names to new entity names

    for entity in tqdm(combined_dict['entities'], desc="Resolving entities"):
        found_duplicate = False
        for unique_entity in unique_entities:
            if is_duplicate_emb(entity, unique_entity, llm):
                # Merge attributes if needed
                found_duplicate = True
                entity_map[entity['entity_name']] = unique_entity['entity_name']
                # Summarize descriptions if they differ
                if entity['entity_description'] != unique_entity['entity_description']:
                    unique_entity['entity_description'] = summarize_descriptions_llm(
                        unique_entity['entity_description'],
                        entity['entity_description'],
                        llm
                    )
                break
        if not found_duplicate:
            unique_entities.append(entity)
            entity_map[entity['entity_name']] = entity['entity_name']

    # Update relationships to point to resolved entities
    for relationship in combined_dict['relationships']:
        relationship['source_entity'] = entity_map.get(relationship['source_entity'], relationship['source_entity'])
        relationship['target_entity'] = entity_map.get(relationship['target_entity'], relationship['target_entity'])

    return {
        'entities': unique_entities,
        'relationships': combined_dict['relationships']
    }, entity_map, unique_entities


class GraphDataManager:
    def __init__(self):
        self.graph = nx.Graph()

    def create_entity(self, entity_name, entity_type, entity_description):
        self.graph.add_node(entity_name, type=entity_type, description=entity_description)

    def create_relationship(self, source_entity, target_entity, relationship_description, relationship_strength):
        self.graph.add_edge(source_entity, target_entity, description=relationship_description, strength=relationship_strength)

    def load_graph_from_data(self, data: dict):
        for entity in data['entities']:
            self.create_entity(entity['entity_name'], entity['entity_type'], entity['entity_description'])
        for relationship in data['relationships']:
            self.create_relationship(
                relationship['source_entity'],
                relationship['target_entity'],
                relationship['relationship_description'],
                relationship['relationship_strength']
            )

    def drop_existing_graph(self):
        self.graph.clear()

    def verify_relationship_weights(self):
        missing_weights = [
            (u, v) for u, v, d in self.graph.edges(data=True) if 'strength' not in d
        ]
        if missing_weights:
            print("Warning: Some relationships do not have strengths assigned:", missing_weights)

