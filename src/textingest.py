import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
from helper import LLMAPI

import re
import polars as pl
import json

from prompt import graph_extraction_prompt, json_formatting_prompt, example_1_prompt, example_2_prompt


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


    def __process_sentence_chunk(self, sentence_chunk, embedding):
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" -> ". A"
        chunk_char_count = len(joined_sentence_chunk)
        chunk_word_count = len(joined_sentence_chunk.split(" "))
        chunk_token_count = chunk_char_count / 4  # 1 token = ~4 characters
        embedding_result = embedding.embedding_text(joined_sentence_chunk)
        return joined_sentence_chunk, chunk_char_count, chunk_word_count, chunk_token_count, embedding_result

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
                chunk_dict["embedding"] = embedding.embedding_text(joined_sentence_chunk)

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