import fitz
from tqdm.auto import tqdm
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from helper import LLMAPI, calculate_cosine_similarity
from docling.document_converter import DocumentConverter

import re
import polars as pl
import json
import numpy as np
import os

from prompt import graph_extraction_prompt, extraction_json_formatting_prompt, extraction_example_1_prompt, extraction_example_2_prompt, check_duplicate_entities_prompt, summarize_descriptions_prompt, community_report_generation_prompt, community_report_format_prompt, community_report_example_prompt

import networkx as nx
import community as community_louvain
import plotly.graph_objects as go

import logging
# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDocumentHandler:
    def __init__(self, pdf_path: str, dict_prompt: dict = None, chunk_size: int = 10):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size

        if dict_prompt is None:
            dict_prompt = {
                "graph_extraction_prompt": graph_extraction_prompt,
                "json_formatting_prompt": extraction_json_formatting_prompt,
                "example_1_prompt": extraction_example_1_prompt,
                "example_2_prompt": extraction_example_2_prompt,
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


    def __chunk_sentences(self, sentences: list[str], overlap_size: int = 2) -> list[list[str]]:
        """
        Chunks a list of sentences into overlapping groups of a specified size.
    
        Parameters:
            sentences (list[str]): A list of sentences to be chunked.
            overlap_size (int): Number of overlapping sentences between chunks.
    
        Returns:
            list[list[str]]: A list of lists, each containing a chunk of sentences with overlap.
        """
        if overlap_size >= self.chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size.")
        
        step_size = self.chunk_size - overlap_size
        chunks = []
        for i in range(0, len(sentences), step_size):
            chunk = sentences[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

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

        # loop through the pdf page using enumerate
        print("Loading PDF document...")
        for iter_page in tqdm(enumerate(self.pdf_document), total=len(self.pdf_document), desc="Reading PDF"):
            page_num = iter_page[0]
            text = iter_page[1].get_text()
            clean_text = text.replace("\n", " ").strip()

            # Get the sentences from the text using NLTK
            sentences = sent_tokenize(clean_text)
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
                
                emb_array = embedding.embedding_text(joined_sentence_chunk)
                if emb_array is None:
                    logger.error(f"Error embedding text: {e}")
                    logger.error(f"Text: {joined_sentence_chunk}")
                    continue
                chunk_dict["embedding"] = np.array(emb_array)
                pages_and_chunks.append(chunk_dict)

        self.pages_and_chunks = pages_and_chunks
        return pl.DataFrame(pages_and_chunks)
    
    def __get_graph(self, text:str, nlp: LLMAPI):
        main_prompt = self.dict_prompt["graph_extraction_prompt"]
        formatted_prompt = main_prompt.format(extraction_json_formatting_prompt=self.dict_prompt["json_formatting_prompt"], extraction_example_1_prompt=self.dict_prompt["example_1_prompt"], extraction_example_2_prompt=self.dict_prompt["example_2_prompt"], text=text)
        output = nlp.invoke(formatted_prompt)
        # check if the output is valid JSON
        try:
            json_output = json.loads(output)
        except json.JSONDecodeError:
            print(text)
            print(output)
            print("Invalid JSON output from the NLP model.")
            return None
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

# TODO implement a PDFDocumentHandler class that utilizes docling and markdown format of the extracted text which doesnt have page numbers
class MarkdownDocumentHandler(PDFDocumentHandler):
    def __init__(self, file_path: str, dict_prompt: dict = None, chunk_size: int = 10):
        super().__init__(pdf_path=file_path, dict_prompt=dict_prompt, chunk_size=chunk_size)
        self.converter = DocumentConverter()

    def read_pdf(self):
        # Process the Markdown text
        print("Converting PDF to Markdown...")
        result = self.converter.convert(self.pdf_path)
        md_text = result.document.export_to_markdown()

        # Get the sentences from the text using NLTK
        print("Chunking sentences...")
        sentences = sent_tokenize(md_text)
        sentence_count = len(sentences)
        chunked_sentences = self._PDFDocumentHandler__chunk_sentences(sentences)

        self.pdf_content = [{
            "page": 1,
            "char_count": len(md_text),
            "word_count": len(md_text.split(" ")),
            "sentence_spacy_count": sentence_count,
            "chunk_count": self.chunk_size,
            "token_count": len(md_text.split()) / 4,
            "text": md_text,
            "sentences": sentences,
            "sentence_chunks": chunked_sentences,
        }]
        return pl.DataFrame(self.pdf_content)

    def embed_chunks(self, embedding: LLMAPI) -> pl.DataFrame:
        if self.pdf_content is None:
            self.read_pdf()
        pages_and_chunks = []
        page = self.pdf_content[0]

        for sentence_chunk in tqdm(page["sentence_chunks"], desc="Embedding chunks"):

            chunk_dict = {}
            chunk_dict["page"] = page["page"]
            joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
            chunk_dict["chunk_token_count"] = chunk_dict["chunk_char_count"] / 4
            
            emb_array = embedding.embedding_text(joined_sentence_chunk)
            if emb_array is None:
                print("Error embedding text")
                print("Text: ", joined_sentence_chunk)
                continue
            chunk_dict["embedding"] = np.array(emb_array)
            pages_and_chunks.append(chunk_dict)
        self.pages_and_chunks = pages_and_chunks
        return pl.DataFrame(pages_and_chunks)

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
    def __init__(self, dict_prompt: dict = None):
        self.graph = nx.Graph()
        self.community_summaries =[]
        if dict_prompt is None:
            dict_prompt = {
                "community_report_generation_prompt": community_report_generation_prompt,
                "community_report_format_prompt": community_report_format_prompt,
                "community_report_example_prompt": community_report_example_prompt,
            }
        self.dict_prompt = dict_prompt

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
    
    def detect_communities(self):
        # Use the Louvain method to detect communities
        partition = community_louvain.best_partition(self.graph)
        # Assign communityID to each node
        for node, community_id in partition.items():
            self.graph.nodes[node]['communityID'] = community_id
        return partition

    def render_graph(self, graph=None):
        if graph is None:
            graph = self.graph
        pos = nx.spring_layout(graph)  # positions for all nodes

        # Extract node and edge information
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            # TODO this doesn show up in the hover text, try fixing it.
            edge_text.append(f"Source: {edge[0]}<br>Target: {edge[1]}<br>Description: {edge[2].get('description', 'N/A')}<br>Strength: {edge[2].get('strength', 'N/A')}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')

        node_x = []
        node_y = []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Rainbow',
                size=10,
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Community ID',
                    xanchor='left',
                    titleside='right'
                ),
            )
        )

        node_text = []
        node_color = []
        communities = nx.get_node_attributes(graph, 'communityID')
        for node in graph.nodes():
            description = graph.nodes[node].get('description', 'N/A')
            node_text.append(f'{node}<br>Description: {description}<br>Community: {communities.get(node, 0)}')
            node_color.append(communities.get(node, 0))

        node_trace.text = node_text
        node_trace.marker.color = node_color

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Graph Visualization with Communities',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Graph Visualization",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                        )
        fig.show()
    
    def drop_existing_community_reports(self):
        self.community_summaries = []

    def community_report_gen(self, llm:LLMAPI, max_retries:int=3):
        """
        Generates reports for each community using an LLM.

        :param llm: An instance of LLMAPI or similar class with an `invoke` method.
        :param max_retries: Maximum number of retries for correcting JSON summaries.
        :return: Dictionary with community IDs as keys and their summaries as values.
        """
        rejected_id = []
        # Get the community IDs
        communities = nx.get_node_attributes(self.graph, 'communityID')

        # Group nodes by community
        community_nodes = {}
        for node, community_id in communities.items():
            if community_id not in community_nodes:
                community_nodes[community_id] = []
            community_nodes[community_id].append(node)

        # Loop through each community
        for community_id, nodes in tqdm(community_nodes.items(), desc="Generating community reports"):
            # Create a subgraph for the community
            subgraph = self.graph.subgraph(nodes)

            # Generate a summary of the subgraph using LLM
            summary, subgraph_str = self.summarize_subgraph(subgraph, llm, max_retries)

            if summary == 'timeout':
                logger.error(f"LLM invocation timed out for community {community_id}. Skipping.")
                self.community_summaries[community_id] = {"error": "Summary unavailable due to timeout."}
                continue
                # check if the summary can be extract as json from string

            json_summary = self._parse_summary_json(summary, community_id)
            dict_summary = {
                "community_id": community_id,
                "summary": json_summary,
                "subgraph": subgraph_str
            }
            self.community_summaries.append(dict_summary)
            # check if json_summary has the key value "error"
            if "error" in json_summary:
                rejected_id.append(community_id)

        return self.community_summaries, rejected_id

    def regenerate_community_report(self, community_id, llm:LLMAPI, max_retries:int=1):
        """
        Regenerates a report for a specific community using an LLM.

        :param community_id: The ID of the community to regenerate the report for.
        :param llm: An instance of LLMAPI or similar class with an `invoke` method.
        :param max_retries: Maximum number of retries for correcting JSON summaries.
        :return: Dictionary with the community ID as the key and its summary as the value.
        """

        # Get the subgraph with the specified community ID
        community_nodes = [node for node, community in nx.get_node_attributes(self.graph, 'communityID').items() if community == community_id]
        subgraph = self.graph.subgraph(community_nodes)
        
        # Generate a summary of the subgraph using LLM
        summary, subgraph_str = self.summarize_subgraph(subgraph, llm, max_retries)

        if summary == 'timeout':
            logger.error(f"LLM invocation timed out for community {community_id}. Skipping.")
            self.community_summaries[community_id] = {"error": "Summary unavailable due to timeout."}

        # check if the summary can be extract as json from string

        json_summary = self._parse_summary_json(summary, community_id)
        dict_summary = {
            "community_id": community_id,
            "summary": json_summary,
            "subgraph": subgraph_str
        }
        # remove the old summary
        self.community_summaries = [report for report in self.community_summaries if report['community_id'] != community_id]
        # add the new summary
        self.community_summaries.append(dict_summary)
        return dict_summary

    
    def summarize_subgraph(self, subgraph, llm, retries):
        # Convert subgraph to the specified string format
        node_lines = ["id,entity,description"]
        for i, (node, data) in enumerate(subgraph.nodes(data=True), start=1):
            node_lines.append(f"{i},{node},{data.get('description', 'N/A')}")

        edge_lines = ["id,source,target,description"]
        for i, (u, v, data) in enumerate(subgraph.edges(data=True), start=1):
            edge_lines.append(f"{i},{u},{v},{data.get('description', 'N/A')}")

        subgraph_str = "Entities:\n"+"\n".join(node_lines) + "\n\nRelationships:\n" + "\n".join(edge_lines)

        main_prompt = self.dict_prompt["community_report_generation_prompt"]
        formatted_prompt = main_prompt.format(community_report_format_prompt=self.dict_prompt["community_report_format_prompt"], community_report_example_prompt=self.dict_prompt["community_report_example_prompt"], input_text=subgraph_str)
        
        for attempt in range(retries):
            try:
                output = llm.invoke(formatted_prompt)
                return output, subgraph_str
            except TimeoutError:
                if attempt < retries-1:
                    continue
                else:
                    logger.error("LLM invocation timed out.")
                    return 'timeout'
            except Exception as e:
                logger.error(f"An error occurred with using llm to summarize subgraph: {e}")
                return 'timeout'

    
    def _parse_summary_json(self, summary, community_id):
        try:
            json_summary = json.loads(summary)
            logger.info(f"JSON parsed successfully for community {community_id}.")
            return json_summary
        except json.JSONDecodeError:
            logger.error(f"Decoder error when parse JSON for community {community_id}.")
            return {"error": summary}
        except Exception as e:
            logger.error(f"Error {e} occured when parse JSON for community {community_id}.")
            return {"error": summary}
        
    def load_community_reports(self, path:str):
        try:
            with open(path, 'r') as file:
                self.community_summaries = json.load(file)
            logger.info(f"Community reports loaded from {path}.")
        except Exception as e:
            logger.error(f"Error loading community reports: {e}")
            self.community_summaries = []        