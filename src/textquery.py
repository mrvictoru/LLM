import numpy as np
import polars as pl
import json
import logging
# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from textingest import GraphDataManager, PDFDocumentHandler, LLMAPI

from prompt import simple_query_answer_prompt, map_global_search_prompt, map_response_format_prompt, map_response_example_prompt, reduce_global_search_prompt

class Queryhandler:
    def __init__(self, graph_manager: GraphDataManager, pdf_handler: PDFDocumentHandler, embedding: LLMAPI, llm: LLMAPI, dict_prompt:dict = None):
        self.graph_manager = graph_manager
        self.PDFDocumentHandler = pdf_handler
        self.embedding = embedding
        self.llm = llm
        if dict_prompt is None:
            dict_prompt = {
                "simple_query_answer_prompt": simple_query_answer_prompt,
                "map_global_search_prompt": map_global_search_prompt,
                "map_response_format_prompt": map_response_format_prompt,
                "map_response_example_prompt": map_response_example_prompt,
                "reduce_global_search_prompt": reduce_global_search_prompt
            }
        self.dict_prompt = dict_prompt

    def _find_similar_chunk_np(self, prompt_embedding: np.ndarray, threshold=0.5):
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
        pages_and_chunks = pl.DataFrame(self.pdf_handler.get_pages_and_chunks)
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
    
    def vector_search_response(self, query: str):
        """
        Process a query and return a response based on vector search.

        Parameters:
            query (str): The query to process.

        Returns:
            str: The response to the query.
        """
        context_prompt = self.dict_prompt["simple_query_answer_prompt"]
        # Embed the query
        query_embedding = self.embedding.embedding_text(query)
        # Find similar chunks
        similar_chunks = self._find_similar_chunk_np(query_embedding)
        # Get the top sentence chunk
        context = similar_chunks["sentence_chunk"][0]
        # formate the prompt
        formatted_prompt = context_prompt.format(context = context, query = query)
        # Get the response
        response = self.llm.invoke(formatted_prompt)
        return response, similar_chunks.head(3)

    #TODO: implement GraphRAG local search and global search
    def _map_intermediate_response(self, query: str, threshold: int = 0.6):
        rated_inter_responses = []
        # loop through each community summaries (in self.graph_manager.community_summaries) and use map_global_search_prompt to get the intermediate response
        for report in self.graph_manager.community_summaries:
            summary = str(report['summary'])
            # Get the response
            response = self.llm.invoke(self.dict_prompt["map_global_search_prompt"].format(summary=summary, query=query))
            # read the response as json and check if the response is an empty list
            try:
                json_response = json.loads(response)
                if json_response['point']:
                    rated_inter_responses.extend(json_response['point'])
                    logging.info(f"Response lodge for community {report['community_id']}")
            except Exception as e:
                logging.error(f"Error from community {report['community_id']} in response: {e}")
        # sort the rated_inter_responses by the score in descending order
        rated_inter_responses = sorted(rated_inter_responses, key=lambda x: x['score'], reverse=True)
        # filter the responses to only include those with a score higher than a threshold
        return [response["description"] for response in rated_inter_responses if response['score'] > threshold]
    
    def _reduce_intermediate_responses(self, query:str, intermediate_responses: list, response_type: str = "medium_length"):
        response = self.llm.invoke(self.dict_prompt["reduce_global_search_prompt"].format(report_data=intermediate_responses, user_query=query, response_type=response_type))
        return response

    def graph_global_search_response(self, query: str, threshold: int = 0.6):
        """
        Process a query and return a response based on graph global search.

        Parameters:
            query (str): The query to process.

        Returns:
            str: The response to the query.
        """
        # get the intermediate responses against the query and all the community summaries
        intermediate_responses = self._map_intermediate_response(query, threshold)
        # reduce the intermediate responses to a single response



            