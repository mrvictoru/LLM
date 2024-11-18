import numpy as np
import polars as pl
from textingest import GraphDataManager, PDFDocumentHandler, LLMAPI


from prompt import simple_query_answer_prompt

class Queryhandler:
    def __init__(self, graph_manager: GraphDataManager, pdf_handler: PDFDocumentHandler, embedding: LLMAPI, llm: LLMAPI):
        self.graph_manager = graph_manager
        self.PDFDocumentHandler = pdf_handler
        self.embedding = embedding
        self.llm = llm

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
    
    def vector_search_response(self, query: str, context_prompt:str = simple_query_answer_prompt):
        """
        Process a query and return a response based on vector search.

        Parameters:
            query (str): The query to process.

        Returns:
            str: The response to the query.
        """
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
