import json
from IPython.display import Image, display
import requests
import numpy as np

def get_api_key(selection: int) -> str:
    
    # read json file for api key
    try:
        with open('api_key.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('api_key.json not found')
        return None
    
    # get the selected key
    selected_key = list(data.keys())[selection]
    
    return data[selected_key]


def render_graph(graph):
    try:
        # display the graph as a PNG image
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        print('Could not render the graph.')
        print(Exception)
        pass

class LLMAPI:
    def __init__(self, url: str = "http://llama_server:8080"):
        self.url = url
        self.ping_url(url)
        
    def ping_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            print("URL is reachable")
        except requests.exceptions.RequestException as e:
            print("URL is not reachable")
            raise e
    
    # this function is used to call the embedding endpoint of the LLM
    def embedding_text(self, text: str, timeout: int = 60):
        url = self.url + '/embedding'
        data = {"input": text, "thread": 5}
        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            api_data = response.json()
            return api_data['data'][0]["embedding"]
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    # this function is used to call the completion endpoint of the LLM, not suitable for chat format
    # default timeout is 300 seconds
    def invoke(self, text: str, max_tokens: int = 4096, temperature: float = 0.2, timeout: int = 360):
        url = self.url + "/completion"
        data = {
            "prompt": text,
            "max_token": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            api_data = response.json()
            return api_data["content"]
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
        
def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray):
    """
    Calculate the cosine similarity between two embeddings.

    Parameters:
        embedding1 (np.ndarray): The first embedding.
        embedding2 (np.ndarray): The second embedding.

    Returns:
        float: The cosine similarity between the two embeddings.
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    

