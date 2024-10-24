import json
from IPython.display import Image, display
import requests
from prompt import check_duplicate_entities_prompt

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
    def embedding_text(self, text: str):
        url = self.url + '/embedding'
        data = {"content": text}
        response = requests.post(url, json=data)
        api_data = response.json()
        return api_data["embedding"]

    # this function is used to call the completion endpoint of the LLM, not suitable for chat format
    def invoke(self, text: str, max_tokens: int = 1042, temperature: float = 0.2):
        url = self.url + "/completion"
        data = {
            "prompt":text,
            "max token": max_tokens,
            "temperature":temperature
        }
        response = requests.post(url, json=data)
        api_data = response.json()
        return api_data["content"]
    
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

def normalize_text(text):
    return text.strip().lower()

def is_duplicate(entity1, entity2):
    return normalize_text(entity1["entity_name"]) == normalize_text(entity2["entity_name"]) and entity1["entity_type"] == entity2["entity_type"]
