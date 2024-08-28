import json
from IPython.display import Image, display
import requests

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

# write a helper function to use embedding API to get embeddings of text
def embedding_text(text, url):
    url = url + '/embedding'
    data = {"input":text,}
    response = requests.post(url, json=data)
    api_data = response.json()

    return api_data["data"][0]["embedding"]