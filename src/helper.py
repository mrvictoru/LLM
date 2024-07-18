import json

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
