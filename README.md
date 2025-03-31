This repo contain the docker compose files to start a local LLM server using Llama CPP, a jupyternotebook server and a sample SQL server to test local LLM development

Requirement:
- Have docker and docker compose installed on the system
- Download GUFF weight you want to use to ./models
- Have CUDA installed (since my system has CUDA device, if you dont have it you should tweak the docker config accordingly)

Startup:
- If you want to start a seperate Llama cpp local server from your python environment, run the following in your CLI:
sudo docker compose -f docker-compose-server.yml up

- To stop and kill the running containers:
sudo docker compose -f docker-compose-server.yml down -v

- if you want to start a python environment with llama-cpp-python installed, run the following:
sudo docker compose -f docker-compose-python.yml up

- **For the graphrag implementation, use the following:
sudo docker compose -f docker-compose-server-graph.yml up

Notebook:
testresponse.ipynb -> for testing response with ther server

agent_test.ipynb -> for test running ReAct agent

langchain_sql_test.ipynb -> for test running langchain TEXT2SQL workflow

testllmgraph.ipynb -> for test running the my custom index and query process (with vector search and graphrag global search)

testsmolagents.ipynb -> for test running huggingface smolagent library 
