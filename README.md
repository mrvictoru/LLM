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

---

## LLM Distillation / Fine-Tuning (Unsloth + HuggingFace)

Fine-tune a small LLM (e.g., Llama-3.2-3B-Instruct) on a website-generation task
using **Unsloth** QLoRA, a dataset generated from a powerful teacher model via
**OpenRouter**, and HuggingFace TRL.

### Requirements
- NVIDIA GPU (tested on RTX 2080 Ti, 22 GB VRAM)
- Docker + NVIDIA Container Toolkit
- An [OpenRouter](https://openrouter.ai/) API key (for data generation)

### Quick Start

```bash
# 1. Export your API keys
export OPENROUTER_API_KEY=<your-key>
export HF_TOKEN=<your-hf-token>      # optional, only needed to push to Hub

# 2. Start the distillation container
sudo docker compose -f docker-compose-distill.yml up

# 3. Open the demo notebook in your browser
#    http://localhost:8888  (token: llmdistill_4699)
#    Open: distillation_demo.ipynb
```

### Scripts (inside `distillation/`)

| Script | Description |
|---|---|
| `generate_training_data.py` | Queries a teacher model via OpenRouter to build a JSONL training dataset |
| `evaluate_model.py` | Runs the student model on test prompts and saves HTML files for visual inspection |
| `train_model.py` | Fine-tunes the student model with Unsloth QLoRA + TRL SFTTrainer |
| `distillation_demo.ipynb` | End-to-end demo notebook covering all steps above |

### Running scripts directly

```bash
# Step 1 — Generate training data
python distillation/generate_training_data.py \
    --api_key $OPENROUTER_API_KEY \
    --model   anthropic/claude-3.5-sonnet \
    --n_samples 200 \
    --output  distillation/website_dataset.jsonl

# Step 2 — Evaluate base model
python distillation/evaluate_model.py \
    --model_name unsloth/Llama-3.2-3B-Instruct \
    --output_dir distillation/eval_results/base

# Step 3 — Fine-tune
python distillation/train_model.py \
    --model_name   unsloth/Llama-3.2-3B-Instruct \
    --dataset_path distillation/website_dataset.jsonl \
    --output_dir   distillation/outputs/lora_model \
    --epochs 3

# Step 4 — Evaluate fine-tuned model
python distillation/evaluate_model.py \
    --model_name   unsloth/Llama-3.2-3B-Instruct \
    --adapter_path distillation/outputs/lora_model \
    --output_dir   distillation/eval_results/finetuned
```
