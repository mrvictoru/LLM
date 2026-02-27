"""
generate_training_data.py

Generates a dataset of (instruction, response) pairs for single-file HTML website
creation by querying a capable model via the OpenRouter API.

Topics and instruction templates are loaded from a JSON prompts file
(default: prompts.json) so they can be edited without touching this script.

Usage:
    python generate_training_data.py \
        --api_key      YOUR_OPENROUTER_KEY \
        --model        anthropic/claude-3.5-sonnet \
        --n_samples    200 \
        --output       website_dataset.jsonl \
        --prompts_file prompts.json
"""

import argparse
import json
import os
import time
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# System prompt (shared with evaluate_model.py and train_model.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert web developer. "
    "When asked to build a website, you produce a single self-contained HTML file "
    "that includes all CSS and JavaScript inline. "
    "Output ONLY the raw HTML — no explanations, no markdown fences."
)


def load_prompts(prompts_file: str) -> tuple[list[str], list[str]]:
    """Load training topics and instruction templates from the prompts JSON file."""
    with open(prompts_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    topics = data["training_topics"]
    templates = data["instruction_templates"]
    if not topics:
        raise ValueError(f"No training_topics found in {prompts_file}")
    if not templates:
        raise ValueError(f"No instruction_templates found in {prompts_file}")
    return topics, templates


def build_prompt(topic: str, templates: list[str], template_idx: int = 0) -> str:
    tmpl = templates[template_idx % len(templates)]
    return tmpl.format(topic=topic)


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def query_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    retries: int = 3,
    backoff: float = 5.0,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            wait = backoff * (attempt + 1)
            print(f"  [attempt {attempt + 1}/{retries}] Error: {exc}. Retrying in {wait}s …")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Public API (importable from notebook or other scripts)
# ---------------------------------------------------------------------------

def generate_dataset(
    api_key: str,
    output: str = "website_dataset.jsonl",
    model: str = "qwen/qwen3-235b-a22b-thinking-2507",
    n_samples: int = 200,
    max_tokens: int = 50000,
    temperature: float = 0.7,
    delay: float = 1.0,
    prompts_file: str = "prompts.json",
) -> None:
    """Generate website HTML training data via OpenRouter and save to a JSONL file.

    Args:
        api_key:      OpenRouter API key.
        output:       Path of the output JSONL file.
        model:        OpenRouter model id to use as the teacher.
        n_samples:    Total number of training samples to generate.
        max_tokens:   Maximum tokens per model response.
        temperature:  Sampling temperature.
        delay:        Seconds to wait between API requests.
        prompts_file: Path to the JSON file with training_topics and instruction_templates.
    """
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. "
            "Pass api_key= explicitly or set the OPENROUTER_API_KEY environment variable."
        )

    topics, templates = load_prompts(prompts_file)
    print(f"Loaded {len(topics)} topics and {len(templates)} templates from {prompts_file}.")

    # Cycle through topics and templates to reach n_samples
    samples = []
    for i in range(n_samples):
        topic = topics[i % len(topics)]
        instruction = build_prompt(topic, templates, template_idx=i)
        samples.append(instruction)

    existing = set()
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                existing.add(obj["instruction"])
        print(f"Resuming — {len(existing)} samples already in {output}.")

    with open(output, "a", encoding="utf-8") as out_fh:
        for instruction in tqdm(samples, desc="Generating"):
            if instruction in existing:
                continue
            response = query_openrouter(
                prompt=instruction,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response:
                record = {
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                }
                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()
                existing.add(instruction)
            time.sleep(delay)

    print(f"Done. {len(existing)} samples saved to {output}.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate website HTML training data via OpenRouter.")
    parser.add_argument("--api_key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key")
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-thinking-2507", help="OpenRouter model id")
    parser.add_argument("--n_samples", type=int, default=200, help="Total samples to generate")
    parser.add_argument("--max_tokens", type=int, default=50000, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="website_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests")
    parser.add_argument("--prompts_file", default="prompts.json",
                        help="Path to JSON file with training_topics and instruction_templates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        api_key=args.api_key,
        output=args.output,
        model=args.model,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay,
        prompts_file=args.prompts_file,
    )


if __name__ == "__main__":
    main()
