"""
generate_training_data.py

Generates a dataset of (instruction, response) pairs for single-file HTML website
creation by querying a capable model via the OpenRouter API.

Usage:
    python generate_training_data.py \
        --api_key  YOUR_OPENROUTER_KEY \
        --model    anthropic/claude-3.5-sonnet \
        --n_samples 200 \
        --output   website_dataset.jsonl
"""

import argparse
import json
import os
import time
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert web developer. "
    "When asked to build a website, you produce a single self-contained HTML file "
    "that includes all CSS and JavaScript inline. "
    "Output ONLY the raw HTML — no explanations, no markdown fences."
)

WEBSITE_TOPICS = [
    "a personal portfolio page for a software engineer",
    "a landing page for a mobile app",
    "a simple todo list app with add/delete functionality",
    "a countdown timer to a user-specified date",
    "a BMI calculator with a results display",
    "a flashcard quiz app for learning vocabulary",
    "a minimal blog homepage with three sample posts",
    "a responsive pricing page with three tiers",
    "a weather dashboard with a search bar (mock data)",
    "a Pomodoro productivity timer",
    "a dark-mode toggle demo page",
    "a simple image gallery with lightbox effect",
    "a contact form with client-side validation",
    "a currency converter (mock exchange rates)",
    "a markdown previewer",
    "a color palette generator",
    "a simple drawing canvas with color picker",
    "an interactive quiz with score tracking",
    "a recipe card page with ingredients and steps",
    "a music player UI (no audio required, mock UI)",
    "a kanban board with three columns (To Do, In Progress, Done)",
    "a login and register modal demo",
    "a typing speed test app",
    "a digital clock showing hours, minutes and seconds",
    "a progress bar animation demo",
    "a star rating component",
    "a responsive navbar with hamburger menu",
    "a testimonial carousel slider",
    "a snake game in a canvas element",
    "a tic-tac-toe game",
    "a memory card matching game",
    "a simple e-commerce product card grid",
    "a FAQ accordion page",
    "a modal dialog demo with overlay",
    "a sticky header that changes color on scroll",
    "a multi-step form wizard",
    "a skeleton loading screen demo",
    "a drag-and-drop sortable list",
    "a data table with sorting and filtering",
    "a line chart using only SVG (no external libraries)",
]

INSTRUCTION_TEMPLATES = [
    "Create a single-file HTML website: {topic}.",
    "Build a self-contained HTML page for {topic}.",
    "Write a complete single-file HTML + CSS + JS website that implements {topic}.",
    "Generate a responsive single-file HTML website for {topic}. Include all styles and scripts inline.",
    "Produce a polished, self-contained HTML file that acts as {topic}.",
]


def build_prompt(topic: str, template_idx: int = 0) -> str:
    tmpl = INSTRUCTION_TEMPLATES[template_idx % len(INSTRUCTION_TEMPLATES)]
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
# Main
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("OpenRouter API key is required. Pass --api_key or set OPENROUTER_API_KEY.")

    # Cycle through topics and templates to reach n_samples
    samples = []
    for i in range(args.n_samples):
        topic = WEBSITE_TOPICS[i % len(WEBSITE_TOPICS)]
        instruction = build_prompt(topic, template_idx=i)
        samples.append(instruction)

    output_path = args.output
    existing = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                existing.add(obj["instruction"])
        print(f"Resuming — {len(existing)} samples already in {output_path}.")

    with open(output_path, "a", encoding="utf-8") as out_fh:
        for instruction in tqdm(samples, desc="Generating"):
            if instruction in existing:
                continue
            response = query_openrouter(
                prompt=instruction,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
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
            time.sleep(args.delay)

    print(f"Done. {len(existing)} samples saved to {output_path}.")


if __name__ == "__main__":
    main()
