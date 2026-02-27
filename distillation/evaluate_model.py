"""
evaluate_model.py

Evaluates a (fine-tuned or base) smaller LLM on a handful of single-file website
generation prompts and saves the generated HTML files so you can open them in a
browser to visually inspect quality.

Usage:
    # Evaluate a base model
    python evaluate_model.py \
        --model_name  unsloth/Llama-3.2-3B-Instruct \
        --output_dir  ./eval_results/base

    # Evaluate a fine-tuned LoRA adapter
    python evaluate_model.py \
        --model_name  unsloth/Llama-3.2-3B-Instruct \
        --adapter_path ./outputs/lora_model \
        --output_dir  ./eval_results/finetuned
"""

import argparse
import os

import torch
from transformers import TextStreamer

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert web developer. "
    "When asked to build a website, you produce a single self-contained HTML file "
    "that includes all CSS and JavaScript inline. "
    "Output ONLY the raw HTML — no explanations, no markdown fences."
)

TEST_CASES = [
    {
        "id": "todo_app",
        "instruction": "Create a single-file HTML website: a simple todo list app with add/delete functionality.",
    },
    {
        "id": "portfolio",
        "instruction": "Build a self-contained HTML page for a personal portfolio page for a software engineer.",
    },
    {
        "id": "pomodoro_timer",
        "instruction": "Write a complete single-file HTML + CSS + JS website that implements a Pomodoro productivity timer.",
    },
    {
        "id": "quiz_app",
        "instruction": "Generate a responsive single-file HTML website for an interactive quiz with score tracking. Include all styles and scripts inline.",
    },
    {
        "id": "snake_game",
        "instruction": "Produce a polished, self-contained HTML file that acts as a snake game in a canvas element.",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, adapter_path: str | None):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_html(model, tokenizer, instruction: str, max_new_tokens: int = 2048) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.6,
            do_sample=True,
        )

    generated = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_html(raw: str) -> str:
    """Strip markdown code fences if the model wrapped its output."""
    if "```html" in raw:
        raw = raw.split("```html", 1)[1]
        raw = raw.split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1]
        raw = raw.split("```", 1)[0]
    return raw.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on website generation test cases.")
    parser.add_argument("--model_name", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--output_dir", default="./eval_results", help="Directory to save HTML outputs")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    if args.adapter_path:
        print(f"Applying adapter: {args.adapter_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_path)

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] Generating …")
        raw = generate_html(model, tokenizer, tc["instruction"], max_new_tokens=args.max_new_tokens)
        html = extract_html(raw)

        out_path = os.path.join(args.output_dir, f"{tc['id']}.html")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        starts_with_html = html.lower().lstrip().startswith("<!doctype") or html.lower().lstrip().startswith("<html")
        has_body = "<body" in html.lower()
        has_script_or_style = ("<script" in html.lower()) or ("<style" in html.lower())

        results.append(
            {
                "id": tc["id"],
                "output_file": out_path,
                "char_count": len(html),
                "starts_with_html": starts_with_html,
                "has_body": has_body,
                "has_script_or_style": has_script_or_style,
            }
        )
        print(f"  Saved to {out_path}  ({len(html)} chars)")

    print("\n--- Evaluation Summary ---")
    for r in results:
        status = "✓" if (r["starts_with_html"] and r["has_body"]) else "✗"
        print(
            f"  {status} {r['id']:20s}  chars={r['char_count']:5d}  "
            f"html={r['starts_with_html']}  body={r['has_body']}  "
            f"script/style={r['has_script_or_style']}"
        )

    import json
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
