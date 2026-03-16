"""
evaluate_model.py

Evaluates a (fine-tuned or base) smaller LLM on a handful of single-file website
generation prompts and saves the generated HTML files so you can open them in a
browser to visually inspect quality.

Test cases are loaded from a JSON prompts file (default: prompts.json) so they
can be edited without touching this script.

Usage:
    # Evaluate a base model
    python evaluate_model.py \
        --model_name  unsloth/Llama-3.2-3B-Instruct \
        --output_dir  ./eval_results/base

    # Evaluate a fine-tuned LoRA adapter
    python evaluate_model.py \
        --model_name   unsloth/Llama-3.2-3B-Instruct \
        --adapter_path ./outputs/lora_model \
        --output_dir   ./eval_results/finetuned

    # Use a custom prompts file
    python evaluate_model.py \
        --model_name   unsloth/Llama-3.2-3B-Instruct \
        --prompts_file prompts.json \
        --output_dir   ./eval_results/base
"""

import argparse
import json
import os

import torch

# ---------------------------------------------------------------------------
# System prompt (same as training)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert web developer. "
    "When asked to build a website, you produce a single self-contained HTML file "
    "that includes all CSS and JavaScript inline. "
    "Output ONLY the raw HTML — no explanations, no markdown fences."
)


def load_test_cases(prompts_file: str) -> list[dict]:
    """Load eval test cases from the prompts JSON file."""
    with open(prompts_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    test_cases = data["eval_test_cases"]
    if not test_cases:
        raise ValueError(f"No eval_test_cases found in {prompts_file}")
    # Return only the required fields so extra metadata keys in the JSON are ignored
    return [{"id": tc["id"], "instruction": tc["instruction"]} for tc in test_cases]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, adapter_path: str | None):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=40960,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_html(model, tokenizer, instruction: str, max_new_tokens: int = 20480) -> str:
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
# Public API (importable from notebook or other scripts)
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    output_dir: str = "./eval_results",
    adapter_path: str | None = None,
    max_new_tokens: int = 20480,
    prompts_file: str = "prompts.json",
) -> list[dict]:
    """Evaluate a model on website-generation test cases and save HTML outputs.

    Args:
        model_name:     HuggingFace model id or local path for the student model.
        output_dir:     Directory where generated HTML files and summary.json are saved.
        adapter_path:   Optional path to a LoRA adapter to apply before inference.
        max_new_tokens: Maximum tokens to generate per test case.
        prompts_file:   Path to the JSON file with eval_test_cases.

    Returns:
        List of result dicts with keys: id, output_file, char_count,
        starts_with_html, has_body, has_script_or_style.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_cases = load_test_cases(prompts_file)
    print(f"Loaded {len(test_cases)} eval test cases from {prompts_file}.")

    print(f"Loading model: {model_name}")
    if adapter_path:
        print(f"Applying adapter: {adapter_path}")
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_path)

    results = []
    for tc in test_cases:
        print(f"\n[{tc['id']}] Generating …")
        raw = generate_html(model, tokenizer, tc["instruction"], max_new_tokens=max_new_tokens)
        html = extract_html(raw)

        out_path = os.path.join(output_dir, f"{tc['id']}.html")
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

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on website generation test cases.")
    parser.add_argument("--model_name", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--output_dir", default="./eval_results", help="Directory to save HTML outputs")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompts_file", default="prompts.json",
                        help="Path to JSON file with eval_test_cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_name=args.model_name,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        prompts_file=args.prompts_file,
    )


if __name__ == "__main__":
    main()
