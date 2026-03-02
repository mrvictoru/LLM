"""
generate_training_data.py

Generates a dataset of (instruction, response) pairs for single-file HTML website
creation by querying a capable model via the OpenRouter API.

Topics and instruction templates are loaded from a JSON prompts file
(default: prompts.json) so they can be edited without touching this script.

Usage:
    python generate_training_data.py \
        --provider     openrouter \
        --api_key      YOUR_OPENROUTER_KEY \
        --model        qwen/qwen3-235b-a22b-thinking-2507 \
        --n_samples    200 \
        --output       website_dataset.jsonl \
        --prompts_file prompts.json

    # Hugging Face Router example:
    python generate_training_data.py \
        --provider     huggingface \
        --api_key      YOUR_HF_TOKEN \
        --model        deepseek-ai/DeepSeek-R1-0528:together \
        --n_samples    200 \
        --output       website_dataset.jsonl \
        --prompts_file prompts.json
"""

import argparse
import json
import os
import random
import re
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


def extract_html(raw: str) -> str:
    """Strip markdown code fences if the model wrapped its output."""
    if "```html" in raw:
        raw = raw.split("```html", 1)[1]
        raw = raw.split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1]
        raw = raw.split("```", 1)[0]
    return raw.strip()


def _safe_filename(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if not cleaned:
        cleaned = "sample"
    return cleaned[:max_len]


def export_jsonl_outputs_to_html(
    jsonl_path: str,
    html_output_dir: str,
    output_field: str = "output",
    html_path_field: str = "output_file",
    index_output_path: str | None = None,
) -> str:
    """Export outputs from a JSONL dataset to HTML files and write a path-index JSONL.

    Args:
        jsonl_path:        Source JSONL file containing model outputs.
        html_output_dir:   Directory to store generated HTML files.
        output_field:      Record key containing raw HTML/text output.
        html_path_field:   Key name to store output HTML path in index records.
        index_output_path: Optional path for the index JSONL.

    Returns:
        Path to the generated index JSONL file.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    os.makedirs(html_output_dir, exist_ok=True)
    index_output_path = index_output_path or f"{jsonl_path}.html_index.jsonl"

    exported = 0
    with open(jsonl_path, "r", encoding="utf-8") as in_fh, open(
        index_output_path, "w", encoding="utf-8"
    ) as idx_fh:
        for line_num, line in enumerate(in_fh, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            instruction = record.get("instruction", "")
            raw_output = record.get(output_field)
            if not raw_output:
                continue

            html = extract_html(str(raw_output))
            file_stem = _safe_filename(instruction or f"sample-{line_num}")
            file_name = f"{line_num:05d}_{file_stem}.html"
            out_path = os.path.join(html_output_dir, file_name)

            with open(out_path, "w", encoding="utf-8") as html_fh:
                html_fh.write(html)

            idx_record = {
                "instruction": instruction,
                "input": record.get("input", ""),
                html_path_field: out_path,
            }
            idx_fh.write(json.dumps(idx_record) + "\n")
            exported += 1

    print(f"Exported {exported} HTML files to {html_output_dir}")
    print(f"Wrote HTML index JSONL: {index_output_path}")
    return index_output_path


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HUGGINGFACE_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def _compute_retry_wait(resp: requests.Response | None, attempt: int, backoff: float) -> float:
    """Use Retry-After when available, otherwise exponential backoff + jitter."""
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass

    base_wait = backoff * (2 ** attempt)
    jitter = random.uniform(0.0, 1.0)
    return min(base_wait + jitter, 120.0)


def query_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    retries: int = 6,
    backoff: float = 2.0,
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
    non_retriable_statuses = {400, 401, 403, 404}
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)

            if resp.status_code == 429:
                wait = _compute_retry_wait(resp, attempt, backoff)
                print(
                    f"  [attempt {attempt + 1}/{retries}] 429 rate-limited. "
                    f"Retrying in {wait:.1f}s ..."
                )
                time.sleep(wait)
                continue

            if resp.status_code in non_retriable_statuses:
                detail = ""
                try:
                    err = resp.json().get("error", {})
                    detail = err.get("message") or err.get("code") or ""
                except Exception:  # noqa: BLE001
                    detail = resp.text[:300]
                raise RuntimeError(
                    f"OpenRouter request failed with HTTP {resp.status_code}. "
                    f"This is usually non-retriable (e.g., invalid/unavailable model). "
                    f"Model='{model}'. Details: {detail}"
                )

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except RuntimeError:
            raise

        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            wait = _compute_retry_wait(resp, attempt, backoff)
            status = resp.status_code if resp is not None else "unknown"
            print(
                f"  [attempt {attempt + 1}/{retries}] HTTP {status}: {exc}. "
                f"Retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)

        except Exception as exc:  # noqa: BLE001
            wait = min(backoff * (2 ** attempt) + random.uniform(0.0, 1.0), 120.0)
            print(
                f"  [attempt {attempt + 1}/{retries}] Error: {exc}. "
                f"Retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)
    return None


def query_huggingface_router(
    prompt: str,
    hf_token: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    retries: int = 6,
    backoff: float = 2.0,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {hf_token}",
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
    non_retriable_statuses = {400, 401, 403, 404}
    for attempt in range(retries):
        try:
            resp = requests.post(HUGGINGFACE_ROUTER_URL, headers=headers, json=payload, timeout=180)

            if resp.status_code == 429:
                wait = _compute_retry_wait(resp, attempt, backoff)
                print(
                    f"  [attempt {attempt + 1}/{retries}] 429 rate-limited (HF Router). "
                    f"Retrying in {wait:.1f}s ..."
                )
                time.sleep(wait)
                continue

            if resp.status_code in non_retriable_statuses:
                detail = ""
                try:
                    err = resp.json().get("error", {})
                    if isinstance(err, dict):
                        detail = err.get("message") or err.get("code") or ""
                    else:
                        detail = str(err)
                except Exception:  # noqa: BLE001
                    detail = resp.text[:300]
                raise RuntimeError(
                    f"Hugging Face Router request failed with HTTP {resp.status_code}. "
                    f"This is usually non-retriable (e.g., invalid/unavailable model). "
                    f"Model='{model}'. Details: {detail}"
                )

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except RuntimeError:
            raise

        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            wait = _compute_retry_wait(resp, attempt, backoff)
            status = resp.status_code if resp is not None else "unknown"
            print(
                f"  [attempt {attempt + 1}/{retries}] HF Router HTTP {status}: {exc}. "
                f"Retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)

        except Exception as exc:  # noqa: BLE001
            wait = min(backoff * (2 ** attempt) + random.uniform(0.0, 1.0), 120.0)
            print(
                f"  [attempt {attempt + 1}/{retries}] HF Router error: {exc}. "
                f"Retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Public API (importable from notebook or other scripts)
# ---------------------------------------------------------------------------

def generate_dataset(
    api_key: str | None = None,
    output: str = "website_dataset.jsonl",
    model: str = "qwen/qwen3-235b-a22b-thinking-2507",
    n_samples: int = 200,
    max_tokens: int = 50000,
    temperature: float = 0.7,
    delay: float = 1.0,
    prompts_file: str = "prompts.json",
    provider: str = "openrouter",
    store_mode: str = "inline",
    html_output_dir: str | None = None,
) -> None:
    """Generate website HTML training data via OpenRouter and save to a JSONL file.

    Args:
        api_key:      Router API token. For OpenRouter this is OPENROUTER_API_KEY;
                      for Hugging Face Router this is HF_TOKEN.
        output:       Path of the output JSONL file.
        model:        OpenRouter model id to use as the teacher.
        n_samples:    Total number of training samples to generate.
        max_tokens:   Maximum tokens per model response.
        temperature:  Sampling temperature.
        delay:        Seconds to wait between API requests.
        prompts_file: Path to the JSON file with training_topics and instruction_templates.
        provider:     Inference provider: 'openrouter' or 'huggingface'.
        store_mode:   How outputs are stored: 'inline' (JSONL output text) or
                      'file_path' (write .html files and store output_file path).
        html_output_dir: Directory for HTML files when store_mode='file_path'.
    """
    provider = provider.lower().strip()
    if provider not in {"openrouter", "huggingface"}:
        raise ValueError("provider must be 'openrouter' or 'huggingface'.")

    store_mode = store_mode.lower().strip()
    if store_mode not in {"inline", "file_path"}:
        raise ValueError("store_mode must be 'inline' or 'file_path'.")

    if provider == "openrouter":
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required when provider='openrouter'. "
                "Pass api_key= explicitly or set OPENROUTER_API_KEY."
            )
    else:
        api_key = api_key or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "Hugging Face token is required when provider='huggingface'. "
                "Pass api_key= explicitly or set HF_TOKEN."
            )

    topics, templates = load_prompts(prompts_file)
    print(f"Loaded {len(topics)} topics and {len(templates)} templates from {prompts_file}.")

    if store_mode == "file_path":
        if not html_output_dir:
            base_dir = os.path.dirname(os.path.abspath(output)) or "."
            html_output_dir = os.path.join(base_dir, "html_outputs")
        os.makedirs(html_output_dir, exist_ok=True)

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
        for sample_idx, instruction in enumerate(tqdm(samples, desc="Generating"), start=1):
            if instruction in existing:
                continue
            response = query_openrouter(
                prompt=instruction,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            ) if provider == "openrouter" else query_huggingface_router(
                prompt=instruction,
                hf_token=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response:
                html = extract_html(response)
                record = {
                    "instruction": instruction,
                    "input": "",
                }

                if store_mode == "inline":
                    record["output"] = html
                else:
                    file_stem = _safe_filename(instruction or f"sample-{sample_idx}")
                    file_name = f"{sample_idx:05d}_{file_stem}.html"
                    out_path = os.path.join(html_output_dir, file_name)
                    with open(out_path, "w", encoding="utf-8") as html_fh:
                        html_fh.write(html)
                    record["output_file"] = out_path

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
    parser.add_argument(
        "--provider",
        choices=["openrouter", "huggingface"],
        default="openrouter",
        help="Inference provider for generating teacher responses",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Router API token. Uses OPENROUTER_API_KEY for openrouter or HF_TOKEN for huggingface if omitted",
    )
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-thinking-2507", help="OpenRouter model id")
    parser.add_argument("--n_samples", type=int, default=200, help="Total samples to generate")
    parser.add_argument("--max_tokens", type=int, default=50000, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="website_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests")
    parser.add_argument(
        "--store_mode",
        choices=["inline", "file_path"],
        default="inline",
        help="Store output directly in JSONL or as HTML files with output_file paths",
    )
    parser.add_argument(
        "--html_output_dir",
        default=None,
        help="Directory for HTML files when --store_mode file_path (default: <output_dir>/html_outputs)",
    )
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
        provider=args.provider,
        store_mode=args.store_mode,
        html_output_dir=args.html_output_dir,
    )


if __name__ == "__main__":
    main()
