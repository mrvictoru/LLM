"""
train_model.py

Fine-tunes a smaller LLM (e.g., Llama-3.2-3B-Instruct or Mistral-3B) on the
website-generation dataset produced by generate_training_data.py.

Uses Unsloth for fast QLoRA fine-tuning and HuggingFace TRL's SFTTrainer.

``model_name`` accepts either a HuggingFace Hub model id **or** a path to a
local HuggingFace-format model directory.  When a local path is supplied the
directory is validated (existence, config.json, weight files) before any GPU
memory is allocated.

After fine-tuning the LoRA adapter can optionally be exported to GGUF format
(for use with llama.cpp / Ollama) via the standalone ``save_gguf()`` function.

Usage:
    python train_model.py \
        --model_name   unsloth/Llama-3.2-3B-Instruct \
        --dataset_path website_dataset.jsonl \
        --output_dir   ./outputs/lora_model \
        --epochs       3 \
        --batch_size   2 \
        --grad_accum   4

    # Export the trained adapter to GGUF afterwards:
    python train_model.py \
        --model_name   unsloth/Llama-3.2-3B-Instruct \
        --output_dir   ./outputs/lora_model \
        --save_gguf \
        --gguf_quant   q4_k_m
"""

import argparse
import json
import os

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

# ---------------------------------------------------------------------------
# Defaults tuned for RTX 2080 Ti (22 GB VRAM) + 3-4 B parameter models
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Weight file extensions recognised as valid HuggingFace model files
_LOCAL_MODEL_WEIGHT_EXTENSIONS = frozenset({".safetensors", ".bin"})

# GGUF quantization methods supported by Unsloth
GGUF_QUANT_METHODS = [
    "q4_k_m",   # recommended: good balance of size and quality
    "q5_k_m",
    "q8_0",
    "f16",
    "q4_0",
    "q5_0",
    "q2_k",
    "q3_k_m",
]

SYSTEM_PROMPT = (
    "You are an expert web developer. "
    "When asked to build a website, you produce a single self-contained HTML file "
    "that includes all CSS and JavaScript inline. "
    "Output ONLY the raw HTML — no explanations, no markdown fences."
)


# ---------------------------------------------------------------------------
# Local model validation helpers
# ---------------------------------------------------------------------------

def validate_local_model(path: str) -> None:
    """Validate that *path* points to a loadable HuggingFace model directory.

    Checks (in order):
    1. The path exists on disk.
    2. It is a directory (Unsloth requires HF-format directories, not single
       files such as GGUF blobs).
    3. It contains ``config.json`` (mandatory for every HF model).
    4. It contains at least one ``.safetensors`` or ``.bin`` weight file.

    Args:
        path: Local filesystem path to validate.

    Raises:
        FileNotFoundError:  The path does not exist.
        NotADirectoryError: The path exists but is not a directory.
        ValueError:         The directory is missing ``config.json`` or has no
                            recognised weight files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Local model path does not exist: {path!r}. "
            "Provide a valid directory path or a HuggingFace Hub model id."
        )
    if not os.path.isdir(path):
        raise NotADirectoryError(
            f"Local model path is not a directory: {path!r}. "
            "Unsloth requires a HuggingFace-format model directory, not a "
            "single file (e.g. a GGUF blob cannot be used for training)."
        )
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise ValueError(
            f"No config.json found in {path!r}. "
            "The directory does not appear to contain a HuggingFace model."
        )
    weight_files = [
        f for f in os.listdir(path)
        if os.path.splitext(f)[1] in _LOCAL_MODEL_WEIGHT_EXTENSIONS
    ]
    if not weight_files:
        raise ValueError(
            f"No model weight files (.safetensors or .bin) found in {path!r}. "
            "The directory does not appear to contain a valid HuggingFace model."
        )


def _resolve_model_name(model_name: str) -> str:
    """Return *model_name* unchanged, but validate it first when it is a local path.

    A value is treated as a local path when ``os.path.exists()`` returns True.
    HuggingFace Hub ids (e.g. ``unsloth/Llama-3.2-3B-Instruct``) are passed
    through untouched so that Unsloth / HF Hub can resolve them normally.

    Args:
        model_name: HuggingFace Hub model id or local directory path.

    Returns:
        The same *model_name* string (validated if it was a local path).
    """
    if os.path.exists(model_name):
        validate_local_model(model_name)
        print(f"Using local model weights: {os.path.abspath(model_name)}")
    return model_name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_sample(record: dict, tokenizer) -> dict:
    """Convert a flat {instruction, input, output} record to a chat template string."""
    user_content = record["instruction"]
    if record.get("input"):
        user_content = f"{user_content}\n\n{record['input']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ---------------------------------------------------------------------------
# Public API (importable from notebook or other scripts)
# ---------------------------------------------------------------------------

def train(
    model_name: str = DEFAULT_MODEL,
    dataset_path: str = "website_dataset.jsonl",
    output_dir: str = "./outputs/lora_model",
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 2e-4,
    warmup_ratio: float = 0.05,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    save_merged: bool = False,
    push_to_hub: str | None = None,
) -> None:
    """Fine-tune a model using Unsloth + TRL SFTTrainer.

    Args:
        model_name:     HuggingFace Hub model id **or** path to a local
                        HuggingFace-format model directory.  Local paths are
                        validated (existence, config.json, weight files) before
                        any GPU memory is allocated.
        dataset_path:   Path to the JSONL training dataset.
        output_dir:     Directory to save the LoRA adapter.
        epochs:         Number of training epochs.
        batch_size:     Per-device training batch size.
        grad_accum:     Gradient accumulation steps.
        lr:             Learning rate.
        warmup_ratio:   Fraction of steps used for LR warm-up.
        max_seq_length: Maximum sequence length for training.
        load_in_4bit:   Whether to load the model in 4-bit quantisation.
        save_merged:    Also save a merged 16-bit model (needs more VRAM/disk).
        push_to_hub:    Optional HuggingFace repo id to push the adapter to.
    """
    # ------------------------------------------------------------------
    # 0. Validate model source (local path or HF Hub id)
    # ------------------------------------------------------------------
    model_name = _resolve_model_name(model_name)

    # ------------------------------------------------------------------
    # 1. Load base model with Unsloth
    # ------------------------------------------------------------------
    print(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # ------------------------------------------------------------------
    # 2. Apply LoRA
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 3. Prepare dataset
    # ------------------------------------------------------------------
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run generate_training_data.py first."
        )

    raw_records = load_jsonl(dataset_path)
    print(f"Loaded {len(raw_records)} training samples.")

    formatted = [format_sample(r, tokenizer) for r in raw_records]
    hf_dataset = Dataset.from_list(formatted)

    # ------------------------------------------------------------------
    # 4. Configure and run SFTTrainer
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=training_args,
    )

    print("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 5. Save the LoRA adapter
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")

    if save_merged:
        merged_dir = output_dir + "_merged_16bit"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Merged 16-bit model saved to {merged_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print(f"Adapter pushed to HuggingFace Hub: {push_to_hub}")


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------

def save_gguf(
    adapter_dir: str,
    output_dir: str | None = None,
    quantization_method: str = "q4_k_m",
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    push_to_hub: str | None = None,
) -> str:
    """Export a fine-tuned LoRA adapter to GGUF format for llama.cpp / Ollama.

    Loads the base model (identified by *model_name*) together with the LoRA
    adapter saved in *adapter_dir*, then calls Unsloth's
    ``save_pretrained_gguf`` to produce a quantised GGUF file.

    Args:
        adapter_dir:         Path to the LoRA adapter directory produced by
                             :func:`train`.
        output_dir:          Directory to write the GGUF file into.  Defaults
                             to ``<adapter_dir>_gguf``.
        quantization_method: GGUF quantization method.  Defaults to
                             ``"q4_k_m"`` (recommended: good quality/size
                             balance).  Valid options: ``q4_k_m``, ``q5_k_m``,
                             ``q8_0``, ``f16``, ``q4_0``, ``q5_0``, ``q2_k``,
                             ``q3_k_m``.
        model_name:          HuggingFace Hub model id **or** local path of the
                             base model used during training.  Local paths are
                             validated before GPU memory is allocated.
        max_seq_length:      Maximum sequence length (should match training).
        load_in_4bit:        Whether to load the base model in 4-bit.
        push_to_hub:         Optional HuggingFace repo id to upload the GGUF
                             file after export.

    Returns:
        Absolute path to the directory containing the exported GGUF file.

    Raises:
        ValueError:        *quantization_method* is not in
                           :data:`GGUF_QUANT_METHODS`.
        FileNotFoundError: *adapter_dir* does not exist or *model_name* is a
                           local path that does not exist.
        NotADirectoryError: A local *model_name* path is not a directory.
    """
    if quantization_method not in GGUF_QUANT_METHODS:
        raise ValueError(
            f"Unknown quantization_method {quantization_method!r}. "
            f"Supported methods: {GGUF_QUANT_METHODS}"
        )

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir!r}. "
            "Run train() first to produce a LoRA adapter."
        )

    if output_dir is None:
        output_dir = adapter_dir + "_gguf"
    os.makedirs(output_dir, exist_ok=True)

    # Validate local model path when supplied
    model_name = _resolve_model_name(model_name)

    print(f"Loading base model: {model_name}")
    print(f"Loading adapter   : {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LoRA adapter from {adapter_dir!r}: {exc}"
        ) from exc

    print(f"Exporting to GGUF ({quantization_method}) → {output_dir}")
    try:
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization_method)
    except Exception as exc:
        raise RuntimeError(
            f"GGUF export failed: {exc}\n"
            "Ensure Unsloth is up to date and llama.cpp is available."
        ) from exc

    print(f"GGUF model saved to {output_dir}")

    if push_to_hub:
        model.push_to_hub_gguf(push_to_hub, tokenizer, quantization_method=quantization_method)
        print(f"GGUF pushed to HuggingFace Hub: {push_to_hub}")

    return os.path.abspath(output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM with Unsloth + TRL SFTTrainer.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL,
                        help="HuggingFace Hub model id or path to a local HF model directory")
    parser.add_argument("--dataset_path", default="website_dataset.jsonl", help="Training data JSONL")
    parser.add_argument("--output_dir", default="./outputs/lora_model", help="Where to save the adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Use 4-bit quantisation (recommended for 22 GB VRAM)")
    parser.add_argument("--save_merged", action="store_true", default=False,
                        help="Also save a merged 16-bit model (requires more VRAM/disk)")
    parser.add_argument("--push_to_hub", default=None, help="Optional HF repo id to push the adapter to")
    # GGUF export options
    parser.add_argument("--save_gguf", action="store_true", default=False,
                        help="Export the fine-tuned adapter to GGUF after training")
    parser.add_argument("--gguf_output_dir", default=None,
                        help="Directory for the GGUF file (defaults to <output_dir>_gguf)")
    parser.add_argument("--gguf_quant", default="q4_k_m",
                        choices=GGUF_QUANT_METHODS,
                        help="GGUF quantization method (default: q4_k_m)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        save_merged=args.save_merged,
        push_to_hub=args.push_to_hub,
    )
    if args.save_gguf:
        save_gguf(
            adapter_dir=args.output_dir,
            output_dir=args.gguf_output_dir,
            quantization_method=args.gguf_quant,
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )


if __name__ == "__main__":
    main()
