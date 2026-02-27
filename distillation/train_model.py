"""
train_model.py

Fine-tunes a smaller LLM (e.g., Llama-3.2-3B-Instruct or Mistral-3B) on the
website-generation dataset produced by generate_training_data.py.

Uses Unsloth for fast QLoRA fine-tuning and HuggingFace TRL's SFTTrainer.

Usage:
    python train_model.py \
        --model_name   unsloth/Llama-3.2-3B-Instruct \
        --dataset_path website_dataset.jsonl \
        --output_dir   ./outputs/lora_model \
        --epochs       3 \
        --batch_size   2 \
        --grad_accum   4
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

SYSTEM_PROMPT = (
    "You are an expert web developer. "
    "When asked to build a website, you produce a single self-contained HTML file "
    "that includes all CSS and JavaScript inline. "
    "Output ONLY the raw HTML — no explanations, no markdown fences."
)


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
        model_name:     Base model HuggingFace id or local path.
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
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM with Unsloth + TRL SFTTrainer.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Base model HF id or local path")
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


if __name__ == "__main__":
    main()
