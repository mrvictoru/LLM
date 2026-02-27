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
# Main
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

    # ------------------------------------------------------------------
    # 1. Load base model with Unsloth
    # ------------------------------------------------------------------
    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
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
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Run generate_training_data.py first."
        )

    raw_records = load_jsonl(args.dataset_path)
    print(f"Loaded {len(raw_records)} training samples.")

    formatted = [format_sample(r, tokenizer) for r in raw_records]
    hf_dataset = Dataset.from_list(formatted)

    # ------------------------------------------------------------------
    # 4. Configure and run SFTTrainer
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        max_seq_length=args.max_seq_length,
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
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    if args.save_merged:
        merged_dir = args.output_dir + "_merged_16bit"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Merged 16-bit model saved to {merged_dir}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Adapter pushed to HuggingFace Hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
