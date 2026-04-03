"""Direct LoRA fine-tuning for Qwen3-4B on satellite MDP task.
Bypasses LlamaFactory — uses peft + transformers directly.
"""

import json
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Paths
BASE = Path(__file__).parent.parent
TRAIN_FILE = BASE / "data" / "finetune" / "train.json"
VAL_FILE = BASE / "data" / "finetune" / "val.json"
OUTPUT_DIR = BASE / "models" / "qwen3-4b-satcom-lora"
MODEL_ID = "Qwen/Qwen3-4B"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    # Format as chat messages
    formatted = []
    for item in raw:
        text = f"<|im_start|>system\n{item['instruction']}<|im_end|>\n<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


def main():
    print("Loading data...")
    train_ds = load_data(TRAIN_FILE)
    val_ds = load_data(VAL_FILE)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    print(f"Loading model {MODEL_ID} (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Done! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
