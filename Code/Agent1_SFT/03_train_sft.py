import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ================= Configuration =================
MODEL_PATH = "/home/ubuntu/finqa_sft/models/Qwen/Qwen2___5-3B-Instruct"
DATA_PATH = "/home/ubuntu/finqa_sft/data/finqa_cot_sft_data_gpt4o.json"
OUTPUT_DIR = "/dev/shm/qwen-3b-sft"

INSTRUCTION = (
    "Please analyze the provided financial context and table to answer the question. "
    "You should first show your step-by-step reasoning process inside <think> tags, "
    "and then provide the final numeric value as the answer inside <answer> tags."
)

def formatting_prompts_func(example):
    text = (
        f"<|im_start|>user\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n\n"
        f"{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{example['model_cot']}<|im_end|>"
    )
    return text

def train():
    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 3. Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=3072,
        packing=False,
        
        per_device_train_batch_size=1,      
        gradient_accumulation_steps=16,      
        gradient_checkpointing=True,       
        
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        bf16=True,
        deepspeed="/home/ubuntu/finqa_sft/ds_config.json",
        
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        overwrite_output_dir=True,
        
        eval_strategy="no",
        report_to="none"
    )

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model=MODEL_PATH,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    # 5. Start Training
    print("--- Starting High-Speed SFT (4x 5090) ---")
    trainer.train()
    
    # 6. Final Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"--- Process Finished. Output saved to: {OUTPUT_DIR} ---")

if __name__ == "__main__":
    train()