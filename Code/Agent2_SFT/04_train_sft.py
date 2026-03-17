# -*- coding: utf-8 -*-
# train_sft_review.py
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# ================= Configuration =================
MODEL_PATH = "/autodl-fs/data/models/Qwen2.5-3B-Instruct/"
SOURCE_DATA_PATH = "/autodl-fs/data/finqa_dpo/data/to_be_corrected_with_solutions.json"
AUDIT_DATA_PATH = "/autodl-fs/data/finqa_dpo/data/final_sft_correction_data_with_standard.json"
OUTPUT_DIR = "/autodl-fs/data/models/qwen-3b-sft-review"
DEEPSPEED_CONFIG = "/autodl-fs/data/finqa_dpo/ds_config_zero2.json"

INSTRUCTION = (
    "Analyze the provided financial context and table to answer the question. "
    "A previous answer is given above - review it for correctness. "
    "Use <review>...</review> to analyze errors or confirm correctness. "
    "Use <think>...</think> for step-by-step reasoning. "
    "Use <answer>...</answer> for the final numeric value."
)

def prepare_dataset():
    with open(AUDIT_DATA_PATH, 'r', encoding='utf-8') as f:
        audits = json.load(f)
    
    with open(SOURCE_DATA_PATH, 'r', encoding='utf-8') as f:
        sources = {item['id']: item for item in json.load(f)}
    
    formatted_list = []
    for audit in audits:
        item_id = audit['id']
        if item_id in sources:
            formatted_list.append({
                "context": sources[item_id]['context'],
                "question": sources[item_id]['question'],
                "model_prediction": sources[item_id]['model_prediction'],
                "gpt_response": audit['gpt4o_response']
            })
    
    return Dataset.from_list(formatted_list)

def formatting_prompts_func(example):
    text = (
        f"<|im_start|>user\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n\n"
        f"Previous Answer:\n{example['model_prediction']}\n\n"
        f"{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{example['gpt_response']}<|im_end|>"
    )
    return text

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    print("Model loaded successfully")
    
    dataset = prepare_dataset()
    print(f"Total samples for SFT: {len(dataset)}")
    
    print("\n" + "="*80)
    print("COMPLETE FIRST TRAINING SAMPLE")
    print("="*80)
    sample_text = formatting_prompts_func(dataset[0])
    print(sample_text)
    print("="*80 + "\n")
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=4096,
        packing=False,
        
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        
        learning_rate=1e-6,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        bf16=True,
        bf16_full_eval=True,
        deepspeed=DEEPSPEED_CONFIG,
        
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=1,
        overwrite_output_dir=True,
        
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=4,
        seed=42
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )
    
    print("="*60)
    print("Starting SFT Training (Review Format)")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: 3")
    print(f"Effective batch size: {4 * 3 * 8}")
    print(f"Steps per epoch: ~{len(dataset) // (4 * 3 * 8)}")
    print(f"Total steps: ~{3 * len(dataset) // (4 * 3 * 8)}")
    print("="*60 + "\n")
    
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()