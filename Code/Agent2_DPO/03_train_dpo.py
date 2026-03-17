# -*- coding: utf-8 -*-
# train_dpo_review.py
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig

# ================= Configuration =================
MODEL_PATH = "/autodl-fs/data/models/qwen-3b-sft-review/"
DATA_PATH = "/autodl-fs/data/finqa_dpo/final_dpo_dataset_review.jsonl"
OUTPUT_DIR = "/autodl-fs/data/models/dpo_qwen_3b_review"
DEEPSPEED_CONFIG = "/autodl-fs/data/finqa_dpo/ds_config_zero2.json"

AUTODL_LOG_DIR = "/root/tf-logs"
os.makedirs(AUTODL_LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Callback =================
class SaveMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metric_file = os.path.join(args.output_dir, "metrics_history.jsonl")
            with open(metric_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(logs) + "\n")

# ================= Main =================
def main():
    print("="*60)
    print("DPO TRAINING - Agent2 Review Format")
    print("="*60)
    
    # 1. Load Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print(f"  -> Tokenizer loaded (vocab size: {len(tokenizer)})")

    # 2. Load Dataset
    print("\n[2/5] Loading dataset...")
    data_list = []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data_list.append({
                    "prompt": obj["prompt"],
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"]
                })
            except:
                continue
    
    dataset = Dataset.from_list(data_list)
    print(f"  -> Dataset loaded: {len(dataset)} pairs")

    # 3. Load Model
    print("\n[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    
    print(f"  -> Model loaded ({model.num_parameters() / 1e9:.2f}B parameters)")

    # 4. Training Configuration
    print("\n[4/5] Setting up training config...")
    
    num_gpus = 3
    per_device_batch = 4
    grad_accum = 8
    num_epochs = 3
    effective_batch = per_device_batch * num_gpus * grad_accum
    total_steps = (len(dataset) // effective_batch) * num_epochs
    
    print(f"  - Effective batch size: {effective_batch}")
    print(f"  - Total training steps: {total_steps}")
    print(f"  - Training epochs: {num_epochs}")
    
    training_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        beta=0.1,
        max_length=4096,
        max_prompt_length=3072,
        
        bf16=True,
        bf16_full_eval=True,
        
        deepspeed=DEEPSPEED_CONFIG,
        gradient_checkpointing=True,
        
        logging_dir=AUTODL_LOG_DIR,
        logging_steps=10,
        logging_first_step=True,
        report_to="tensorboard",
        
        save_strategy="epoch",
        save_total_limit=1,
        
        remove_unused_columns=False,
        dataloader_num_workers=4,
        
        run_name="dpo_review_agent2_3epochs",
        disable_tqdm=False,
        seed=42
    )
    
    print(f"  -> Config created")

    # 5. Initialize DPO Trainer
    print("\n[5/5] Initializing DPO Trainer...")
    print("  - Loading reference model...")
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[SaveMetricsCallback()]
    )
    
    print(f"  -> Trainer initialized")
    print("\n" + "="*60)
    print(f"Starting training for {num_epochs} epochs...")
    print("="*60)
    
    dpo_trainer.train()
    
    print("\n" + "="*60)
    print("Saving final model...")
    print("="*60)
    
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    dpo_trainer.save_state()
    
    print(f"\n-> Model saved to: {OUTPUT_DIR}")
    print(f"-> Metrics: {OUTPUT_DIR}/metrics_history.jsonl")
    print(f"-> TensorBoard: {AUTODL_LOG_DIR}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()