import os
import json
import re
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

# EXACTLY match the format from your train_sft.py
INSTRUCTION = (
    "Please analyze the provided financial context and table to answer the question. "
    "You should first show your step-by-step reasoning process inside <think> tags, "
    "and then provide the final numeric value as the answer inside <answer> tags."
)

def format_table(table):
    """Formats FinQA table to match SFT training style."""
    if not table or len(table) == 0:
        return ""
    lines = []
    headers = table[0]
    lines.append(" | ".join(str(h) for h in headers))
    lines.append("-" * 60)
    for row in table[1:]:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)

def format_context(sample):
    """Combines segments to match SFT training prompt context structure."""
    parts = []
    if sample.get('pre_text'):
        parts.append(" ".join([t.strip() for t in sample['pre_text'] if t.strip()]))
    if sample.get('table'):
        table_text = format_table(sample['table'])
        if table_text:
            parts.append(f"\nTable:\n{table_text}")
    if sample.get('post_text'):
        parts.append(" ".join([t.strip() for t in sample['post_text'] if t.strip()]))
    return "\n\n".join(parts)

def normalize_number(num_str):
    """Cleans ground truth for easy numerical comparison in GRPO rewards."""
    if num_str is None: return ""
    # Remove symbols that interfere with float() conversion
    s = str(num_str).replace('$', '').replace(',', '').replace('%', '').strip()
    # Extract the core numeric part
    match = re.search(r'-?\d+\.?\d*', s)
    return match.group(0) if match else s

def process_sample(sample, idx, split):
    """Constructs verl-compatible data with chat template alignment."""
    context = format_context(sample)
    question = sample['qa']['question']
    
    # Matching the SFT prompt structure: Context -> Question -> Instruction
    prompt_content = (
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        f"{INSTRUCTION}"
    )
    
    return {
        "data_source": "finqa",
        "prompt": [{"role": "user", "content": prompt_content}], # verl will apply template here
        "ability": "financial_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": normalize_number(sample['qa']['answer'])
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "id": sample.get('id', f"{split}_{idx}")
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="/autodl-fs/data/finqa_grpo_rag/raw_data/finqa")
    parser.add_argument("--output_dir", default="/autodl-fs/data/finqa_grpo_rag/processed_data/finqa")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(args.raw_data_dir, f"{split}.json")
        if not os.path.exists(input_file): continue
            
        print(f"Converting {split} to verl parquet...")
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        processed_data = [process_sample(s, i, split) for i, s in enumerate(tqdm(raw_data))]
        
        df = pd.DataFrame(processed_data)
        df.to_parquet(os.path.join(args.output_dir, f"{split}.parquet"), index=False)

if __name__ == "__main__":
    main()