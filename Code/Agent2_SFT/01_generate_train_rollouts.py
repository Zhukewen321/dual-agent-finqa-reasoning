import os
import re
import json
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path

# ================= Configuration =================
MODEL_PATH = "/autodl-fs/data/models/qwen_3b_step240"
INPUT_JSON_PATH = "/autodl-fs/data/finqa_dpo/data/cleaned_train.json"
OUTPUT_DIR = "/autodl-fs/data/finqa_dpo/data"

# Set n=3 to generate 3 different paths per question
# Temperature 0.7 allows for diversity (some correct, some incorrect)
SAMPLING_PARAMS = SamplingParams(
    n=3,
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,
    stop=["<|im_end|>", "<|endoftext|>"]
)
# =================================================

def extract_answer(text):
    """
    Extract the numeric content from the last <answer> tag.
    Removes currency symbols, commas, and other non-numeric characters.
    """
    if not text:
        return None
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        raw_ans = matches[-1].strip()
        # Keep only digits, dots, and minus signs
        clean_ans = re.sub(r'[^0-9.\-]', '', raw_ans)
        return clean_ans
    return None

def compute_score(pred_text, gold_answer):
    """
    Check if the extracted answer matches the gold answer numerically.
    """
    pred_ans = extract_answer(pred_text)
    if pred_ans is not None:
        try:
            if abs(float(pred_ans) - float(gold_answer)) < 1e-3:
                return 1.0
        except (ValueError, TypeError):
            pass
    return 0.0

def main():
    # 1. Check input file
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: Input file not found at {INPUT_JSON_PATH}")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 2. Load cleaned data
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 3. Construct Prompts using chat template
    prompts = []
    for s in samples:
        # Standard FinQA Instruction used during SFT/GRPO
        instruction_text = (
            f"Context: {s['context']}\n"
            f"Question: {s['question']}\n\n"
            "Please analyze the provided financial context and table to answer the question. "
            "You should first show your step-by-step reasoning process inside <think> tags, "
            "and then provide the final numeric value as the answer inside <answer> tags."
        )
        # Applying ChatML template manually
        formatted_prompt = f"<|im_start|>user\n{instruction_text}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(formatted_prompt)

    # 4. Initialize vLLM Engine
    print(f"Loading vLLM model from: {MODEL_PATH}")
    # Optimization: enforce_eager=True to avoid CUDA graph overhead in some environments
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=True
    )

    # 5. Execute Batch Inference
    print(f"Generating roll-outs for {len(samples)} samples (3 paths each)...")
    outputs = llm.generate(prompts, SAMPLING_PARAMS)

    # 6. Organize Results
    final_results = []
    for i, output in enumerate(outputs):
        m = samples[i]
        rollout_list = []
        
        for completion in output.outputs:
            pred_text = completion.text
            score = compute_score(pred_text, m['gold_answer'])
            rollout_list.append({
                "prediction": pred_text,
                "score": score
            })
            
        final_results.append({
            "id": m['id'],
            "context": m['context'],
            "question": m['question'],
            "gold_answer": m['gold_answer'],
            "rollouts": rollout_list
        })

    # 7. Save Final Output
    output_path = os.path.join(OUTPUT_DIR, "train_rollouts.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"Success! Saved {len(final_results)} samples with roll-outs to: {output_path}")

if __name__ == "__main__":
    main()