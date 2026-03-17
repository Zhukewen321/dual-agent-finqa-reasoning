# collect_all_failed_and_getfinal_review.py
import json
import os

SFT_SOURCE_PATH = "/autodl-fs/data/finqa_dpo/data/final_sft_correction_data_with_standard.json"
FAILED_SAMPLES_PATH = "/autodl-fs/data/finqa_dpo/full_output_review/failed_samples_with_responses.json"
DPO_RESULT_PATH = "/autodl-fs/data/finqa_dpo/full_output_review/dpo_pairs_step_3.jsonl"
FINAL_CLEAN_DPO = "/autodl-fs/data/finqa_dpo/final_dpo_dataset_review.jsonl"

INSTRUCTION = (
    "Analyze the provided financial context and table to answer the question. "
    "A previous answer is given above - review it for correctness. "
    "Use <review>...</review> to analyze errors or confirm correctness. "
    "Use <think>...</think> for step-by-step reasoning. "
    "Use <answer>...</answer> for the final numeric value."
)

def main():
    print("="*50)
    print("MERGING DPO PAIRS WITH EXPERT RESCUE")
    print("="*50)
    
    success_pairs = []
    if os.path.exists(DPO_RESULT_PATH):
        print(f"\nLoading: {DPO_RESULT_PATH}")
        with open(DPO_RESULT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    success_pairs.append(json.loads(line))
        print(f"Loaded {len(success_pairs)} success pairs")
    else:
        print(f"Warning: {DPO_RESULT_PATH} not found!")
        return

    print(f"\nLoading: {SFT_SOURCE_PATH}")
    with open(SFT_SOURCE_PATH, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
        sft_lookup = {item['id']: item.get('gpt4o_response') for item in sft_data}
    
    print(f"Loaded {len(sft_lookup)} GPT-4o responses")

    rescue_pairs = []
    if os.path.exists(FAILED_SAMPLES_PATH):
        print(f"\nLoading: {FAILED_SAMPLES_PATH}")
        with open(FAILED_SAMPLES_PATH, 'r', encoding='utf-8') as f:
            failed_samples = json.load(f)
        
        print(f"Loaded {len(failed_samples)} failed samples")
        print("\nCreating Expert Rescue pairs...")
        
        for item in failed_samples:
            sid = item['id']
            
            if sid not in sft_lookup:
                continue
            
            if not item['all_wrong_responses']:
                continue
            
            prompt = (
                f"<|im_start|>user\n"
                f"Context: {item['context']}\n"
                f"Question: {item['question']}\n\n"
                f"Previous Answer:\n{item['model_prediction']}\n\n"
                f"{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n"
            )
            
            rescue_pairs.append({
                "id": sid,
                "prompt": prompt,
                "chosen": sft_lookup[sid],
                "rejected": item['all_wrong_responses'][0],
                "stage": "expert_rescue"
            })
        
        print(f"Created {len(rescue_pairs)} rescue pairs")
    else:
        print(f"\nWarning: {FAILED_SAMPLES_PATH} not found")

    final_dataset = success_pairs + rescue_pairs
    
    print(f"\n{'='*50}")
    print("FINAL DATASET SUMMARY")
    print("="*50)
    print(f"Success pairs: {len(success_pairs)}")
    print(f"Rescue pairs: {len(rescue_pairs)}")
    print(f"Total pairs: {len(final_dataset)}")

    print(f"\nSaving: {FINAL_CLEAN_DPO}")
    with open(FINAL_CLEAN_DPO, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print("MERGE COMPLETE")
    print("="*50)
    
    print("\nVerifying format...")
    with open(FINAL_CLEAN_DPO, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line.strip():
            sample = json.loads(first_line)
            has_review_chosen = '<review>' in sample.get('chosen', '')
            has_review_rejected = '<review>' in sample.get('rejected', '')
            has_prev_ans = 'Previous Answer:' in sample.get('prompt', '')
            
            print(f"  Prompt has 'Previous Answer': {has_prev_ans}")
            print(f"  Chosen has <review>: {has_review_chosen}")
            print(f"  Rejected has <review>: {has_review_rejected}")
            
            if has_prev_ans and has_review_chosen:
                print(f"  ? Format PASSED")
            else:
                print(f"  ? Format FAILED")

if __name__ == "__main__":
    main()