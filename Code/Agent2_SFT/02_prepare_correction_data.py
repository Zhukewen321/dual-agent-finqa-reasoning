import json
import os
import re
from pathlib import Path

# ================= Configuration =================
INPUT_ROLLOUT_PATH = "/autodl-fs/data/finqa_dpo/data/train_rollouts.json"
OUTPUT_FILTERED_PATH = "/autodl-fs/data/finqa_dpo/data/to_be_corrected.json"
# =================================================

def extract_answer(text):
    if not text: return None
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        raw_ans = matches[-1].strip()
        return re.sub(r'[^0-9.\-]', '', raw_ans)
    return "no_answer_tag"

def main():
    if not os.path.exists(INPUT_ROLLOUT_PATH):
        print(f"Error: {INPUT_ROLLOUT_PATH} not found.")
        return

    with open(INPUT_ROLLOUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_to_correct = []

    for item in data:
        correct_paths = [r for r in item['rollouts'] if r['score'] == 1.0]
        incorrect_paths = [r for r in item['rollouts'] if r['score'] == 0.0]

        # 1. Handle Correct Path (Take only one if exists)
        if correct_paths:
            final_to_correct.append({
                "id": item['id'],
                "context": item['context'],
                "question": item['question'],
                "gold_answer": item['gold_answer'],
                "model_prediction": correct_paths[0]['prediction'],
                "is_correct": True
            })

        # 2. Handle Incorrect Paths (Unique by Answer)
        seen_wrong_answers = set()
        for i_path in incorrect_paths:
            pred_ans = extract_answer(i_path['prediction'])
            
            if pred_ans not in seen_wrong_answers:
                final_to_correct.append({
                    "id": item['id'],
                    "context": item['context'],
                    "question": item['question'],
                    "gold_answer": item['gold_answer'],
                    "model_prediction": i_path['prediction'],
                    "is_correct": False
                })
                seen_wrong_answers.add(pred_ans)

    # Save the filtered results
    with open(OUTPUT_FILTERED_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_to_correct, f, indent=2, ensure_ascii=False)

    print(f"Filtering complete.")
    print(f"Total entries for GPT-4o processing: {len(final_to_correct)}")
    print(f"Saved to: {OUTPUT_FILTERED_PATH}")

if __name__ == "__main__":
    main()