# merge_standard_solutions.py
import json

# Load to_be_corrected.json (has wrong answers)
with open('/autodl-fs/data/finqa_dpo/data/to_be_corrected.json', 'r') as f:
    wrong_data = json.load(f)

# Load finqa_cot_sft_data_gpt4o.json (has correct solutions)
with open('/autodl-fs/data/finqa_grpo_rag/processed_data/finqa/finqa_cot_sft_data_gpt4o.json', 'r') as f:
    correct_data = json.load(f)

# Create lookup by id
correct_lookup = {item['id']: item for item in correct_data}

# Merge
merged = []
for item in wrong_data:
    if item['id'] in correct_lookup:
        merged_item = item.copy()
        merged_item['standard_solution'] = correct_lookup[item['id']]['model_cot']
        merged.append(merged_item)
    else:
        print(f"Warning: No standard solution for {item['id']}")
        merged.append(item)

# Save
with open('/autodl-fs/data/finqa_dpo/data/to_be_corrected_with_solutions.json', 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged)} samples")
print(f"Samples with standard solutions: {sum(1 for item in merged if 'standard_solution' in item)}")