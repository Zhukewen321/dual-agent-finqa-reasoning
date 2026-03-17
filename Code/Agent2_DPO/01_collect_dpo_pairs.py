# collect_dpo_pairs_review.py
import json
import os
import re
import random
import asyncio
from typing import List, Dict
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from collections import Counter

# ================= Configuration =================
MODEL_PATH = "/autodl-fs/data/models/qwen-3b-sft-review/"
SOURCE_DATA = "/autodl-fs/data/finqa_dpo/data/to_be_corrected_with_solutions.json"
OUTPUT_DIR = "/autodl-fs/data/finqa_dpo/full_output_review/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = "sk-JIyqCKChYEdL4DBa8CzJ9RZTjnEqVezazqPDERZOfGEX7SyF"
BASE_URL = "https://api.qingyuntop.top/v1"
AGENT_MODEL = "gpt-4o"

TIP_MAX = 3
SAMPLES_PER_STEP = 3
CONCURRENCY_LIMIT = 50

INSTRUCTION = (
    "Analyze the provided financial context and table to answer the question. "
    "A previous answer is given above - review it for correctness. "
    "Use <review>...</review> to analyze errors or confirm correctness. "
    "Use <think>...</think> for step-by-step reasoning. "
    "Use <answer>...</answer> for the final numeric value."
)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ================= Utilities =================
def extract_answer(text: str):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        ans = match.group(1).strip().replace(',', '').replace('$', '').replace('%', '')
        return ans
    return None

def is_correct(model_output: str, gold_answer: str):
    pred = extract_answer(model_output)
    if pred is None:
        return False
    try:
        gold = str(gold_answer).replace(',', '').replace('$', '').replace('%', '')
        return abs(float(pred) - float(gold)) < 1e-4
    except:
        return str(pred).lower() == str(gold_answer).lower()

def deduplicate_by_answer(resps: List[str]) -> List[str]:
    seen_answers = set()
    unique_resps = []
    for resp in resps:
        ans = extract_answer(resp)
        if ans not in seen_answers:
            unique_resps.append(resp)
            seen_answers.add(ans)
    return unique_resps

async def fetch_gpt4o_tip(sid: str, question_data: Dict, failed_resps: List[str], previous_tips: List[str]):
    """Generate tip with complete information including standard solution"""
    async with semaphore:
        prompt = (
            f"You are helping a student solve a financial question.\n"
            f"\n"
            f"=== Context ===\n"
            f"{question_data['context']}\n"
            f"\n"
            f"=== Question ===\n"
            f"{question_data['question']}\n"
            f"\n"
            f"=== Ground Truth Answer ===\n"
            f"{question_data['gold_answer']}\n"
            f"\n"
        )
        
        if 'standard_solution' in question_data:
            prompt += (
                f"=== STANDARD SOLUTION (correct approach) ===\n"
                f"{question_data['standard_solution']}\n"
                f"\n"
            )
        
        prompt += (
            f"=== Agent1's Original Wrong Answer ===\n"
            f"{question_data['model_prediction']}\n"
            f"\n"
            f"=== Agent2's Failed Review Attempts ===\n"
        )
        
        for i, resp in enumerate(failed_resps[:2], 1):
            prompt += f"Attempt {i}:\n{resp}\n\n"
        
        if previous_tips:
            prompt += "=== Previous Tips ===\n"
            for i, tip in enumerate(previous_tips, 1):
                prompt += f"{i}. {tip}\n"
            prompt += "\n"
        
        prompt += (
            f"=== Your Task ===\n"
            f"Agent2 reviewed Agent1 but is STILL wrong. "
            f"Compare Agent2's attempts to the standard solution. "
            f"Give ONE specific tip (1-2 sentences).\n"
            f"\n"
            f"Focus on:\n"
            f"- What specific step they're missing\n"
            f"- Which calculation is wrong\n"
            f"- A hint to match the standard approach\n"
            f"\n"
            f"Tip:"
        )
        
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200,
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"Failed tip for {sid}: {e}")
                await asyncio.sleep(2)
        
        return "Double-check the calculation steps against the standard approach."

# ================= Processing =================
def main():
    print("Loading data from:", SOURCE_DATA)
    with open(SOURCE_DATA, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Total samples loaded: {len(raw_data)}")
    
    active_samples = {
        item['id']: {
            "data": item,
            "tips": [],
            "history_rejected": [],
            "gold": item['gold_answer']
        } for item in raw_data
    }

    START_FROM_ROUND = 0
    all_dpo_pairs = []
    
    for round_idx in range(TIP_MAX + 1):
        step_file = f"{OUTPUT_DIR}dpo_pairs_step_{round_idx}.jsonl"
        if os.path.exists(step_file):
            print(f"Found existing: {step_file}")
            START_FROM_ROUND = round_idx + 1
            
            with open(step_file, 'r', encoding='utf-8') as f:
                all_dpo_pairs = []
                for line in f:
                    if line.strip():
                        all_dpo_pairs.append(json.loads(line))
            
            solved_ids = set(pair['id'] for pair in all_dpo_pairs)
            for sid in list(active_samples.keys()):
                if sid in solved_ids:
                    del active_samples[sid]
            
            print(f"  Loaded {len(all_dpo_pairs)} pairs, {len(solved_ids)} solved")
            print(f"  Remaining: {len(active_samples)}")
    
    if START_FROM_ROUND > 0:
        if START_FROM_ROUND > TIP_MAX:
            print("\nAll rounds completed!")
            return
        print(f"\nRESUMING FROM ROUND {START_FROM_ROUND}")

    print(f"\nLoading Agent2 SFT model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
    
    sampling_params = SamplingParams(
        n=SAMPLES_PER_STEP,
        temperature=0.7,
        max_tokens=1536,
        stop=["<|im_end|>"]
    )

    for round_idx in range(START_FROM_ROUND, TIP_MAX + 1):
        if not active_samples:
            break
            
        print(f"\n{'='*50}")
        print(f"STAGE: Tip-{round_idx} | Active: {len(active_samples)}")
        print("="*50)

        current_ids = list(active_samples.keys())
        
        prompts = []
        for sid in current_ids:
            agent1_pred = active_samples[sid]['data']['model_prediction']
            
            prompt = (
                f"<|im_start|>user\n"
                f"Context: {active_samples[sid]['data']['context']}\n"
                f"Question: {active_samples[sid]['data']['question']}\n\n"
                f"Previous Answer:\n{agent1_pred}\n\n"
            )
            
            if active_samples[sid]['tips']:
                tip_text = "\n".join([f"Tip {i+1}: {t}" for i, t in enumerate(active_samples[sid]['tips'])])
                prompt += f"{tip_text}\n\n"
            
            prompt += f"{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n"
            
            prompts.append(prompt)

        print(f"Generating {SAMPLES_PER_STEP} responses per question...")
        outputs = llm.generate(prompts, sampling_params)
        
        needs_tip_ids = []
        stats = Counter()
        all_correct_count = 0

        for i, output in enumerate(outputs):
            sid = current_ids[i]
            resps = [o.text for o in output.outputs]
            
            correct_ones = [r for r in resps if is_correct(r, active_samples[sid]['gold'])]
            wrong_ones = [r for r in resps if not is_correct(r, active_samples[sid]['gold'])]
            
            stats[f"{len(correct_ones)} Correct / {len(wrong_ones)} Wrong"] += 1

            if len(correct_ones) == SAMPLES_PER_STEP:
                all_correct_count += 1
                del active_samples[sid]
                continue
            
            if correct_ones:
                chosen = correct_ones[0]
                unique_wrong = deduplicate_by_answer(wrong_ones)
                
                for rej in unique_wrong:
                    all_dpo_pairs.append({
                        "id": sid,
                        "prompt": prompts[i],
                        "chosen": chosen,
                        "rejected": rej,
                        "stage": round_idx
                    })
                
                for prev_round_errs in active_samples[sid]['history_rejected']:
                    if prev_round_errs:
                        all_dpo_pairs.append({
                            "id": sid,
                            "prompt": prompts[i],
                            "chosen": chosen,
                            "rejected": random.choice(prev_round_errs),
                            "stage": round_idx
                        })
                
                del active_samples[sid]
            else:
                active_samples[sid]['history_rejected'].append(wrong_ones)
                needs_tip_ids.append(sid)

        print(f"\n[Round Tip-{round_idx} Summary]")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
        print(f"  All correct (discarded): {all_correct_count}")
        print(f"  Solved this round: {sum(v for k, v in stats.items() if '0 Correct' not in k) - all_correct_count}")
        print(f"  Total DPO pairs: {len(all_dpo_pairs)}")
        print(f"  Need next tip: {len(needs_tip_ids)}")

        step_file = f"{OUTPUT_DIR}dpo_pairs_step_{round_idx}.jsonl"
        with open(step_file, 'w', encoding='utf-8') as f:
            for p in all_dpo_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved: {step_file}")
        
        if round_idx < TIP_MAX and needs_tip_ids:
            print(f"\nCalling GPT-4o for {len(needs_tip_ids)} tips...")
            
            loop = asyncio.get_event_loop()
            tasks = [
                fetch_gpt4o_tip(
                    sid,
                    active_samples[sid]['data'],
                    active_samples[sid]['history_rejected'][-1],
                    active_samples[sid]['tips']
                ) for sid in needs_tip_ids
            ]
            
            new_tips = loop.run_until_complete(asyncio.gather(*tasks))
            
            for sid, tip in zip(needs_tip_ids, new_tips):
                active_samples[sid]['tips'].append(tip)
            
            print(f"? Generated {len(new_tips)} tips")

    if active_samples:
        print(f"\n{'='*50}")
        print("SAVING FAILED SAMPLES")
        print("="*50)
        print(f"Failed after Tip-3: {len(active_samples)}")
        
        failed_data = []
        for sid, sample_info in active_samples.items():
            all_wrong = []
            for round_errs in sample_info['history_rejected']:
                all_wrong.extend(round_errs)
            
            unique_wrongs = deduplicate_by_answer(all_wrong)
            
            failed_data.append({
                "id": sid,
                "context": sample_info['data']['context'],
                "question": sample_info['data']['question'],
                "gold_answer": sample_info['gold'],
                "model_prediction": sample_info['data']['model_prediction'],
                "all_wrong_responses": unique_wrongs[:1],
                "tips_tried": sample_info['tips']
            })
        
        failed_file = f"{OUTPUT_DIR}failed_samples_with_responses.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {failed_file}")

    print(f"\n{'='*50}")
    print("DPO DATA COLLECTION COMPLETE")
    print("="*50)
    print(f"Total pairs: {len(all_dpo_pairs)}")
    print(f"Solved: {len(raw_data) - len(active_samples)}")
    print(f"Failed: {len(active_samples)}")

if __name__ == "__main__":
    main()