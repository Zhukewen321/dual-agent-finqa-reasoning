import asyncio
import aiohttp
import json
import re
import os
from tqdm.asyncio import tqdm

# ================= 配置区 =================
API_KEY = "sk-JIyqCKChYEdL4DBa8CzJ9RZTjnEqVezazqPDERZOfGEX7SyF"
BASE_URL = "https://api.qingyuntop.top/v1"
AGENT_MODEL = "gpt-4o"
INPUT_FILE = r"D:\FinQA-main\processed_train.json"
OUTPUT_FILE = r"/finqa_cot_sft_data_gpt4o.json"
CONCURRENCY_LIMIT = 50


# ==========================================

def clean_financial_value(val_str):
    """进一步确保答案是纯数字字符串"""
    if not val_str: return ""
    val_str = str(val_str).strip()
    # 处理括号负数 (100) -> -100
    if val_str.startswith('(') and val_str.endswith(')'):
        val_str = '-' + val_str[1:-1]
    # 移除货币符号、逗号、百分号和空格
    val_str = re.sub(r'[$,\s%]', '', val_str)
    return val_str


async def fetch_cot(session, semaphore, sample):
    async with semaphore:
        # 再次确保答案纯净
        clean_ans = clean_financial_value(sample['gold_answer'])

        prompt = f"""You are a professional financial analyst. Based on the provided context, help me generate a reasoning path (Chain of Thought).

[Context]:
{sample['context']}

[Question]:
{sample['question']}

[Reference Calculation Steps]:
{sample['gold_program']}
(Target Numerical Answer: {clean_ans})

[Requirements]:
1. Reasoning: Think step-by-step. Identify relevant data from the table/text, then show the arithmetic operations.
2. Answer Format: You MUST output the answer in two parts:
   - Reasoning process wrapped in <think>...</think> tags.
   - The final numerical value wrapped in <answer>...</answer> tags.
3. STRICT Numerical Rule: The content in <answer> must be a PURE NUMBER. 
   - NO symbols like '$', '%', or commas.
   - If the answer is '5%', output '5'.
   - The value must be EXACTLY: {clean_ans}

[Output Example]:
<think>
First, I find the revenue in 2022 is 1000 and in 2021 is 800. 
The growth is 1000 - 800 = 200. 
The growth rate is 200 / 800 = 0.25, which is 25%.
</think>
<answer>25</answer>
"""

        payload = {
            "model": AGENT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # 低随机性保证逻辑一致性
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(3):  # 失败重试 3 次
            try:
                async with session.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers,
                                        timeout=90) as response:
                    if response.status == 200:
                        result = await response.json()
                        cot_content = result['choices'][0]['message']['content']

                        # 返回格式化的样本
                        return {
                            "id": sample['id'],
                            "question": sample['question'],
                            "context": sample['context'],
                            "gold_answer": clean_ans,
                            "logic_reference": sample['gold_program'],
                            "model_cot": cot_content
                        }
                    else:
                        await asyncio.sleep(2)
            except Exception as e:
                if attempt == 2:
                    return {"id": sample['id'], "error": str(e)}
                await asyncio.sleep(2)


async def main():
    # 1. 读取预处理后的数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        samples = json.load(f)


    print(f"已加载 {len(samples)} 条数据，准备开始并发调用（限制: {CONCURRENCY_LIMIT}）...")

    # 2. 异步执行
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_cot(session, semaphore, s) for s in samples]

        results = []
        # 使用 tqdm 显示异步进度
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GPT-4o Generating"):
            res = await f
            results.append(res)

    # 3. 保存结果
    # 过滤掉有错误的记录
    final_data = [r for r in results if "error" not in r]
    error_count = len(results) - len(final_data)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成！")
    print(f"成功: {len(final_data)} 条")
    print(f"失败: {error_count} 条")
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())