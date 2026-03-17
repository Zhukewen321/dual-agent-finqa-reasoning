import json
import re


def clean_financial_value(val_str):
    """
    将金融字符串（如 $3,800, 5%, (100)）转换为纯数字字符串
    """
    if val_str is None:
        return ""

    # 1. 处理括号表示的负数: (100) -> -100
    if val_str.startswith('(') and val_str.endswith(')'):
        val_str = '-' + val_str[1:-1]

    # 2. 移除货币符号、逗号、空格
    val_str = re.sub(r'[$,\s]', '', val_str)

    # 3. 处理百分号: "5%" -> "5"
    # 注意：在FinQA逻辑中，通常公式计算结果如果是5%，answer存的是5
    if '%' in val_str:
        val_str = val_str.replace('%', '')

    return val_str


def preprocess_finqa(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_samples = []

    for item in data:
        qa = item['qa']

        # 清洗标准答案
        raw_answer = qa['answer']
        clean_ans = clean_financial_value(raw_answer)

        # 将表格转换为更易读的 Markdown 风格
        table_str = "\n".join([" | ".join(row) for row in item['table']])

        context = "Pre-text:\n" + "\n".join(item['pre_text']) + \
                  "\n\nTable:\n" + table_str + \
                  "\n\nPost-text:\n" + "\n".join(item['post_text'])

        processed_samples.append({
            "id": item['id'],
            "context": context,
            "question": qa['question'],
            "gold_answer": clean_ans,  # 纯数字答案，如 380
            "raw_answer": raw_answer,  # 保留原始答案备查
            "gold_program": qa['program']
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)

    print(f"预处理完成！清洗后的样本已保存。示例: {raw_answer} -> {clean_ans}")


# 执行预处理
preprocess_finqa(r"D:\FinQA-main\dataset\train.json", "processed_train.json")