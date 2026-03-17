import re

def extract_solution(solution_str):
    """Extract numeric answer from the last <answer> tag."""
    matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if matches:
        raw_ans = matches[-1].strip()
        # Keep only numbers, dot, and minus sign
        clean_ans = re.sub(r'[^0-9.\-]', '', raw_ans)
        return clean_ans
    return None

def validate_format(solution_str):
    """Check if <think> and <answer> exist and are in order."""
    think_match = re.search(r'<think>.*?</think>', solution_str, re.DOTALL)
    answer_match = re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL)
    if think_match and answer_match:
        return think_match.start() < answer_match.start()
    return False

def compute_score(solution_str, ground_truth, format_score=0.3, score=1.0):
    """The scoring function for FinQA."""
    answer = extract_solution(solution_str)
    
    # 1. Check numeric correctness
    if answer is not None:
        try:
            if abs(float(answer) - float(ground_truth)) < 1e-3:
                return score
        except (ValueError, TypeError):
            pass

    # 2. Check format if answer is wrong
    if validate_format(solution_str):
        return format_score

    return 0.0