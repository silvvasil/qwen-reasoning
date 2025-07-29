import re
import math

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def grpo_correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward = 2.0 for exact match of extracted <answer> vs ground truth, else 0.0"""
    contents = [completion[0]['content'] for completion in completions]
    extracted = [extract_xml_answer(c) for c in contents]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]


def grpo_formatting_reward(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
    contents = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, content, re.DOTALL) else 0.0 for content in contents]


def grpo_tag_count_reward(completions, **kwargs) -> list[float]:
    def count_tags(text: str) -> float:
        count = 0.0
        count += 0.125 if text.count("<reasoning>\n") == 1 else 0.0
        count += 0.125 if text.count("\n</reasoning>\n") == 1 else 0.0
        count += 0.125 if text.count("\n<answer>\n") == 1 else 0.0
        count += 0.125 if text.count("\n</answer>") == 1 else 0.0
        return count

    return [count_tags(completion[0]["content"]) for completion in completions]

def grpo_reasoning_steps_reward(completions, **kwargs) -> list[float]:
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    contents = [completion[0]["content"] for completion in completions]
    return [min(1.0, len(re.findall(pattern, c)) / 3) for c in contents]


def grpo_length_efficiency_reward(completions, answer, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    correctness = [extract_xml_answer(c) == a for c, a in zip(contents, answer)]
    lengths = [len(c) for c in contents]
    min_len = min(lengths)
    max_len = max(lengths)
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
        rewards.append(lambda_val if is_correct else min(0.0, lambda_val))
    return rewards

