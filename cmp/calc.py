import argparse
import json
import re
from datasets import load_dataset
from tqdm import tqdm

# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="base", help="Mode: 'tuned' or 'base'")
args = parser.parse_args()

# --------------------
dataset = load_dataset("gsm8k", "main", split="test")


# --------------------
if args.mode == "tuned":
    filename = 'original_responses.json'
else:
    filename = 'tuned_model_responses.json'

with open(filename, "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

responses = [pair["answer"] for pair in qa_pairs]

# --------------------
def clear_number(text: str) -> str:
    text = text.replace(",", ".")
    if text.count(".") > 1:
        return text.replace(".", "")
    return text

def extract_hash_answer(text: str) -> float | None:
    if "####" not in text:
        return None
    try:
        return float(clear_number(text.split("####")[1].strip()))
    except ValueError:
        return None

def extract_last_number(text: str) -> float | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return float(clear_number(numbers[-1]))
    except ValueError:
        return None

def xml_format(texts):
    pattern = re.compile(
        r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>",
        re.DOTALL | re.IGNORECASE
    )
    return [bool(pattern.search(text)) for text in texts]

# --------------------
matches = xml_format(responses)
num_valid = sum(matches)
print(f"[{args.mode}] xml format valid: {num_valid}/{len(matches)}")

correct = 0
for response, answer in zip(responses, dataset["answer"]):
    if extract_last_number(response) == extract_hash_answer(answer):
        correct += 1

print(f"[{args.mode}] correct numeric answers: {correct}/{len(matches)}")
