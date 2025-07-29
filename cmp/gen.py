import argparse
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from vllm import SamplingParams
from unsloth import FastLanguageModel, PatchFastRL

# --------------------
PatchFastRL("GRPO", FastLanguageModel)

LORA_RANK = 64
DEVICE = "cuda"

# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="base", help="Mode: 'base' or 'tuned'")
args = parser.parse_args()

# --------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK,
    random_state=42,
)

model.eval()

# --------------------
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=4096,
)

# --------------------
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def build_prompt(question):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# --------------------
dataset = load_dataset("gsm8k", "main", split="test")

# --------------------
responses = []
prompts = []

if args.mode == "tuned":
    lora = model.load_lora("grpo_saved_lora")
    filename = 'cmp/tuned_model_responses.json'
else:
    filename = 'cmp/original_model_responses.json'
    lora = None

batch_size = 32
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i + batch_size]
    batch_prompts = [
        build_prompt(example) for example in batch["question"]
    ]
    results = model.fast_generate(
        batch_prompts,
        sampling_params=sampling_params,
        lora_request=lora,
    )
    batch_responses = [ 
        result.outputs[0].text
        for result in results
    ]
    responses += batch_responses
    prompts += batch_prompts 

# --------------------
qa_pairs = [{"question": q, "answer": a} for q, a in zip(prompts, responses)]
try:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print("Saved")
except Exception as e:
    print(f"Failed to write JSON: {e}")
