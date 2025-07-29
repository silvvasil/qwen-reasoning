from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
) # если не хватает памяти уменьшайте значение параметра gpu_memory_utilization

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Если не хватает памяти обучайте только QKVO
    lora_alpha = lora_rank,
    random_state = 42,
)

import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    # data = data.select(range(len(data) // 2))
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

from rewards import *
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    max_grad_norm = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 10,
    fp16 = False,
    bf16 = True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    num_generations = 8, # Если не хватает памяти, уменьшайте
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs = 1,
    max_steps = len(dataset),
    save_steps = 250,
    report_to = "wandb",
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        grpo_correctness_reward,
        grpo_formatting_reward,
        grpo_tag_count_reward,
        grpo_reasoning_steps_reward,
        grpo_length_efficiency_reward
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_lora("grpo_saved_lora")