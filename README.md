# Qwen Reasoning

## Setup

```
uv venv --python=3.12.3
uv pip install -r requirements.txt
uv pip install torch -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
uv pip install torch-geometric -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
uv pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

Работало на Intel Ice Lake with NVIDIA® Tesla® T4	и Intel Ice Lake with T4i

## Step 1

```
python3 main.py
```

Функции для наград были вдохновлены [Open R1](https://github.com/huggingface/open-r1).

До этого я использовал функции примера в документации unsloth. И там была следующая проблема: не было reward функции "хвалящей" за длину reasoning. Поэтому ответы модели свелись к 
```
<reasoning>
#2
</resoning>
<answer>
#1
<answer>
```
И `#1` и `#2` были близки к случайным числам. Точно ответов модели сильно упала: с 300 до 80, но при этом, модель идеально научилась соблюдать требуемый формат.

## Сравнение моделей

Сравниваем модели.

```
python3 cmp/gen.py --mode base
python3 cmp/calc.py --mode base
python3 cmp/gen.py --mode tuned
python3 cmp/calc.py --mode tuned
```

| | Before (0.5b)  | Open R1 rewards | Qwen Colab rewards | 
| ------------- | ------------- | ------------- | ------------- |
| Correct Format | 4/1319  | ?/l319  | 1319/1319 |
| Correct Answer | 301/1319  | ?/1319  | 81/1319 |
