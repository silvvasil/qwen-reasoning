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


## Step 1.1

Сравниваем модели. `0.5b` это мало: изначально мы получаем `4/1319` соблюдений формата. После обучения, имеем уже `?/1319`.
| | Before  | After |
| ------------- | ------------- | ------------- |
| Format | 4/1319  | ?/l319  |
| Answer | 290/1319  | ?/1319  |
