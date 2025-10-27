import tiktoken
import json

path = "./dickens/results/novel_results_hybrid.json"
encoding = tiktoken.get_encoding("o200k_base")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
    tokens = 0
    for v in data:
        tokens += len(encoding.encode(v['answer']))
print(f"Total tokens: {tokens}")