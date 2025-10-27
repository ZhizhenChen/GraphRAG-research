import tiktoken
import json

path = "./dickens/kv_store_llm_response_cache.json"
encoding = tiktoken.get_encoding("o200k_base")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
    tokens = 0
    for k,v in data.items():
        tokens += len(encoding.encode(v['return']))
print(f"Total tokens: {tokens}")