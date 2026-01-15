from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import json
import numpy as np

ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

passages = []  # all passages from your dpr-ready JSON
embeddings = []

passages = []
with open("./data/train_dpr_ready.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        for ctx_type in ["positive_ctxs", "hard_negative_ctxs", "negative_ctxs"]:
            for p in entry.get(ctx_type, []):
                if isinstance(p, dict) and "text" in p:
                    passages.append({"text": p["text"], "title": p["title"], "id": p["passage_id"]})

print(f"Total passages: {len(passages)}")

for p in passages:
    inputs = ctx_tokenizer(p['text'], return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        emb = ctx_encoder(**inputs).pooler_output
    embeddings.append(emb.cpu().numpy())

embeddings = np.vstack(embeddings)
np.memmap('passages.memmap', dtype='float32', mode='w+', shape=embeddings.shape)[:] = embeddings