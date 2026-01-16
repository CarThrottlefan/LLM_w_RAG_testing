import json
from pathlib import Path

DPR_ENTRIES_PATH = "D:/NLP_proj_LLM_w-RAG/data/train_dpr_ready.jsonl"
PREPROCESSED_PATH = "D:/NLP_proj_LLM_w-RAG/data/train_dpr_ready.jsonl"
input_file = Path(DPR_ENTRIES_PATH)
preprocess_dir = Path(PREPROCESSED_PATH)
preprocess_dir.mkdir(exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f, \
     open(preprocess_dir / "queries.tsv", "w", encoding="utf-8") as fq, \
     open(preprocess_dir / "collection.tsv", "w", encoding="utf-8") as fc, \
     open(preprocess_dir / "qrels.tsv", "w", encoding="utf-8") as fr:

    seen_passages = set()
    for line in f:
        entry = json.loads(line)
        qid = entry["id"]
        question = entry["question"]
        fq.write(f"{qid}\t{question}\n")

        # positives
        for p in entry.get("positive_ctxs", []):
            pid = p["passage_id"]
            if pid not in seen_passages:
                fc.write(f"{pid}\t{p['text']}\n")
                seen_passages.add(pid)
            fr.write(f"{qid}\t0\t{pid}\n")

        # optionally include hard negatives in the collection but not in qrels
        for hn in entry.get("hard_negative_ctxs", []):
            pid = hn["passage_id"]
            if pid not in seen_passages:
                fc.write(f"{pid}\t{hn['text']}\n")
                seen_passages.add(pid)