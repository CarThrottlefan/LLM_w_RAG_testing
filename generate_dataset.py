from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split

loader = RetrieverDataset("wikimultihopqa",
                          "wiki-musiqueqa-corpus",
                          "D:\\NLP_proj_LLM_w-RAG\\config.ini",
                          Split.TRAIN,
                          tokenizer=None)
queries, qrels, corpus = loader.qrels()


with open("collection.tsv", "w", encoding="utf-8") as f:
    for doc in corpus:
        f.write(f"{doc.id()}\t{doc.text()}\n")

with open("queries.train.tsv", "w", encoding="utf-8") as f:
    for q in queries:
        f.write(f"{q.id()}\t{q.text()}\n")

with open("qrels.train.tsv", "w", encoding="utf-8") as f:
    for qid, docs in qrels.items():
        for docid in docs:
            f.write(f"{qid}\t0\t{docid}\t1\n")