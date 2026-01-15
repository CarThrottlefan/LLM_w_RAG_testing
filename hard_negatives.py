import pickle
import json
from random import shuffle, sample
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from run_rag import LlamaEngine

os.environ["HF_HUB_OFFLINE"] = "1"
loader = RetrieverDataset("wikimultihopqa",
                          "wiki-musiqueqa-corpus",
                          "D:\\NLP_proj_LLM_w-RAG\\config.ini",
                          Split.TRAIN,
                          tokenizer=None)
queries, qrels, corpus = loader.qrels()
config_instance = DenseHyperParams(
    query_encoder_path="facebook/contriever",
    document_encoder_path="facebook/contriever",
    batch_size=64,
    show_progress_bar=True
)
retriever = Contriever(config_instance)
similarity_measure = DotScore()

corpus_map = {doc.id(): doc for doc in corpus} # info for each passage
queries_map = {query.id(): query for query in queries} # id and text of each question
all_augmented_contexts = [] # positive passages + hard negatives
dpr_entries = [] # DPR entries used for ADORE-tuning later
num_hard_negs = 3
num_negs = 2
broad_results = retriever.retrieve(corpus, queries, 100, similarity_measure, chunk=True, chunksize=50000)

for query_id, retrieved_docs in broad_results.items():
    gold_doc_ids = set(qrels[query_id].keys())
    ranked_docs = list(retrieved_docs.items())
    positives = []
    hard_negs = []
    negs = []

    for doc_id, score in ranked_docs:
        doc_obj = corpus_map.get(doc_id)
        if not doc_obj:
            continue

        doc_entry = {
            "id": doc_id,
            "title": doc_obj.title(),
            "text": doc_obj.text(),
            "score": score
        }

        if doc_id in gold_doc_ids:
            positives.append(doc_entry)
        else:
            negs.append(doc_entry)

    negs.sort(key=lambda x: x["score"], reverse=True)
    hard_negs = negs[:num_hard_negs]
    remaining_negs = negs[num_hard_negs:]
    if len(remaining_negs) >= num_negs:
        random_negs = sample(remaining_negs, num_negs)
    else:
        random_negs = remaining_negs

    selected_docs = positives + hard_negs
    shuffle(selected_docs)
    all_augmented_contexts.append(selected_docs)

    dpr_entry = {
        "id": query_id,
        "question": queries_map[query_id].text(),
        "positive_ctxs": [
            {"title": p["title"], "text": p["text"], "passage_id": p["id"]} 
            for p in positives
        ],
        "hard_negative_ctxs": [
            {"title": hn["title"], "text": hn["text"], "passage_id": hn["id"]} 
            for hn in hard_negs
        ],
        "negative_ctxs": [
            {"title": rn["title"], "text": rn["text"], "passage_id": rn["id"]} 
            for rn in random_negs
        ]
    }
    dpr_entries.append(dpr_entry)

# with open("   .jsonl", "w", encoding="utf-8") as f:
#     for entry in dpr_entries:
#         f.write(json.dumps(entry) + "\n")

# read
# with open("./data/all_augmented_contexts.pkl", "rb") as f:
#     all_augmented_contexts = pickle.load(f)

# save
# with open('all_augmented_contexts.pkl', 'wb') as f:
#     pickle.dump(all_augmented_contexts, f)

# LlamaEngine.run_rag(all_augmented_contexts)