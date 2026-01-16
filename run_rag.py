from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

class LlamaEngine:

    def __init__(self, data, model_name="meta-llama/Llama-2-7b-chat-hf",
                 temperature=0.3, top_n=1, max_new_tokens=256):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        
        # Simple approach without device_map (no accelerate needed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device=0 if device == "cuda" else -1,
        )

    def get_llama_completion(self, system_prompt: str, user_prompt: str):

        def build_llama_prompt(system_prompt, user_prompt):
            return (
                "<s>[INST] <<SYS>>\n"
                f"{system_prompt.strip()}\n"
                "<</SYS>>\n\n"
                f"{user_prompt.strip()} [/INST]"
            )

        prompt = build_llama_prompt(system_prompt, user_prompt)

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            num_return_sequences=1,
            temperature=self.temperature,
            top_k=10,
            top_p=0.95
        )
        return outputs[0]["generated_text"]


if __name__ == "__main__":
    print("="*60)
    print("STEP 1: Initializing LLM")
    
    llm_instance = LlamaEngine(
        data="",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.3,
        top_n=1
    )

    print("\n" + "="*60)
    print("STEP 2: Loading Data")
    
    print("Loading retrieval results...")
    with open("retrieval_results.json") as f:
        evidence = json.load(f)

    print("\nLoading dataset...")
    loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", "config.ini", Split.DEV)
    queries, qrels, corpus = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    print(f"✓ Corpus size: {len(corpus)} documents")

    print("\n" + "="*60)
    print("STEP 3: Experiment Configuration")
    
    k_relevant = 3 
    noise_levels = [1, 2, 3]
    
    # Create a mapping of indices for random sampling
    if isinstance(corpus, dict):
        all_corpus_ids = list(corpus.keys())
    else:
        # corpus is a list, use indices
        all_corpus_ids = list(range(len(corpus)))
    
    print(f"Relevant documents per query: {k_relevant}")
    print(f"Noise levels to test: {noise_levels}")
    print(f"Questions to process: 1200")
    print(f"Corpus type: {type(corpus).__name__} with {len(corpus)} documents")

    system_prompt = (
        "Follow the given examples and Given the question and context "
        "output final answer for the question using information in the "
        "context and give answer in form of [Final Answer]: \n"
    )

    few_shot_examples = """[Question]: When does monsoon season end in the state the area code 575 is located? 
[Final Answer]: mid-September. 
[Question]: The birth country of Jayantha Ketagoda left the British Empire when? 
[Final Answer]: February 4, 1948.\n\n"""

    print("\n" + "="*60)
    print("STEP 4: Running Experiments")
    
    performance_summary = []

    for n_noise in noise_levels:
        print(f"\n{'='*60}")
        print(f"Running: {k_relevant} Relevant + {n_noise} Random Noise Docs")
        
        matches = 0
        mismatches = 0
        processed_ids = []
        results_log = {"questions": [], "answers": [], "ground_truth": []}

        for row in tqdm(raw_data[:1200], desc=f"Processing (noise={n_noise})"):
            q_id = row.question.id()
            if q_id in processed_ids:
                continue
            processed_ids.append(q_id)

            if q_id not in evidence:
                mismatches += 1
                continue

            # Get top-k relevant documents
            top_k_docs_ids = list(evidence[q_id].keys())[:k_relevant]
            
            if isinstance(corpus, dict):
                relevant_texts = [corpus[int(doc_id)].text() for doc_id in top_k_docs_ids]
            else:
                relevant_texts = [corpus[int(doc_id)].text() for doc_id in top_k_docs_ids]

            # Sample random noise documents
            noise_texts = []
            attempts = 0
            max_attempts = 1000
            while len(noise_texts) < n_noise and attempts < max_attempts:
                random_id = random.choice(all_corpus_ids)
                if str(random_id) not in top_k_docs_ids:
                    if isinstance(corpus, dict):
                        noise_texts.append(corpus[int(random_id)].text())
                    else:
                        noise_texts.append(corpus[random_id].text())
                attempts += 1

            combined_contexts = relevant_texts + noise_texts
            random.shuffle(combined_contexts)
            context_string = " ".join(combined_contexts)

            user_prompt = (
                few_shot_examples + 
                f"Evidence: {context_string} \n" + 
                f"Question: {row.question.text()}"
            )

            chain_answer = llm_instance.get_llama_completion(system_prompt, user_prompt)
            
            chain_answer_lower = chain_answer.lower()
            gold_answer = row.answer.text().strip().lower()

            if "not possible" in chain_answer_lower or "unknown" in chain_answer_lower:
                mismatches += 1
            else:
                pred_answer = chain_answer.split("[Final Answer]:")[-1] if "[Final Answer]:" in chain_answer else chain_answer
                pred_answer = pred_answer.strip().lower()

                if gold_answer in pred_answer:
                    matches += 1
                else:
                    mismatches += 1

            results_log["questions"].append(row.question.text())
            results_log["answers"].append(chain_answer)
            results_log["ground_truth"].append(row.answer.text())

        total = matches + mismatches
        em_score = matches / total if total > 0 else 0
        
        print(f"\nResults for noise={n_noise}:")
        print(f"   Exact Match: {em_score:.4f}")
        print(f"   Correct: {matches}/{total}")
        
        performance_summary.append({
            "noise_count": n_noise, 
            "em_score": em_score,
            "correct": matches,
            "total": total
        })
        
        pd.DataFrame(results_log).to_csv(f"rag_noise_{n_noise}_results.tsv", sep="\t", index=False)
        print(f"Saved: rag_noise_{n_noise}_results.tsv")
    
    summary_df = pd.DataFrame(performance_summary)
    summary_df.to_csv("noise_impact_summary.tsv", sep="\t", index=False)
    
    print("\nSummary of Results:")
    print(summary_df.to_string(index=False))
