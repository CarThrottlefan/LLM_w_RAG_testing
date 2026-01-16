from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

class LlamaEngine:
    
    """
    A wrapper class for running the Llama 2 language model for text generation tasks.
    Provides a simple interface for getting completions from the Llama model.
    """

    def __init__(self, data, model_name="meta-llama/Llama-2-7b-chat-hf",
                 temperature=0.3, top_n=1, max_new_tokens=256):
        
        """
        Initialize the Llama engine with model configuration.
        
        Args:
            data: Dataset (not actively used in this implementation)
            model_name: HuggingFace model identifier for Llama
            temperature: Controls randomness (lower = more deterministic)
            top_n: Number of sequences to return (currently fixed at 1)
            max_new_tokens: Maximum length of generated response
        """
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        # Create a text generation pipeline with automatic device placement
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16, # Use half precision for memory efficiency
            device_map="auto", # Automatically distribute model across available devices
        )

    def get_llama_completion(self, system_prompt: str, user_prompt: str):
        
        """
        Generate a completion from Llama using the chat template format.
        
        Args:
            system_prompt: Instructions that set the model's behavior
            user_prompt: The actual question/task for the model
            
        Returns:
            str: The full generated text including the original prompt
        """

        def build_llama_prompt(system_prompt, user_prompt):
            
            """
            Format prompts according to Llama 2's chat template.
            Uses special tokens: <s>, [INST], <<SYS>>, etc.
            """
            
            return (
                "<s>[INST] <<SYS>>\n"
                f"{system_prompt.strip()}\n"
                "<</SYS>>\n\n"
                f"{user_prompt.strip()} [/INST]"
            )

        # Build the properly formatted prompt
        prompt = build_llama_prompt(system_prompt, user_prompt)

        # Generate response using the pipeline
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True, # Enable sampling (vs greedy decoding)
            num_return_sequences=1, # Generate one response
            temperature=self.temperature,
            top_k=10, # Consider top 10 tokens at each step
            top_p=0.95 # Nucleus sampling threshold
        )
        return outputs[0]["generated_text"]


if __name__ == "__main__":

    # Create an instance of the Llama model engine
    llm_instance = LlamaEngine(
        data="",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.3,
        top_n=1
    )

    # Load retrieval results from JSON file generated using contriver
    with open("./retrieval_results.json") as f:
        evidence = json.load(f)

    # Load the WikiMultiHopQA dataset for evaluation
    # This dataset contains multi-hop questions requiring reasoning across multiple documents
    loader = RetrieverDataset(
        "wikimultihopqa",
        "wiki-musiqueqa-corpus",
        "config.ini",
        Split.DEV
    )

    queries, qrels, corpus = loader.qrels()
    raw_data = loader.base_dataset.raw_data

     # System prompt: instructs the model on how to format its answer
    system_prompt = (
        "Given the question and context, output the final answer using "
        "information in the context. Give the answer in the form of "
        "[Final Answer]: <answer>\n"
    )

    # -------------------------------
    # NEW: top-k settings + EM storage
    # -------------------------------
    k_values = [1, 3, 5]
    performance = []

    for k in k_values:

        print(f"\n===== Running evaluation for top-k = {k} =====")

        matches = 0
        mismatches = 0
        ids = []
        question_df = {"questions": [], "answers": []}

        counter=0
        
        for row in raw_data:

            if row.question.id() in ids:
                continue
            ids.append(row.question.id())

            # select top-k retrieved documents
            top_k_docs = list(evidence[row.question.id()])[:k]

            top_k_context = " ".join(
                corpus[int(doc_id)].text() for doc_id in top_k_docs
            )

            # We construct a clean prompt with ONLY the Context (Evidence) and Question.
            user_prompt = (
                f"Evidence: {top_k_context}\n\n"
                f"Based on the evidence above, answer the Question: {row.question.text()}"
            )

            chain_answer = llm_instance.get_llama_completion(
                system_prompt,
                user_prompt
            )

            # Lowercase once for easier checking
            chain_answer_lower = chain_answer.lower()

            # 1. Check for refusal/uncertainty first
            if "not possible" in chain_answer_lower or "unknown" in chain_answer_lower:
                mismatches += 1

            else:
                # 2. Attempt to extract the answer using the tag
                if "[Final Answer]:" in chain_answer:
                    # Best case: Model followed instructions
                    pred_answer = chain_answer.split("[Final Answer]:")[-1]
                else:
                    # Fallback case: Model forgot the tag, so we check the whole text
                    pred_answer = chain_answer

                # 3. Compare with Ground Truth
                # We strip whitespace to avoid issues with trailing newlines
                pred_answer = pred_answer.strip()
                gold_answer = row.answer.text().strip()

                # Check if the correct answer is contained within the prediction
                if gold_answer.lower() in pred_answer.lower():
                    matches += 1
                else:
                    mismatches += 1

            question_df["questions"].append(row.question.text())
            question_df["answers"].append(chain_answer)

            counter+=1
            if counter>=1200:
              break

        # compute EM
        em_score = matches / (matches + mismatches)

        # report EM
        print(f"Top-{k} Exact Match (EM): {em_score:.4f}")

        # store EM
        performance.append({
            "k": k,
            "exact_match": em_score,
            "matches": matches,
            "mismatches": mismatches
        })

        # store generated answers
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv(
            f"llama_rag_top{k}_results.tsv",
            sep="\t",
            index=False
        )

    # -------------------------------
    # Save EM summary across k
    # -------------------------------
    perf_df = pd.DataFrame(performance)
    perf_df.to_csv("llama_rag_em_results.tsv", sep="\t", index=False)

    print("\n===== Final EM Summary =====")
    print(perf_df)


