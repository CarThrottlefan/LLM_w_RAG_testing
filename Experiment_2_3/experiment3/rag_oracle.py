from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
import torch
import transformers
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


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

if __name__=="__main__":
    
    # Create an instance of the Llama model engine
    llm_instance = LlamaEngine(
        data="",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.3,
        top_n=1
    )
    
    # Dictionary to store questions and generated answers for later analysis
    question_df = {"questions": [], "answers": []}

    # Load the WikiMultiHopQA dataset for evaluation
    # This dataset contains multi-hop questions requiring reasoning across multiple documents
    loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", "config.ini", Split.DEV)
    queries, qrels, corpus = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    
    # System prompt: instructs the model on how to format its answer
    system_prompt = (
        "Given the question and context, output the final answer using "
        "information in the context. Give the answer in the form of "
        "[Final Answer]: <answer>\n"
    )
    
    # Counters for evaluation metrics
    matches = 0 # Correct answers
    mismatches = 0 # Incorrect or refused answers

    # Group all evidence passages by question ID
    # This creates "oracle contexts" - the perfect set of passages for each question
    grouped_data = defaultdict(lambda: {"q_text": "", "answer": "", "evidences": []})
    ids = []

    for row in raw_data:
        
        q_id = row.question.id()
                
        # Update the entry for this Question ID
        grouped_data[q_id]["q_text"] = row.question.text()
        grouped_data[q_id]["answer"] = row.answer.text() 
        grouped_data[q_id]["evidences"].append(row.evidences.text())

    print(f"Processing {len(grouped_data)} unique questions...")

    # Process each unique question
    for q_id, data in grouped_data.items():
        q_text = data["q_text"]
        gold_answer_obj = data["answer"]
        gold_answer_text = gold_answer_obj # The ground truth answer
        
        # Combine all oracle evidence passages into a single context string    
        all_oracle_evidences = data["evidences"]
        evidence_text = " ".join(all_oracle_evidences)
        
        # Construct the user prompt with evidence and question
        user_prompt = (
                f"Evidence: {evidence_text}\n\n"
                f"Based on the evidence above, answer the Question: {q_text}"
            )
                
        # Generate answer using the Llama model
        chain_answer = llm_instance.get_llama_completion(
                system_prompt,
                user_prompt
            )
        
        # Convert to lowercase for case-insensitive comparison
        clean_chain_answer = chain_answer.lower()
        
        # Check if model refused to answer or expressed uncertainty
        if "not possible" in clean_chain_answer or "unknown" in clean_chain_answer:
            mismatches += 1

        else:
            # Extract the final answer from the model's response
            if "[Final Answer]:" in chain_answer:
                # Best case: Model followed the instructed format
                pred_answer = chain_answer.split("[Final Answer]:")[-1]
            else:
                # Fallback: Model didn't use the tag, check entire response
                pred_answer = chain_answer

            # Compare predicted answer with ground truth
            pred_answer = pred_answer.strip()

            # Use substring matching: correct if gold answer appears in prediction
            if gold_answer_text.lower() in pred_answer.lower():
                matches += 1
            else:
                mismatches += 1
        
        # Store the question and answer for qualitative analysis
        question_df["answers"].append(chain_answer)
        question_df["questions"].append(q_text)

    # Calculate Exact Match (EM) score
    em_score = matches / (matches + mismatches)
    
    # Prepare performance metrics for saving
    performance=[]
    performance.append({
            "exact_match": em_score,
            "matches": matches,
            "mismatches": mismatches
        })
    
    # Save performance metrics to a TSV file
    perf_df = pd.DataFrame(performance)
    perf_df.to_csv("llama_rag_em_oracle_results.tsv", sep="\t", index=False)

    # Save all questions and generated answers for detailed analysis
    final_questions = pd.DataFrame(question_df)
    final_questions.to_csv("llama_rag_oracle_results_questions.tsv", sep="\t", index=False)