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

    def __init__(self, data, model_name="meta-llama/Llama-2-7b-chat-hf",
                 temperature=0.3, top_n=1, max_new_tokens=256):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
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

if __name__=="__main__":
    config_instance = LLMEngineOrchestrator()
    #llm_instance = config_instance.get_llm_engine(data="", llm_class="openai", model_name="gpt-3.5-turbo")
    
    llm_instance = LlamaEngine(
        data="",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.3,
        top_n=1
    )
    
    question_df = {"questions": [], "answers": []}

    loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", "config.ini", Split.DEV)
    #raw_data = loader.base_dataset.raw_data
    queries, qrels, corpus = loader.qrels()
    raw_data = loader.base_dataset.raw_data
    
    system_prompt = (
        "Given the question and context, output the final answer using "
        "information in the context. Give the answer in the form of "
        "[Final Answer]: <answer>\n"
    )
    
    matches = 0
    mismatches = 0

    # --- STEP 1: PRE-PROCESS / GROUP DATA ---
    # The original loop was risky. We group data by ID first to get all oracle contexts.
    grouped_data = defaultdict(lambda: {"q_text": "", "answer": "", "evidences": []})
    ids = []

    for row in raw_data:
        
        q_id = row.question.id()
        
        if q_id in ids:
            continue
        
        # Update the entry for this Question ID
        grouped_data[q_id]["q_text"] = row.question.text()
        grouped_data[q_id]["answer"] = row.answer.text() # or row.answer.text() depending on object structure
        grouped_data[q_id]["evidences"].append(row.evidences.text())

    # --- STEP 2: RUN ORACLE EXPERIMENT ---
    print(f"Processing {len(grouped_data)} unique questions...")

    for q_id, data in grouped_data.items():
        q_text = data["q_text"]
        gold_answer_obj = data["answer"]
        gold_answer_text = gold_answer_obj#.text() # Extract text string
        
        # --- CRITICAL CHANGE FOR ORACLE EXPERIMENT ---
        # 1. We do NOT calculate embeddings.
        # 2. We do NOT use get_top_k_similar_instances.
        # 3. We use ALL collected evidences (Oracle Contexts) directly.
        
        all_oracle_evidences = data["evidences"]
        evidence_text = " ".join(all_oracle_evidences)
        
        # Construct the User Prompt
        # (This remains largely the same, but uses the full evidence_text)
#         user_prompt = """[Question]: When does monsoon season end in the state the area code 575 is located?
# [Final Answer]: mid-September.
# [Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?
# [Final Answer]: United States dollar.
# [Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?
# [Final Answer]: Jefferson.
# [Question]: What language is used by the director of Tiffany Memorandum?
# [Final Answer]: Italian.
# [Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
# [Final Answer]: Green Bay Packers.
# [Question]: The birth country of Jayantha Ketagoda left the British Empire when?
# [Final Answer]: February 4, 1948.\n\n """ + f"Follow the above example, and Given the evidence, Evidence: {evidence_text} \n use the information and answer the Question: {q_text}"

        user_prompt = (
                f"Evidence: {evidence_text}\n\n"
                f"Based on the evidence above, answer the Question: {q_text}"
            )
        
        #print(f"Processing ID: {q_id}")
        
        # Call LLM
        #chain_answer = llm_instance.get_chat_completion(user_prompt, system_prompt)
        chain_answer = llm_instance.get_llama_completion(
                system_prompt,
                user_prompt
            )
        
        # --- EVALUATION LOGIC ---
        clean_chain_answer = chain_answer.lower()
        
        # 1. Check for refusal/uncertainty first
        if "not possible" in clean_chain_answer or "unknown" in clean_chain_answer:
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
            #gold_answer = row.answer.text().strip()

            # Check if the correct answer is contained within the prediction
            if gold_answer_text.lower() in pred_answer.lower():
                matches += 1
            else:
                mismatches += 1
            
        question_df["answers"].append(chain_answer)
        question_df["questions"].append(q_text)

    em_score = matches / (matches + mismatches)
    
    performance=[]
    performance.append({
            "exact_match": em_score,
            "matches": matches,
            "mismatches": mismatches
        })
    
    perf_df = pd.DataFrame(performance)
    perf_df.to_csv("llama_rag_em_oracle_results.tsv", sep="\t", index=False)

    # Save results
    final_questions = pd.DataFrame(question_df)
    final_questions.to_csv("llama_rag_oracle_results_questions.tsv", sep="\t", index=False)