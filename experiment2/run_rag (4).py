from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

# class LlamaEngine:

#     def __init__(self, data, model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.3, top_n=1, max_new_tokens=256):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.temperature = temperature
#         self.data = data
#         self.top_n = top_n
#         self.max_new_tokens=max_new_tokens
#         self.pipeline = transformers.pipeline(
#             "text-generation",
#             model=self.model_name,
#             torch_dtype=torch.float16,
#             device_map="auto",
#         )

#     def get_llama_completion(self, system_prompt: str, user_prompt: str):
# #         messages = [
# #             {
# #                 "role": "user",
# #                 "content": system_prompt,
# #             },
# #               {
# #                 "role": "assistant",
# #                 "content": "Yes I will reason and generate the answer",
# #             },
# #             {"role": "user", "content": user_prompt
# # },
# #         ]
#         print("removed get chat template")
#         def build_llama_prompt(system_prompt, user_prompt):
#           return (
#               "<s>[INST] <<SYS>>\n"
#               f"{system_prompt.strip()}\n"
#               "<</SYS>>\n\n"
#               f"{user_prompt.strip()} [/INST]"
#           )

#         prompt = build_llama_prompt(system_prompt, user_prompt)

#         outputs = self.pipeline(prompt,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=True,
#             num_return_sequences=1,
#             temperature=self.temperature,
#             top_k=10,
#             top_p=0.95)
#         return outputs[0]["generated_text"]

# if __name__=="__main__":
#         config_instance = LLMEngineOrchestrator()
#         #llm_instance = config_instance.get_llm_engine(data="",llm_class="llama",model_name="meta-llama/Llama-2-7b-chat-hf")
#         llm_instance=LlamaEngine(data="",model_name="meta-llama/Llama-2-7b-chat-hf",temperature=0.3,top_n = 1)
#         #assertTrue(isinstance(llm_instance, OpenAIEngine))
#         with open("/content/retrieval_results.json") as f:
#                 evidence = json.load(f)
#         question_df = {"questions":[],"answers":[]}

#         loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","config.ini",Split.DEV)
#         queries, qrels, corpus = loader.qrels()
#         raw_data = loader.base_dataset.raw_data
#         system_prompt = "Follow the given examples and Given the question and context output final answer for the question using information in the context and give answer in form of  [Final Answer]: \n"
#         matches = 0
#         mismatches = 0
#         ids = []
#         for row in raw_data:
#                 if row.question.id() in ids:
#                         continue
#                 else:
#                         ids.append(row.question.id())
#                 top_3 = list(evidence[row.question.id()])[0:10]
#                 top_k_context = " ".join(corpus[int(doc_id)].text() for doc_id in top_3)
#                 user_prompt = """[Question]: When does monsoon season end in the state the area code 575 is located?
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
# [Final Answer]: February 4, 1948.\n\n """ + "Follow the above example and Given the evidence, Evidence: "+top_k_context+" \n use the information and answer the Question:"+row.question.text()
#                 print("user_prompt",user_prompt)
#                 chain_answer = llm_instance.get_llama_completion(user_prompt,system_prompt)
#                 if "not possible" in chain_answer.lower():
#                         mismatches+=1
#                         continue
#                 elif "unknown" in chain_answer.lower():
#                         mismatches+=1
#                         continue
#                 elif len(chain_answer.split("[Final Answer]:")) >1:
#                         answer = chain_answer.split("[Final Answer]:")[-1]
#                         print("************",answer,row.answer.text())
#                         if row.answer.text().lower() in answer.lower():
#                                 matches+=1
#                         else:
#                                 mismatches+=1
#                 else:
#                         mismatches+=1
#                 question_df["answers"].append(chain_answer)
#                 question_df["questions"].append(row.question.text())


#                 final_questions = pd.DataFrame(question_df)
#                 print("EM", matches/(matches+mismatches))
#                 print(final_questions)
#                 final_questions.to_csv("chatgpt_wqa_rag_10_few_shot.tsv",sep="\t",index=False)

# from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
# import json
# import pandas as pd
# import torch
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from dexter.config.constants import Split
# from dexter.data.loaders.RetrieverDataset import RetrieverDataset


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


if __name__ == "__main__":

    config_instance = LLMEngineOrchestrator()
    llm_instance = LlamaEngine(
        data="",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.3,
        top_n=1
    )

    with open("/content/retrieval_results.json") as f:
        evidence = json.load(f)

    loader = RetrieverDataset(
        "wikimultihopqa",
        "wiki-musiqueqa-corpus",
        "config.ini",
        Split.DEV
    )

    queries, qrels, corpus = loader.qrels()
    raw_data = loader.base_dataset.raw_data

    # system_prompt = (
    #     "Follow the given examples and Given the question and context "
    #     "output final answer for the question using information in the "
    #     "context and give answer in form of  [Final Answer]: \n"
    # )
    # NEW (Zero-shot version)
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

#             user_prompt = """[Question]: When does monsoon season end in the state the area code 575 is located?
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
# [Final Answer]: February 4, 1948.\n\n""" + \
#             "Follow the above example and Given the evidence, Evidence: " + \
#             top_k_context + \
#             " \n use the information and answer the Question: " + \
#             row.question.text()
            # We construct a clean prompt with ONLY the Context (Evidence) and Question.
            user_prompt = (
                f"Evidence: {top_k_context}\n\n"
                f"Based on the evidence above, answer the Question: {row.question.text()}"
            )

            chain_answer = llm_instance.get_llama_completion(
                system_prompt,
                user_prompt
            )

            # if "not possible" in chain_answer.lower():
            #     mismatches += 1
            # elif "unknown" in chain_answer.lower():
            #     mismatches += 1
            # elif "[Final Answer]:" in chain_answer:
            #     answer = chain_answer.split("[Final Answer]:")[-1]
            #     if row.answer.text().lower() in answer.lower():
            #         matches += 1
            #     else:
            #         mismatches += 1
            # else:
            #     mismatches += 1
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


