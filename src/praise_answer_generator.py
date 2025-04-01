import copy
import json
import torch
import sys
from functools import partial
from loguru import logger
sys.path.append(".")
sys.path.append("..")
from utils import  load_config
import random
import numpy as np
from pynvml import *
from praise_base_component import PraiseBaseComponent
from evaluation import is_answer_correct, get_hit_at_5, get_mrr


random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

###########################################################################
# AG module for answer generation based on evidence and rewritten question
###########################################################################
class PraiseAnswerGenerator(PraiseBaseComponent):

    def __init__(self, config ):
        self.p_at_1 = 0
        self.hit_at_5 = 0
        self.mrr = 0
        self.INSTRUCTION = ("You are given a question and references which may or may not help answer the question.  "
               "Please answer the question in as few words as possible by using the information provided in the references that is relevant in answering the question. "
              # "Please learn from the examples below. "
               )
        config["model_path"] =  config["ag_model_path"] 
        if "ag_peft_model_path" in config.keys():
            config["peft_model_path"] =  config["ag_peft_model_path"]
        config["adapter_name"] =  config["ag_adapter_name"]
        super(PraiseAnswerGenerator, self).__init__(config, "answer", self.INSTRUCTION, None)
    

    def final_metrics(self, count):
        logger.info(f"Precision: `{self.p_at_1/count}`")
        logger.info(f"Hit@5: `{self.hit_at_5/count}`")
        logger.info(f"MRR: `{self.mrr/count}`")
        
    
    def inference_per_question(self, rewritten_question, context):
        return self.generate_output(rewritten_question, context)
    

    def inference(self, data, output_path):
        count = 0
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
                count += 1 
                entry = data[i]["questions"][j]
                question = entry["rewritten_question"]
                if len(entry["reranked_evidence"]) > self.config["top_k"]:
                    entry["reranked_evidence"] = entry["reranked_evidence"][:self.config["top_k"]]
                context =  "\n".join(evidence[1] for evidence in entry["reranked_evidence"])
                
                #remove some info from previous step, not required for the answer generation:
                if "noisy_gold_evidence" in entry.keys():
                    del entry["noisy_gold_evidence"]
                if "distractor_evidence" in entry.keys():
                    del entry["distractor_evidence"]

                #sample multiple generations for getting output list, calculate metrics
                if "sampling" in self.config.keys():
                    greedy_out = self.generate_output(question,context)
                    sample_out =  self.generate_sampled_output(question,context)
                    entry["generated_answer"] = [greedy_out] + sample_out
                    gen_ans = greedy_out
                    hit_at_5 = get_hit_at_5(entry["generated_answer"], entry["answers"])
                    entry["hit_at_5"] = hit_at_5
                    self.hit_at_5 += hit_at_5
                    mrr = get_mrr(entry["generated_answer"], entry["answers"])
                    entry["mrr"] = mrr
                    self.mrr += mrr
                else:
                    gen_ans = self.generate_output(question,context)
                    entry["generated_answer"] = gen_ans 
                if is_answer_correct(gen_ans, entry["answers"]):
                    self.p_at_1 += 1
                    entry["p_at_1"] = 1
                else:
                    entry["p_at_1"] = 0
                         
        with open(output_path, "w") as fout:
            json.dump(data, fout)

        self.final_metrics(count)


    #get AG feedback for DPO training of ERF module
    def get_feedback_for_ERF(self, data, output_path):
        count = 0
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
                count += 1
                entry = data[i]["questions"][j]
                entry["generated_answer"] = []
                entry["p_at_1"] = []
                question = entry["rewritten_question"]
                  
                if not "reranked_evidence" in entry.keys():
                    print("no reranking for question ", question)
                    continue
                #go over sampled evidence list and check which list successfuly leads model to answer question correctly
                for r in range(len(entry["reranked_evidence"])):         
                    ev_entry = entry["reranked_evidence"][r] 
                    context = "\n".join(evidence[1] for evidence in  ev_entry)  
                    gen_ans = self.generate_output(question,context)
                    entry["generated_answer"].append(gen_ans) 
                    if is_answer_correct(gen_ans, entry["answers"]):
                        entry["p_at_1"].append(1)
                        self.p_at_1 += 1
                    else:
                        entry["p_at_1"].append(0)
                           
        with open(output_path, "w") as fout:
            json.dump(data, fout)

        self.final_metrics(count)
     

    #create SFT training data
    def prepare_sft_data(self, data):
        inputs = []
        questionDict = dict()
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
                current_data = data[i]["questions"][j]
                if "completed_question" in self.config.keys():
                    if "completed" in current_data.keys():
                        question = current_data["completed"]
                    else:
                        question = current_data["question"]
                else:
                    if not "rewritten_question" in current_data.keys():
                        continue
                    question = current_data["rewritten_question"]
               
                answer =  current_data["answers"][0]["label"]
                question_id = current_data["question_id"]
                if question_id in questionDict.keys():
                    if question.strip().lower() in questionDict[question_id]:
                        print("avoid duplicates")
                        continue
                    else:
                        questionDict[question_id].append(question.strip().lower())
                else:
                    questionDict[question_id] = [question.strip().lower()]
               
                if  not "reranked_evidence" in current_data.keys() or len(current_data["reranked_evidence"]) == 0:
                    continue
                evs = copy.deepcopy(current_data["reranked_evidence"]) 
                if len(evs)>self.config["top_k"]:
                    evs = evs[:self.config["top_k"]]
                joint_evidences = "\n".join(evidence[1] for evidence in  evs)
                full_prompt = self.create_prompt(question, joint_evidences, answer)
                full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
               
                prompt = self.create_prompt(question, joint_evidences, None)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                entry = dict()
                entry["prompt"] = prompt
                entry["full_prompt"] = full_prompt
                inputs.append(entry)

        return inputs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
   
    config = load_config(config_path)
    train = config["train"]
    ansgen = PraiseAnswerGenerator(config)
    ansgen.load_model(config)
    if train == "train":
        ansgen.train(config["input_ag_train_path"], config["sft_data_path"])
        logger.info("Starting eval now...")
        eval_data = ansgen.load_data(config["input_ag_inference_path"])
        ansgen.inference(eval_data, config["output_ag_inference_path"])
        logger.info("Done with eval!")
    elif train == "eval":
        data = ansgen.load_data(config["input_ag_inference_path"])
        if "EFR_feedback"in config.keys():
            ansgen.get_feedback_for_ERF(data, config["output_ag_inference_path"] )
        else:
            ansgen.inference(data, config["output_ag_inference_path"])
        logger.info("Done with eval!")

