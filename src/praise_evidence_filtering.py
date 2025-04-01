import copy
import json
import torch
import pyarrow
import sys
import os

from loguru import logger
sys.path.append(".")
sys.path.append("..")
from src.utils import  load_config, sort_evs_by_score, retrieve_evidence_per_question_clocq_bm25, retrieve_evidence_clocq_bm25
from sentence_transformers import SentenceTransformer
from praise_base_component import PraiseBaseComponent
import random
import wandb
import math
import numpy as np
import time
sys.path.append("../EXPLAIGNN")
sys.path.append("../EXPLAIGNN/CLOCQ")
from explaignn.evidence_retrieval_scoring.clocq_bm25 import ClocqBM25



random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

MAXTIMES = 10
MAX_POS_EV = 10
IGNORE_INDEX=-100

###################################################
# ERF module for evidence retrieval and filtering
###################################################
class PraiseEvidenceFilter(PraiseBaseComponent):
    def __init__(self, config):
        self.config = config
        self.precision_all = 0
        self.recall_all = 0
        self.f1_all = 0
        self.INSTRUCTION = ("You are given a question and a set of references, each with a unique identifier. "
               "Your task is to judge which references provide relevant information for answering the given question. "
               "You should output the corresponding ids of the most relevant references, nothing else. "
              #  "If none of the references is relevant, then output -1. "
             )
        self.ICL_EXAMPLES = None
        dev =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=dev #, cache_dir="checkpoints",
        )
        config["model_path"] =  config["erf_model_path"]
        config["peft_model_path"] =  config["erf_peft_model_path"]
        config["adapter_name"] =  config["erf_adapter_name"]
        super(PraiseEvidenceFilter, self).__init__(config, "answer", self.INSTRUCTION, self.ICL_EXAMPLES)


    def load_clocq(self, config):
        self.clocq = ClocqBM25(config)


    def get_clocq(self):
        return self.clocq

    #add retrieved evidence
    def add_evidence_info(self, data, ev_data, path):
        for i in range(len(data)): 
            for j in range(len(data[i]["questions"])):
                entry = data[i]["questions"][j]       
                evs = ev_data[i]["questions"][j]["top_evidences"]
                answers = entry["answers"]
                entry["retrieved_evidence"] = []
                entry["reranked_evidence"] = []
                entry["noisy_gold_evidence"] = []
                entry["distractor_evidence"] = []
                if len(evs) == 0:
                    continue
                random.shuffle(evs)
                idx = 0
                non_relevant = []
                #assign random ids to evidence
                for ev in evs:
                    ev["id"] = "r" + str(idx)
                    idx += 1

                evs.sort(key=sort_evs_by_score, reverse=True)
                for ev in evs:
                    #initialize/limit reranked evidence to 'top_k'
                    if len(entry["reranked_evidence"]) < self.config["top_k"]:
                        entry["reranked_evidence"].append([ev["id"], ev["evidence_text"]])
                    entry["retrieved_evidence"].append([ev["id"], ev["evidence_text"]])
                    found =  False
                    #get noisy gold evidence
                    for ans in answers:
                        if ans["label"].lower() in ev["evidence_text"].lower():
                            found = True
                            entry["noisy_gold_evidence"].append([ev["id"], ev["evidence_text"]])
                            break
                        if not found:
                            non_relevant.append([ev["id"], ev["evidence_text"]])
                #get most similar but non relevant evidence (used for better training)
                if self.config["sbert"]:
                    entry["distractor_evidence"] = self.get_distractor_evs(entry["question"], entry["retrieved_evidence"],  entry["noisy_gold_evidence"])
                else:
                    entry["distractor_evidence"]  = non_relevant[:self.config["top_k"]]
            
        with open(path, "w") as fout:
            json.dump(data, fout)

      
    def get_distractor_evs(self, query, evidence, gold_evs):
   
        # Initialize a list to hold all extracted sentences from the search results
        all_sentences = [ev[1] for ev in evidence]
        all_ids = [ev[0] for ev in evidence]
        gold_ids = [ev[0] for ev in gold_evs]
       
        # Generate embeddings for all sentences and the query
        all_embeddings = self.sbert_model.encode(
            all_sentences,
            normalize_embeddings=True
        )
        query_embedding = self.sbert_model.encode(
            query,
            normalize_embeddings=True
        )[None, :]

        # Calculate cosine similarity between query and sentence embeddings, and select the top sentences
        cosine_scores = (all_embeddings * query_embedding).sum(1)
        top_sentences = np.array(all_sentences)[
            (-cosine_scores).argsort()[:(self.config["top_k"])]
        ]
        top_ids  = np.array(all_ids)[
            (-cosine_scores).argsort()[:(self.config["top_k"])]
        ]
       
        distractor_evs = []
        for i in range(len(top_ids)):
            if not top_ids[i] in gold_ids:
                distractor_evs.append([top_ids[i],top_sentences[i]])
        
        return distractor_evs


    #calculate precision, recall and f1 based on whether generation contains noisy ground truth evidence ids
    def calculate_metrics(self, input1, input2):
        generated_evs = input1
        gold_evs = input2
        pc = 0
        precision = 0
        recall = 0
        genlen = 0
        for rev in generated_evs:
            if rev.strip == "":
                continue
            if rev.strip() == "r":
                continue
            if rev.strip() in gold_evs:
                pc += 1
            genlen +=1
        if genlen> 0:
            precision = pc/genlen
        if len(gold_evs)>0:
            recall = pc/len(gold_evs)
       
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2* ((precision*recall)/(precision+recall))
        
        self.precision_all += precision
        self.recall_all += recall
        self.f1_all += f1
        return precision, recall, f1
    

    def final_metrics(self, count):
        logger.info(f"Precision: `{self.precision_all/count}`")
        logger.info(f"Recall: `{self.recall_all/count}`")
        logger.info(f"F1: `{self.f1_all/count}`")


    def inference_per_question(self, rewritten_question):
        #retrieve evidence
        retrieved_evidence = retrieve_evidence_per_question_clocq_bm25(self.clocq, rewritten_question)
        evlen = len(retrieved_evidence)
        klen = self.config["top_k"]         
        times = math.ceil(evlen/klen)
        idx = 0
        all_evs = []
        #assign random ids
        for ev in retrieved_evidence:
            ev["id"] = "r" + str(idx)
            all_evs.append([ev["id"], ev["evidence_text"]])
            idx += 1
      
        retrieved_evidence.sort(key=sort_evs_by_score, reverse=True)
        evDict  = dict()
        for id,val in all_evs:
            evDict[id] = val
        reranked_evidence = []
        random.shuffle(all_evs)
        #chunk evidence into smaller sets and predict relevant ids for each set, final output is union of individual predictions
        for t in range(times):               
            if t == MAXTIMES:
                break
            current_evs = all_evs[t*klen: (t+1)*klen]
            joint_evidences = "\n".join(candidate_ev[0] + ": " + candidate_ev[1] for candidate_ev in  current_evs)
        
            genStr = self.generate_output(rewritten_question, joint_evidences)
            if "," in genStr:
                genList = genStr.split(",")
                for gstr in genList:
                    if not gstr.startswith('r'):
                        gstr = 'r'+ gstr
            else:
                genList = [genStr]
            
            for evid in genList:
                if evid in evDict.keys():
                    reranked_evidence.append([evid, evDict[evid]])

        #use previous BM25 ranking to obtain up to 'top_k' evidence
        if "fill_to_top_k" in self.config.keys():
                for ev in retrieved_evidence:
                    if len(reranked_evidence) >= klen:
                        break
                    if not ev["id"] in genList:
                        reranked_evidence.append([ev["id"], ev["evidence_text"]])

        return reranked_evidence


    def inference(self, data, output_path):
       
        with open(output_path, "w") as fp:
            for i in range(len(data)): 
                for j in range(len(data[i]["questions"])):
                    entry = data[i]["questions"][j]
                    if "completed_question" in self.config.keys():
                        if "completed" in entry.keys():
                            question = entry["completed"]
                        else:
                            question = entry["question"]
                    else:
                        question = entry["rewritten_question"]
                    # retrieve evidence if not already done
                    if not "retrieved_evidence" in entry.keys():
                        retrieved_evidence = retrieve_evidence_per_question_clocq_bm25(self.clocq, question)
                        retrieved_evidence.sort(key=sort_evs_by_score, reverse=True)
                        entry["retrieved_evidence"] = []
                        idx = 0
                        for ev in retrieved_evidence:
                            ev["id"] = "r" + str(idx)
                            entry["retrieved_evidence"].append([ev["id"], ev["evidence_text"]])
                            idx += 1
                      
                    evlen = len(entry["retrieved_evidence"])
                    klen = self.config["top_k"]
                    times = math.ceil(evlen/klen)
                    evslist = copy.deepcopy(entry["retrieved_evidence"])
                    evDict  = dict()
                    for id,val in evslist:
                        evDict[id] = val
                    entry["reranked_evidence"] = []
                    random.shuffle(evslist)
                    #chunk evidence into smaller sets and predict relevant ids for each set, final output is union of individual predictions
                    for t in range(times):               
                        if t == MAXTIMES:
                            break
                        current_evs = evslist[t*klen: (t+1)*klen]
                        joint_evidences = "\n".join(candidate_ev[0] + ": " + candidate_ev[1] for candidate_ev in  current_evs)
                        genStr = self.generate_output(question, joint_evidences)
                        if "," in genStr:
                            genList = genStr.split(",")
                            for gstr in genList:
                                if not gstr.startswith('r'):
                                    gstr = 'r'+ gstr
                        else:
                            genList = [genStr]
                        for evid in genList:
                            if evid in evDict.keys():
                                entry["reranked_evidence"].append([evid, evDict[evid]])
                     
                    #use previous BM25 ranking to obtain up to 'top_k' evidence
                    if "fill_to_top_k" in self.config.keys():
                        for evid,evtext  in entry["retrieved_evidence"]:
                            if len(entry["reranked_evidence"]) >= klen:
                                break
                            if not evid in genList:
                                entry["reranked_evidence"].append([evid, evtext])
                
                fp.write(json.dumps(data[i]))
                fp.write("\n")
         
       
    #sample multiple evidence id sets per question from SFT model (used for collecting feedback from AG model, for later DPO training of the ERF model)
    def generate_samples(self, data, output_path):
      
        with open(output_path, "w") as fp:
            for i in range(len(data)):
                for j in range(len(data[i]["questions"])):
                    current_data = data[i]["questions"][j]
                    question = current_data["rewritten_question"]
                    klen = self.config["top_k"]
                    
                    all_evs = current_data["retrieved_evidence"]
                    if len(all_evs) == 0: 
                        continue
                  
                    #input for sampling: noisy ground truth + distractor evidence; labels: noisy ground truth 
                    candidate_evs = copy.deepcopy(current_data["noisy_gold_evidence"])
                   
                    dist_num = klen - len(current_data["noisy_gold_evidence"])
                    if dist_num > 0:
                        candidate_evs.extend(current_data["distractor_evidence"][:dist_num])
                  
                    candidate_evs[:klen]
                    random.shuffle(candidate_evs)

                    joint_evidences = "\n".join(candidate_ev[0] + ": " + candidate_ev[1] for candidate_ev in  candidate_evs)
                    
                    current_data["reranked_evidence"] = []
                    result = self.generate_sampled_output(question, joint_evidences)
                    for r in range(len(result)):
                        genStr = result[r]
                        current_data["reranked_evidence"].append([])
                        if "," in genStr:
                            genList = genStr.split(",")
                            genList = [gstr.strip() for gstr in genList]
                            for gstr in genList:
                                if not gstr.startswith('r'):
                                    gstr = 'r'+ gstr
                        else:
                            genList = [genStr]
                   
                        for evid in genList:
                            if evid in all_evs.keys():
                                current_data["reranked_evidence"][r].append([evid, all_evs[evid]])
                    #add noisy ground truth to check with AG model if leads to correct answer, as additional candidate
                    current_data["reranked_evidence"].append(current_data["noisy_gold_evidence"])
                fp.write(json.dumps(data[i]))
                fp.write("\n")


    #create SFT training data
    def prepare_sft_data(self, data):
        inputs = []
        emptyLabelCount = 0
       
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
                current_data = data[i]["questions"][j]
                question = current_data["rewritten_question"]
                evs = current_data["reranked_evidence"]
                gold_evs = current_data["noisy_gold_evidence"]
                distractor_evs = current_data["distractor_evidence"]
                klen = self.config["top_k"]
              
                if len(evs) == 0 or len(gold_evs) == 0:
                    print("no evs or goldevs!!!!")
                    continue
                if len(gold_evs)> MAX_POS_EV:
                    print("noisy sample: too many pseudo positive evs")
                    continue
                emptyLabelCount +=1 
                entry = dict() 
                #SFT training data: input: noisy ground truth + distractor evidence
                candidate_evs = copy.deepcopy(gold_evs)
                labellist = [cev[0] for cev in candidate_evs]
                labels = ", ".join(labellist)
                dist_num = klen - len(gold_evs)
                candidate_evs.extend(distractor_evs[:dist_num])
                random.shuffle(candidate_evs)

                joint_evidences = "\n".join(candidate_ev[0] + ": " + candidate_ev[1] for candidate_ev in  candidate_evs)
                prompt = self.create_prompt(question, joint_evidences)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                full_prompt = self.create_prompt(question, joint_evidences,label=labels)
                full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
                entry["prompt"] = prompt
                entry["full_prompt"] = full_prompt
                inputs.append(copy.deepcopy(entry))
                #add cases where no relevant evidence is available
                if emptyLabelCount%1000== 0:
                    entry = dict() 
                    labels = ""
                    candidate_evs = distractor_evs[:klen]
                    random.shuffle(candidate_evs)
                    joint_evidences = "\n".join(candidate_ev[0] + ": " + candidate_ev[1] for candidate_ev in  candidate_evs)
                    prompt = self.create_prompt(question, joint_evidences)
                    prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                    full_prompt = self.create_prompt(question, joint_evidences,label=labels)
                    full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
                    entry["prompt"] = prompt
                    entry["full_prompt"] = full_prompt
                    inputs.append(copy.deepcopy(entry))

        return inputs

    #create DPO training data
    def prepare_pref_data(self, data):
        inputs = []
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
               
                current_entry = data[i]["questions"][j]
                question = current_entry["rewritten_question"]
                if not "noisy_gold_evidence" in current_entry.keys():
                    continue
                gold_evs = current_entry["noisy_gold_evidence"] 
                distractor_evs = current_entry["distractor_evidence"]
                klen = self.config["top_k"]
                #input: noisy ground truth + distractor evidence
                candidate_evs =  copy.deepcopy(gold_evs)
                dist_num = klen - len(gold_evs)
                if dist_num > 0:
                    candidate_evs.extend(distractor_evs[:dist_num])
                candidate_evs[:klen]
                
                random.shuffle(candidate_evs)
                joint_evidences = "\n".join(candidate_ev[0] + ": " + candidate_ev[1] for candidate_ev in  candidate_evs)
                prompt = self.create_prompt(question, joint_evidences)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                if not "reranked_evidence" in current_entry.keys():
                    continue
                pos_evs = []
                neg_evs = []
                #if set of evidence lead AG to correct answer: positive set, otherwise: negative set
                for r in range(len(current_entry["reranked_evidence"])):
                    if current_entry["p_at_1"][r] == 0:
                        neg_evs.append(current_entry["reranked_evidence"][r])
                    elif current_entry["p_at_1"][r] == 1:
                        pos_evs.append(current_entry["reranked_evidence"][r])
               
                if  len(neg_evs) == 0:
                    continue
                if len(pos_evs) == 0:
                    continue
                
                #sample in order to create one of <correct,incorrect> pairs
                pos_ev = random.sample(pos_evs,1)[0]
                pos_ev_ids =  ",".join([pev[0] for pev in pos_ev])

                neg_ev = random.sample(neg_evs,1)[0]
                neg_ev_ids = ",".join([nev[0] for nev in neg_ev])

                entry = dict()
                entry["prompt"] = prompt 
                entry["chosen"] = self.tokenizer.apply_chat_template(self.add_assistant_ouput(pos_ev_ids), tokenize=False).replace("<|begin_of_text|>", "")
                entry["rejected"] = self.tokenizer.apply_chat_template(self.add_assistant_ouput(neg_ev_ids), tokenize=False).replace("<|begin_of_text|>", "")      
            
                inputs.append(copy.deepcopy(entry))
                
        return inputs


    def do_retrieval(self, datapath, evidence_path, outpath):
        #check if evidence was retrieved already
        if not os.path.exists(evidence_path):
            self.load_clocq(config)
            retrieve_evidence_clocq_bm25(self.clocq, datapath, evidence_path) 
        
        ev_data = reranker.load_data(evidence_path)
        if not os.path.exists(outpath):
            reranker.add_evidence_info(data, ev_data, outpath)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
   
    config = load_config(config_path)
    train = config["train"] 
    reranker = PraiseEvidenceFilter(config)
    reranker.load_model(config)
    
    #prepare retrieval data
    if os.path.exists(config["output_qu_inference_path"]):
        datapath = config["output_qu_inference_path"]
        data = reranker.load_data(config["output_qu_inference_path"])
        if train == "train":
            evidence_path = config["evidence_train_path"]
            outpath =  config["input_erf_train_path"]
            reranker.do_retrieval(datapath, evidence_path, outpath)
            evidence_path = config["evidence_inference_path"]
            outpath =  config["input_erf_inference_path"]
            reranker.do_retrieval(datapath, evidence_path, outpath)
        else:
            evidence_path = config["evidence_inference_path"]
            outpath = config["input_erf_inference_path"]    
            reranker.do_retrieval(datapath, evidence_path, outpath) 
    else:
        #sampled data is used
        for i in range(config["sample_size"]):
            datapath = config["output_qu_inference_path"].replace(".json",  str(i) + ".json")
            data = reranker.load_data(datapath)
            if train == "train":
                evidence_path = config["evidence_train_path"].replace(".jsonl",  str(i) + ".jsonl")
                outpath =  config["input_erf_train_path"].replace(".json",  str(i) + ".json")
                reranker.do_retrieval(datapath, evidence_path, outpath)
                evidence_path = config["evidence_inference_path"].replace(".jsonl",  str(i) + ".jsonl")
                outpath =  config["input_erf_inference_path"].replace(".json",  str(i) + ".json")
                reranker.do_retrieval(datapath, evidence_path, outpath)
            else:
                evidence_path = config["evidence_inference_path"].replace(".jsonl",  str(i) + ".jsonl")
                outpath = config["input_erf_inference_path"].replace(".json",  str(i) + ".json")
                reranker.do_retrieval(datapath, evidence_path, outpath)
           
    #train and evaluate
    eval_data = reranker.load_data(config["input_erf_inference_path"]) 
    if train == "train":
        if config["train_type"] == "SFT":
            reranker.train(config["input_erf_train_path"], config["sft_data_path"])
        elif config["train_type"] == "DPO":
            reranker.train_dpo(config["input_erf_train_path"], config["preference_data_path"])
        logger.info("Starting eval now...")
        reranker.inference(eval_data, config["output_erf_inference_path"])
        logger.info("Done with eval!")
    elif train == "eval":
        if "sampling" in config.keys():
            reranker.generate_samples(eval_data, config["output_erf_inference_path"])
        else:
            reranker.inference(eval_data, config["output_erf_inference_path"])
            