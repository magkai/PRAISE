import json
import re
import copy
import utils as ut
import os
import random

random.seed(7)

def load_data(path):
    if os.path.exists(path):
        if path.endswith(".jsonl"):
            with open(path, "r") as efp:
                evidence_data = []
                line = efp.readline()
                
                while line:
                    conversation = json.loads(line)
                    evidence_data.append(conversation)
                    line = efp.readline()
        else:
            with open(path, "r") as efp:
                evidence_data = json.load(efp)
   
    return evidence_data
    

def get_mrr(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for i in range(len(answers)):
        if is_answer_correct(answers[i], gold_answers):
            return 1.0 / (i+1)
    return 0.0


def get_hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for i in range(5):
        if len(answers)<i:
            return 0.0
        if is_answer_correct(answers[i], gold_answers):
            return 1.0
      
    return 0.0


def get_p_at_1(generated_answer, gold_answers):
    if isinstance(generated_answer, list):
        generated_answer = generated_answer[0]
    if is_answer_correct(generated_answer, gold_answers):
        return 1.0
    else:
        return 0.0


def is_answer_correct(generated_answer, gold_answers):
    if not isinstance(generated_answer, str):
        print("bad answer: ", generated_answer)
        generated_answer = str(generated_answer)
    for ans in gold_answers:
        if ut.is_timestamp(ans["id"]):
            day, month, year = ut.convertTimestamp(ans["id"])
            if ut.check_correct_time(generated_answer, day, month, year):
               return True
        if ans["label"].lower() in generated_answer.lower():
            return True
     
    return False
   

def sort_evs_by_score(dict):
    return dict['score']


def get_answer_presence(data):
    anspres = 0
    count = 0
    breakFlag = False
    for i in range(len(data)):
        for j in range(len(data[i]["questions"])):
            count += 1
            top_evs = data[i]["questions"][j]["reranked_evidences"] 
            #top_evs = data[i]["questions"][j]["top_evidences"] 
            for ev in top_evs:
                for ans in  data[i]["questions"][j]["answers"]:
                    if ans["label"].lower() in ev[1].lower():
                        anspres += 1
                        breakFlag = True
                        break
                if breakFlag:
                    breakFlag = False
                    break
    print("ans count: ", anspres)
    print("all: ", count)
    print("answer presence: ", anspres/count)
  
  
def evaluate(data):
    p_at_1 = 0
    hit_at_5 = 0
    mrr = 0
    count = 0
    for i in range(len(data)):
        for j in range(len(data[i]["questions"])):
            count += 1
            generated_answer = data[i]["questions"][j]["generated_answer"]
            gold_answers = data[i]["questions"][j]["answers"]
            p_at_1 += get_p_at_1(generated_answer, gold_answers)
            hit_at_5 += get_hit_at_5(generated_answer, gold_answers)
            mrr += get_mrr(generated_answer, gold_answers)

    print("p@1: ", p_at_1/count)
    print("hit@5: ", hit_at_5/count)
    print("mrr: ", mrr/count)


def calculate_reranking_metrics(input1, input2):
        generated_evs = input1
        gold_evs = input2
        pc = 0
        precision = 0
        recall = 0
        relevant  = False
        for rev in generated_evs:
            if rev.strip() == "r":
                continue
            if rev.strip() in gold_evs:
                relevant = True
                pc += 1
        if len(generated_evs)> 0:
            precision = pc/len(generated_evs)
        if len(gold_evs)>0:
            recall = pc/len(gold_evs)
       
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2* ((precision*recall)/(precision+recall))
         
        return precision, recall, f1, relevant

