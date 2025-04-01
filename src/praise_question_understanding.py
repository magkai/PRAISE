import copy
import json
import sys
import random
import torch
from loguru import logger
sys.path.append(".")
sys.path.append("..")
from sacrebleu import corpus_bleu
from evaluate import load
import numpy as np
BERTSCORE = load("bertscore")
from praise_base_component import PraiseBaseComponent
from src.utils import load_config
from evaluation import is_answer_correct

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

##################################################################################
# QU module for creating self-sufficient questions based on the conversation history
##################################################################################
class PraiseQuestionUnderstanding(PraiseBaseComponent):

    INSTRUCTION = (
        "You are given a conversation and the current user question. "
        "Your task is to reformulate the question into a self-sufficient form, that clearly conveys the question intent, by using information from the conversation history. " 
        "Just create the standalone question without commentary and without answering the question. "
    )

    FEWSHOT_INSTRUCTION = (
        "You are given a conversation and the current user question. "
        "Your task is to reformulate the question into a self-sufficient form, that clearly conveys the question intent, by using information from the conversation history. " 
        "Just create the standalone question without commentary and without answering the question. "
        "Please learn from the examples below. "
    )


    ICL_EXAMPLES = [
        {
            "history": "who is the author of book The Velveteen Rabbit? Margery Williams. publisher? George H. Doran Company. illustrator? William Nicholson. Genre? children's novel. ",
            "question": "country of origin?",
            "qu": '''1. Which is the country of origin of book The Velveteen Rabbit?\n
                     2. Where is the book The Velveteen Rabbit from?\n
                     3. Where is Margery Williams from?\n
                     4. Country of origin The Velveteen Rabbit\n
                     5. Place where The Veleveteen Rabbit novel was created?'''
        },
        {
            "history": "For which country team did Paul Pogba play? France. Where was he born? Lagny-sur-Marne. ",
            "question": "position played on team?",
            "qu": '''1. At which position did Paul Pogba play on team?\n
                     2. What is Paul Pogba's postion in the France national team?\n
                     3. Paul Pogba positon on team France\n
                     4. Which position on the field Paul Pogba\n
                     5. What role does Paul Pogba occupy in the french team?'''
        },
        {
            "history": "how many seasons of Mindhunter series? 2. who is the director? David Fincher. ", 
            "question": "How many episodes?",
            "qu": '''1. How many episodes are in the Mindhunter series?
                     2. Number of episodes Mindhunter series\n
                     3. Total amount of episodes of Mindhunter\n
                     4. How many episodes are in David Fincher's Mindhunter series?\n
                     5. Episode count for Mindhunter'''
        },
        {
            "history": " ", 
            "question": "Who played as Mark Zuckerberg in The Social Network?",
            "qu": '''1. Who played as Mark Zuckerberg in The Social Network?
                     2. The Social Networ actor Mark Zuckerberg\n,
                     3. Cast member for Mark Zuckerberg in the movie The Social Network\n
                     4. Who acted as Mark Zuckerberg in The Social network movie\n
                     5. Who starred as Mark Zuckerberg in The Social Network?'''
        },
        {
            "history": "U2 had how many band members? 5. which year did it formed? 1976. what about the band drummer? Larry Mullen Jr. ", 
            "question": "who was their bass guitarist?",
            "qu":  '''1. Who was the bass guitarist of the U2 band?\n
                      2. Which U2 band member played the guitar?\n
                      3. Bass guitar player U2 band\n
                      4. U2 bass guitarist\n
                      5. Who played the guitar in the U2 band formed in 1976'''
        },
    ]


    PROMPT_TEMPLATE = (
        "history: {history}\n" 
        "question: {question}\n"
    )


    def __init__(self, config):
        self.config = config
        self.hyps = []
        self.refs = []
        self.PROMPT_KEY = "qu"
        config["model_path"] =  config["qu_model_path"]
        if "qu_peft_model_path" in config.keys():
            config["peft_model_path"] =  config["qu_peft_model_path"]
        config["adapter_name"] =  config["qu_adapter_name"]
        super(PraiseQuestionUnderstanding, self).__init__(config, self.PROMPT_KEY, self.INSTRUCTION, self.ICL_EXAMPLES)

    #add conversation history to input question
    def prepare_input_context(self, entry, idx=0):
        history = ""
        for k in range(idx):
            ansString = ""
            for ans in entry[k]["answers"]:
                ansString += ans["label"] + ";"
            history += entry[k]["question"] + " " + ansString + " "
        history = history.strip()
        question = entry[idx]["question"]
        return question, history
    

    def create_prompt(self, question, history, label=None):
        
        if "fewshot" in self.config.keys():
            dialog = [
            {
                "role": "system",
                "content": self.FEWSHOT_INSTRUCTION
            },
        ]
        else:
            dialog = [
                {
                    "role": "system",
                    "content": self.INSTRUCTION
                },
            ] 
        examples = self.ICL_EXAMPLES
        if "fewshot" in self.config.keys():
            for ex in examples:
                dialog.append({
                    "role": "user",
                    "content": self.PROMPT_TEMPLATE.format(history=ex["history"], question=ex["question"]),
                })
                dialog.append({
                    "role": "assistant",
                    "content": "qu: " + ex["qu"],
                })

        dialog.append({
            "role": "user",
            "content": self.PROMPT_TEMPLATE.format(history=history, question=question),
        })
        if not label is None:
            dialog.append({
                "role": "assistant",
                "content": "qu: " + label,
            })

        return dialog
        

    def calculate_metrics(self, input1, input2):
        rewritten_question = input1
        reference_question = input2
        self.hyps.append(rewritten_question)
        self.refs.append([reference_question])


    #calculate BLEU and bertscore with respect to completed questions provided in dataset (only used for analysis)
    def final_metric(self, count=None):
        bleuscore = corpus_bleu(self.hyps, self.refs).score
        bertscore = BERTSCORE.compute(predictions=self.hyps, references=self.refs, lang="en")
        bertscore_mean = np.mean(bertscore["precision"])
        logger.info(f"BLEU score: `{bleuscore}`")
        logger.info(f"BERT score: `{bertscore_mean}`")


    def inference_per_question(self, question, history):
        return self.generate_output(question, history)
    

    def inference(self, data, output_path):
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
                entry = data[i]["questions"][j]
                question, history = self.prepare_input_context(data[i]["questions"], j)
                if "completed" in entry.keys():
                    reference = entry["completed"]
                else:
                    reference = entry["question"]
                
                entry["rewritten_question"] = self.generate_output(question, history)
                self.calculate_metrics(entry["rewritten_question"], reference)
                      
        with open(output_path, "w") as fout:
            json.dump(data, fout)
        
        self.final_metric()

    
    def generate_samples(self, data, output_path):
        conversations = []
        for i in range(self.config["sample_size"]):
            conversations.append([])
       
        for i in range(len(data)):
            for s in range(self.config["sample_size"]):
                conversations[s].append(copy.deepcopy(data[i]))
            for j in range(len(data[i]["questions"])):
                # entry = data[i]["questions"][j]
                question, history = self.prepare_input_context(data[i]["questions"], j)
                if "fewshot_paraphrases" in self.config.keys():
                    rewritten_out = self.generate_output(question, history)
                    rewList = rewritten_out.split("\n")
                    rewritten_questions =[]
                    for rq in rewList:
                        rq = rq.strip()[2:]
                        if rq == "":
                            continue
                        rewritten_questions.append(rq)
                
                print("rewritten questions: ", rewritten_questions)
                for s in range(self.config["sample_size"]):
                    if len(rewritten_questions)<= s:
                        print("Warning: less generated questions than requested!")
                    conversations[s][i]["questions"][j]["rewritten_question"] = rewritten_questions[s%len(rewritten_questions)]
                             
        new_output_path = ""
        for s in range(self.config["sample_size"]):
            new_output_path = output_path.replace(".json", "_" + str(s) + ".json")
            with open(new_output_path, "w") as fout:
                json.dump(conversations[s], fout)


    #check for answer presence and answer correctness of sampled data
    def evaluate_feedback(self, input_path, pref_out_path):
        dataList = []
        sample_size = self.config["sample_size"]
   
        for i in range(sample_size):
            path = input_path.replace(".json",   str(i) + ".json")
            with open(path, "r") as efp:
                data = json.load(efp)
                dataList.append(data)
                
        all_pref_data = []
        stats_ans_pres = [0, 0, 0, 0, 0, 0]
        stats_corr_ans = [0, 0, 0, 0, 0, 0]
        for i in range(len(dataList[0])):
            for j in range(len(dataList[0][i]["questions"])):
            
                prefData = dict()
                prefData["question_id"] = dataList[0][i]["questions"][j]["question_id"]
                history = ""
                for h in range(j):
                    ansString = ""
                    for ans in dataList[0][i]["questions"][h]["answers"]:
                        ansString += ans["label"] + ";"
                    history += dataList[0][i]["questions"][h]["question"] + " " + ansString + " "
                history = history.strip()
                prefData["history"] = history
                prefData["original_question"] = dataList[0][i]["questions"][j]["question"]
             
                prefData["answer"] = []
                for a in dataList[0][i]["questions"][j]["answers"]:
                    prefData["answer"].append(a["label"]) 
            
                rew_questions = dict()
                gen_ans = dict()
                ans_pres_idx = []
                answers = dataList[0][i]["questions"][j]["answers"]
            
                for k in range(sample_size):
                    rew_questions.update({str(k): dataList[k][i]["questions"][j]["rewritten_question"]})
                    gen_ans.update({str(k): dataList[k][i]["questions"][j]["generated_answer"]})
                    evidence =  "\n".join(evidence[1] for evidence in  dataList[k][i]["questions"][j]["reranked_evidence"])
                    for ans in  answers:
                        if ans["label"].lower() in evidence.lower():
                            ans_pres_idx.append(k)
                            break
                    
                stats_ans_pres[len(ans_pres_idx)] += 1
                if len(ans_pres_idx) == 0:
                    continue

                prefData["pos_ans_pres"] = []
                prefData["neg_ans_pres"] = []
                for k in range(sample_size):
                    if k in ans_pres_idx:
                        if not rew_questions[str(k)].lower().strip() in [x.lower().strip() for x in prefData["pos_ans_pres"]]:
                            prefData["pos_ans_pres"].append(rew_questions[str(k)].strip())
                    else:
                        if not rew_questions[str(k)].lower().strip() in [x.lower().strip() for x in prefData["neg_ans_pres"]]:
                            prefData["neg_ans_pres"].append(rew_questions[str(k)].strip())
                
                correct_ans_idx = []
               
                for k in range(sample_size):
                    if is_answer_correct(dataList[k][i]["questions"][j]["generated_answer"], answers):
                        correct_ans_idx.append(k)
                  
                stats_corr_ans[len(correct_ans_idx)] += 1
                
                if len(correct_ans_idx) == 0:
                    print("no correct answer for any of the rewritten questions, qid: ", prefData["question_id"])
                    continue 
       
                prefData["pos_ans_corr"] = []
                prefData["neg_ans_corr"] = []
                for k in range(sample_size):
                    #only count as correct if answer also present in evidence, otherwise hallucination
                    #ans corr = answer pres + ans correct (both conditions fulfilled)
                    if not k in ans_pres_idx:
                        continue
                    if k in correct_ans_idx: 
                        if not rew_questions[str(k)].lower().strip() in [x.lower().strip() for x in prefData["pos_ans_corr"]]:
                            prefData["pos_ans_corr"].append(rew_questions[str(k)].strip())
                    else:   
                        if not rew_questions[str(k)].lower().strip() in [x.lower().strip() for x in prefData["neg_ans_corr"]]:
                            prefData["neg_ans_corr"].append(rew_questions[str(k)].strip())
                
                
                all_pref_data.append(copy.deepcopy(prefData))       

        print("answer pres stats: ", stats_ans_pres)
        print("correct ans stats: ", stats_corr_ans)

        with open(pref_out_path, "w") as fp:
            json.dump(all_pref_data, fp)


    #use the ones where both conditions (ans pres + ans corr) fullfilled to train SFT model 
    def prepare_sft_data(self, data): 
        inputs = []
        for entry in data:    
            if "pos_ans_corr" in entry.keys() and len(entry["pos_ans_corr"])>0:
                label =  entry["pos_ans_corr"][0]
                question = entry["original_question"]
                history = entry["history"]
               
                full_prompt = self.create_prompt(question, history, label)
                full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
        
                prompt = self.create_prompt(question, history)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                entry = dict()
                entry["prompt"] = prompt
                entry["full_prompt"] = full_prompt
                inputs.append(entry)
        return inputs


    #pair all possible combinations of pos/neg samples based on the two conditions (ans pres + ans corr) for DPO training
    def prepare_pref_data_allSamples(self, data):
        sampleList = []  
        anspres_count = 0
        anscorr_count = 0

        for entry in data:
            history = entry["history"]
            question = entry["original_question"]
            prompt = self.create_prompt(question, history)
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            if "pos_ans_pres" in entry.keys() and "neg_ans_pres" in entry.keys() and len(entry["pos_ans_pres"])>0 and len(entry["neg_ans_pres"])>0:
                for i in range(len(entry["pos_ans_pres"])):
                    for j in range(len(entry["neg_ans_pres"])):
                        sample = dict()
                        sample["prompt"] = prompt
                        sample["chosen"] = entry["pos_ans_pres"][i]
                        sample["rejected"] =  entry["neg_ans_pres"][j]
                        sampleList.append(copy.deepcopy(sample))
                        anspres_count +=1 
            elif "pos_ans_corr" in entry.keys() and "neg_ans_corr" in entry.keys() and len(entry["pos_ans_corr"])>0 and len(entry["neg_ans_corr"])>0:
                for i in range(len(entry["pos_ans_corr"])):
                    for j in range(len(entry["neg_ans_corr"])):
                        sample = dict()
                        sample["prompt"] = prompt
                        sample["chosen"] = entry["pos_ans_corr"][i]
                        sample["rejected"] =  entry["neg_ans_corr"][j]
                        sampleList.append(copy.deepcopy(sample))
                        anscorr_count += 1
           
        print("ans presence: ", anspres_count)
        print("ans correctness: ", anscorr_count)
        
        return sampleList

    #alternative: randomly select one pos/neg pair per original questions and per condition (ans pres + ans corr) for DPO training
    def prepare_pref_data(self, data):
        sampleList = []  
        anspres_count = 0
        anscorr_count = 0
        if "allSamples" in self.config.keys():
            return self.prepare_pref_data_allSamples(data)
        
        for entry in data:
            sample = dict()
            history = entry["history"]
            question = entry["original_question"]
            if "pos_ans_pres" in entry.keys() and "neg_ans_pres" in entry.keys() and len(entry["pos_ans_corr"])>0 and len(entry["neg_ans_pres"])>0:
                idx_pos = random.sample(list(enumerate(entry["pos_ans_pres"])), 1)[0][0]   
                idx_neg = random.sample(list(enumerate(entry["neg_ans_pres"])), 1)[0][0]   
                prompt = self.create_prompt(question, history)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                sample["prompt"] = prompt
                sample["chosen"] = entry["pos_ans_pres"][idx_pos]
                sample["rejected"] =  entry["neg_ans_pres"][idx_neg]
                sampleList.append(copy.deepcopy(sample))
                anspres_count +=1 
            elif "pos_ans_corr" in entry.keys() and "neg_ans_corr" in entry.keys() and len(entry["pos_ans_corr"])>0 and len(entry["neg_ans_corr"])>0:
                sample = dict()
                idx_pos = random.sample(list(enumerate(entry["pos_ans_corr"])), 1)[0][0]   
                idx_neg = random.sample(list(enumerate(entry["neg_ans_corr"])), 1)[0][0] 
                prompt = self.create_prompt(question, history)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                sample["prompt"] = prompt
                sample["chosen"] = entry["pos_ans_corr"][idx_pos]
                sample["rejected"] =  entry["neg_ans_corr"][idx_neg]
                sampleList.append(copy.deepcopy(sample))
                anscorr_count += 1
           
        print("ans presence: ", anspres_count)
        print("ans correctness: ", anscorr_count)
        
        return sampleList


if __name__ == "__main__":
   
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        logger.error("no config provided")
        exit(-1)
    config = load_config(config_path)
    train = config["train"]
    qu = PraiseQuestionUnderstanding(config)
    qu.load_model(config)

    if train == "train":
        qu.evaluate_feedback(config["feedback_qu_path"], config["input_qu_train_path"])
        if config["train_type"] == "SFT":
            qu.train(config["input_qu_train_path"], config["sft_data_path"])
        elif config["train_type"] == "DPO":
            qu.train_dpo(config["input_qu_train_path"], config["preference_data_path"])

        logger.info("Starting eval now...")
        eval_data = qu.load_data(config["input_qu_inference_path"])
        qu.inference(eval_data, config["output_qu_inference_path"])
        logger.info("Done with eval!")
    elif train == "eval":
        if "sampling" in config.keys() and config["sampling"]:
            sample_data = qu.load_data(config["input_qu_sample_path"])
            qu.generate_samples(sample_data, config["output_qu_sample_path"])
            logger.info("Done with sampling!")
        else:
            eval_data = qu.load_data(config["input_qu_inference_path"])
            qu.inference(eval_data, config["output_qu_inference_path"])
            logger.info("Done with eval!")
