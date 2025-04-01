
import json
import torch
import sys
import logging
import time
import random

random.seed(7)
from utils import  load_config, retrieve_evidence_clocq_bm25
from praise_answer_generator import PraiseAnswerGenerator
from praise_evidence_filtering import PraiseEvidenceFilter
from praise_question_understanding import PraiseQuestionUnderstanding
from evaluation import is_answer_correct


class PraisePipeline():

    def __init__(self, config):
        
        self.config = config
        self.logger = logging.getLogger(__name__)

        # load individual modules and its model adapters
        self.qu =  PraiseQuestionUnderstanding(config)
        self.model = self.qu.load_model(config)
        self.erf = PraiseEvidenceFilter(config)
        self.erf.load_clocq(self.config)
        self.ag = PraiseAnswerGenerator(config)

        self.model.load_adapter(config["erf_peft_model_path"], adapter_name="erf_adapter")
        self.model.load_adapter(config["ag_peft_model_path"], adapter_name="ag_adapter")

      
    def run_inference(self):
        #rewrite questions
        eval_qu_data = self.qu.load_data(self.config["input_qu_inference_path"])
        self.qu.inference(eval_qu_data, self.config["output_qu_inference_path"])
        self.logger.info("question understanding completed")

        #retrieve and filter evidence
        self.model.set_adapter("erf_adapter")
        self.erf.set_model(self.model)
        self.erf.set_generation_pipeline(self.model)
        data = self.erf.load_data(self.config["output_qu_inference_path"])
        retrieve_evidence_clocq_bm25(self.erf.clocq, self.config["output_qu_inference_path"], self.config["evidence_inference_path"]) 
        ev_data = self.erf.load_data(self.config["evidence_inference_path"])
        self.erf.add_evidence_info(data, ev_data, self.config["input_erf_inference_path"])         
        eval_erf_data = self.erf.load_data(self.config["input_erf_inference_path"])
        self.erf.inference(eval_erf_data, self.config["output_erf_inference_path"])
        self.logger.info("evidence retrieval and filtering completed")

        #generate answer based on evidence
        self.model.set_adapter("ag_adapter")
        self.ag.set_model(self.model)
        self.ag.set_generation_pipeline(self.model)
        eval_ag_data = self.ag.load_data(self.config["input_ag_inference_path"])
        self.ag.inference(eval_ag_data, self.config["output_ag_inference_path"])
        self.logger.info("answer generation completed")
       

    
    def run_inference_per_question(self, question, history,answer):
       
        entry = dict()
        entry["question"] = question
        entry["history"] = history
        entry["gold_answer"] = answer

        self.model.set_adapter("qu_adapter")
        self.qu.set_model(self.model)
        self.qu.set_generation_pipeline(self.model)
        rewritten_question = self.qu.inference_per_question(question, history)
        entry["rewritten_question"] = rewritten_question
        
        self.model.set_adapter("erf_adapter")
        self.erf.set_model(self.model)
        self.erf.set_generation_pipeline(self.model)
        reranked_evidence = self.erf.inference_per_question(rewritten_question)
        entry["reranked_evidence"] = reranked_evidence
       
        self.model.set_adapter("ag_adapter")
        self.ag.set_model(self.model)
        self.ag.set_generation_pipeline(self.model)
        final_answer = self.ag.inference_per_question(rewritten_question, reranked_evidence)
        entry["generated_answer"] = final_answer
       
        return entry


    def main(self):
        
        if function == "--inference":
            praise.run_inference()
        
        elif function == "--inference_per_question":
            data = praise.qu.load_data(config["input_qu_inference_path"])
            datalist = []
            for i in range(len(data)):
                for j in range(len(data[i]["questions"])):
                    entry = data[i]["questions"][j]
                    question, history = praise.qu.prepare_input_context(data[i]["questions"], j)
                    datalist.append([question, history, entry["answers"]])

            random.shuffle(datalist)   
            results = []
            for question,history,answer in datalist:  
                result = praise.run_inference_per_question(question, history,answer)
                results.append(result)   
            p_at_1 =  0
            c = 0
            for res in results:
                c+=1
                if is_answer_correct(res["generated_answer"], res["gold_answer"]):
                    p_at_1 += 1
                    res["p_at_1"] = 1
                else:
                    res["p_at_1"] = 0
            results[-1]["avg_p_at_1"] = p_at_1/c
            with open(config["output_inference_path"], "w") as fout:
                json.dump(results, fout)
            

if __name__ == "__main__":
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = load_config(config_path)
    praise = PraisePipeline(config)
    praise.main()
   







    