import copy
import json
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model, PeftModel
from loguru import logger
sys.path.append(".")
sys.path.append("..")
from src.utils import  load_config, sort_evs_by_score
import random
from pynvml import *
from evaluation import is_answer_correct, get_hit_at_5, get_mrr
from datasets import Dataset

IGNORE_INDEX = -100

############################################
# End-to-end model for baseline comparison
############################################
class E2EModel():

    def __init__(self, config, train=False):
        self.p_at_1 = 0
        self.hit_at_5 = 0
        self.mrr  = 0
        self.INSTRUCTION = ( "You are given a conversation and the current user question. "
               "Please answer the question in as few words as possible.  "
              # "Please learn from the examples below. "
               )
        self.INSTRUCTION_FEWSHOT = ( "You are given a conversation and the current user question. "
               "Please answer the question in as few words as possible.  "
               "Please learn from the examples below. "
               )
        self.ICL_EXAMPLES = [
        {
            "history": "who is the author of book The Velveteen Rabbit? Margery Williams. publisher? George H. Doran Company. illustrator? William Nicholson. Genre? children's novel. ",
            "question": "country of origin?",
            "answer": "United States of America"
        },
        {
            "history": "For which country team did Paul Pogba play? France. Where was he born? Lagny-sur-Marne. ",
            "question": "position played on team?",
            "answer":   "midfielder"
        },
        {
            "history": "how many seasons of Mindhunter series? 2. who is the director? David Fincher. ", 
            "question": "How many episodes?",
            "answer":  "19"
        },
        {
            "history": "", 
            "question": "Who played as Mark Zuckerberg in The Social Network?",
            "answer":  "Jesse Eisenberg"
        },
        {
            "history": "U2 had how many band members? 5. which year did it formed? 1976. what about the band drummer? Larry Mullen Jr. ", 
            "question": "who was their bass guitarist?",
            "answer":  "Adam Clayton"
        }

        ]
        self.COMP_ID = "answer" 
        

        self.PROMPT_TEMPLATE = (
            "history: {history}\n" 
            "question: {question}\n"
        )

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], cache_dir="checkpoints", token=config["hf_token"])
        model_path = config["model_path"]

        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules= ["q_proj", "v_proj", "k_proj"], 
            modules_to_save= ["embed_tokens", "lm_head"],
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.debug(f"Loading model from {model_path}")
        if config["pad"]:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token_id = pad_token_id
        
        if "flash" in self.config.keys():
            self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="checkpoints", token=config["hf_token"], device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="checkpoints", token=config["hf_token"], device_map="auto")
        self.model.resize_token_embeddings(len(self.tokenizer))
        if  config["train"] == "train" and "peft_ref" in config.keys() and config["peft_ref"]:
            if "flash" in self.config.keys():
                self.model = PeftModel.from_pretrained(
                    self.model,
                    config["peft_model_path"],
                    is_trainable=True,
                    adapter_name="trainedSFT",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            else:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    config["peft_model_path"],
                    is_trainable=True,
                    adapter_name="trainedSFT",
                )
            # Load the adapter a second time, with a different name, which will be our reference model.
            self.model.load_adapter(config["peft_model_path"], adapter_name="reference")

        if config["peft"]:
            if config["train"] == "train":
                self.model = get_peft_model(self.model, self.peft_config)
                print(self.model.print_trainable_parameters())
            else:
                self.model = PeftModel.from_pretrained(self.model, config["peft_model_path"])
                self.model = self.model.merge_and_unload()     

        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=75,
        )
     
 
    def final_metrics(self, count):
        logger.info(f"Precision: `{self.p_at_1/count}`")
        logger.info(f"Hit@5: `{self.hit_at_5/count}`")
        logger.info(f"MRR: `{self.mrr/count}`")


    def generate_sampled_output(self, question: str, context: dict = {},  create_prompt=None): 
        
        if "temperature" in self.config.keys():
            temp = self.config["temperature"]
        else:
            temp = None
        gen_kwargs_sample= {
            "min_length": -1,
            #"top_k": self.config["top_k_sampling"],
            #"top_p": self.config["top_p_sampling"],
            "do_sample": self.config["do_sample"],
            "pad_token_id": self.tokenizer.pad_token_id,
            #"max_new_tokens": max_length,
            "num_beams": self.config["num_beams"],
            #"num_beam_groups":  self.config["num_beam_groups"],
            #"diversity_penalty":self.config["diversity_penalty"],
            "num_return_sequences": self.config["sample_size"],
            "temperature": temp
        }
        
        if create_prompt is None:
            prompt = self.create_prompt(question, context)
        else:
            prompt = create_prompt(question, context)
        
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_encodings = self.tokenizer.encode(prompt, truncation=True, max_length=2048, return_tensors='pt')
        input_encodings = input_encodings.to(device)
       
        with torch.no_grad():
            sampled_out = self.model.generate(
                input_encodings,
                **gen_kwargs_sample,
            )
            
        sampled_out = sampled_out.cpu().numpy()
        if isinstance(sampled_out, tuple):
           sampled_out = sampled_out[0]
        
        sampled_responses = self.tokenizer.batch_decode(sampled_out, skip_special_tokens=True)  
        for s in range(self.config["sample_size"]):
            if  self.COMP_ID+": " in sampled_responses[s]:
                sampled_responses[s] = sampled_responses[s].split( self.COMP_ID+": ")[-1].strip()
            if "assistant" in sampled_responses[s]:
                sampled_responses[s] = sampled_responses[s].split("assistant")[-1].strip()
            if isinstance(sampled_responses[s], list):
                sampled_responses[s] = sampled_responses[s][0]
       
        print("sampled answers: ", sampled_responses)
        return sampled_responses


    def generate_output(self, question: str, context: dict = {}, create_prompt=None): 
        if create_prompt is None:
            prompt = self.create_prompt(question, context)
        else:
            prompt = create_prompt(question, context)
        outputs = self.generation_pipe(
                prompt,
                do_sample = False,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
            )
        result = outputs[0]["generated_text"][len(prompt):]
        resContent = list(result)[-1]["content"]
        if self.COMP_ID == "":
            answer = resContent
        elif self.COMP_ID+":" in resContent:
            answer = resContent.split(self.COMP_ID+":")[1].strip() 
        else:
            answer = resContent  
     
       # print("output: ", answer)
        return answer


    def return_prompt_and_responses(self, samples):
        if "model_max_length" in self.config.keys():
            input_encodings = self.tokenizer(samples["full_prompt"], truncation=True, padding='max_length', max_length=self.config["model_max_length"]) 
            model_prompts = self.tokenizer(samples["prompt"], truncation=True, padding='max_length', max_length=self.config["model_max_length"])
        else:
            input_encodings = self.tokenizer(samples["full_prompt"]) 
            model_prompts = self.tokenizer(samples["prompt"])

        label = copy.deepcopy(input_encodings["input_ids"])
        prompt_length = len(model_prompts["input_ids"])
        label[1:prompt_length-1] = [IGNORE_INDEX] * (prompt_length-2)  # mask the query
        
        new_label = [(l if l != self.tokenizer.pad_token_id else IGNORE_INDEX) for l in label]
        input_encodings["labels"] = new_label
        
        return input_encodings 


    def train(self, train_in_path, train_out_path, eval_in_path=None, eval_out_path=None):

        train_list = self.load_data(train_in_path)
        train_list = self.prepare_sft_data(train_list)
        with open(train_out_path, "w") as f:
            json.dump(train_list, f)
        random.shuffle(train_list)
        train_dataset = Dataset.from_list(train_list)

        train_dataset = train_dataset.map(
            self.return_prompt_and_responses,
            batched=False,
        )
        train_dataset = train_dataset.remove_columns(['prompt', 'full_prompt'])


        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        dev_dataset = None
        if not eval_in_path is None:
            dev_list = self.load_data(eval_in_path)
            dev_list = self.prepare_sft_data(dev_list)
            with open(eval_out_path, "w") as f:
                json.dump(dev_list, f)
            random.shuffle(dev_list)
            dev_dataset = Dataset.from_list(dev_list)

            dev_dataset = dev_dataset.map(
                self.return_prompt_and_responses,
                batched=False,
            )
            dev_dataset = dev_dataset.remove_columns(['prompt', 'full_prompt'])


        logger.info(f"Cuda available: {torch.cuda.is_available()}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
       
        if dev_dataset is None:
            evalDuringTraining = "no"
        else:
            evalDuringTraining = "epoch"
        
        training_args = TrainingArguments(
            output_dir=self.config["model_save_path"],
            warmup_ratio=self.config["model_warmup_ratio"],
            learning_rate=self.config["model_learningrate"],
            num_train_epochs=self.config["model_num_epochs"],
            per_device_train_batch_size=self.config["model_batch_size"],
            per_device_eval_batch_size=self.config["model_eval_batch_size"],
            remove_unused_columns=False,  # prevents from indexing errors
            save_strategy="epoch",  # epoch-wise eval
            evaluation_strategy=evalDuringTraining,  # epoch-wise eval
            save_only_model=True,  # do not store optimizer state etc. to save space
            save_total_limit=1,  # only store best model
            report_to="none",  # avoid issues with distributed setup
            bf16=torch.cuda.is_bf16_supported(),  # mixed-precision training
           # do_eval=evalDuringTraining,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
          
        )

        # trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        logger.info("Starting training now...")
        trainer.train()
        logger.info("Done with training!")


    def inference(self, input_path, output_path):
        count = 0
        # open data 
        data = list()
        with open(input_path, "r") as fp:
            data = json.load(fp)

            for i in range(len(data)):
                for j in range(len(data[i]["questions"])):
                    count += 1
                    history = ""
                    for k in range(j):
                        ansString = ""
                        for ans in data[i]["questions"][k]["answers"]:
                            ansString += ans["label"] + ";"
                        history += data[i]["questions"][k]["question"] + " " + ansString + " "
                    history = history.strip()
                    question = data[i]["questions"][j]["question"]
                    if "sampling" in self.config.keys():
                        data[i]["questions"][j]["generated_answer"] = self.generate_sampled_output(question,history)
                        gen_ans = ""
                        if len( data[i]["questions"][j]["generated_answer"]) > 0:
                            gen_ans =  data[i]["questions"][j]["generated_answer"][0]
                        self.hit_at_5 += get_hit_at_5(data[i]["questions"][j]["generated_answer"], data[i]["questions"][j]["answers"])
                        self.mrr += get_mrr(data[i]["questions"][j]["generated_answer"], data[i]["questions"][j]["answers"])
                    else:
                        gen_ans = self.generate_output(question,history)
                        data[i]["questions"][j]["generated_answer"] = gen_ans 
                    if is_answer_correct(gen_ans, data[i]["questions"][j]["answers"]):
                        self.p_at_1 += 1
                  
        with open(output_path, "w") as fout:
            json.dump(data, fout)

        self.final_metrics(count)
        

    def prepare_sft_data(self, data):
        inputs = []
            
        for i in range(len(data)):
            for j in range(len(data[i]["questions"])):
                current_data = data[i]["questions"][j]
                history = ""
                for k in range(j):
                    ansString = ""
                    for ans in data[i]["questions"][k]["answers"]:
                        ansString += ans["label"] + ";"
                    history += data[i]["questions"][k]["question"] + " " + ansString + " "
                history = history.strip()
                question = current_data["question"]
                answer =  current_data["answers"][0]["label"]
               
                full_prompt = self.create_prompt(question, history, answer)
                full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)
               
                prompt = self.create_prompt(question, history, None)
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                entry = dict()
                entry["prompt"] = prompt
                entry["full_prompt"] = full_prompt
                inputs.append(entry)
                    
        return inputs


    def create_prompt(self, question, history, label=None):
        if "fewshot" in self.config.keys():
            dialog = [
                {
                    "role": "system",
                    "content": self.INSTRUCTION_FEWSHOT
                },
            ]

        else:
            dialog = [
            {
                "role": "system",
                "content": self.INSTRUCTION
            },
        ]
        if "fewshot" in self.config.keys():
            examples = self.ICL_EXAMPLES
            for ex in examples:
                dialog.append({
                    "role": "user",
                    "content": self.PROMPT_TEMPLATE.format(history=ex["history"], question=ex["question"]),
                })
                dialog.append({
                    "role": "assistant",
                    "content": "answer: " + ex["answer"],
                })
       
        dialog.append({
            "role": "user",
            "content": self.PROMPT_TEMPLATE.format(history=history, question=question),
        })
        if not label is None:
            dialog.append({
                "role": "assistant",
                "content": "answer: " + label,
        })
        return dialog


    def load_data(self, path):
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


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    train = config["train"]
    ansgen = E2EModel(config,train)
    if train:
        if config["train_type"] == "SFT":
            ansgen.train(config["input_train_path"])
        elif config["train_type"] == "DPO":
            ansgen.train_dpo()
        logger.info("Starting eval now...")
        evidence_data = None
        ansgen.inference(config["evidence_inference_path"], config["output_inference_path"], evidence_data)
        logger.info("Done with eval!")
    else:
        ansgen.inference(config["input_inference_path"], config["output_inference_path"])
