import copy
import json
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline, DataCollatorWithPadding
from trl import  DPOTrainer, DPOConfig
from datasets import Dataset
from loguru import logger
sys.path.append(".")
sys.path.append("..")
from peft import LoraConfig, get_peft_model, PeftModel
import random
import os
import numpy as np


random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

IGNORE_INDEX = -100

################################
# base class for PRAISE modules
################################
class PraiseBaseComponent():
    def __init__(self, config, comp_id, instruction, icl_examples):
        self.config = config
        self.COMP_ID = comp_id
        
        #default template
        self.CHAT_PROMPT_TEMPLATE = (
            "question: {question}\n"
            "context: {context}\n" 
       )
        self.INSTRUCTION = instruction
        self.ICL_EXAMPLES = icl_examples

        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], cache_dir="checkpoints", token=config["hf_token"])


    def set_model(self, model):
        self.model = model


    def set_generation_pipeline(self, model):
        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=75,
        )  


    def load_model(self, config):
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
        elif "bloat16" in self.config.keys():
            self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="checkpoints", token=config["hf_token"], device_map="auto", torch_dtype=torch.bfloat16)
        
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="checkpoints", token=config["hf_token"], device_map="auto") 
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        #continue training from a previous peft model 
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
            elif "bfloat16" in self.config.keys():
                self.model = PeftModel.from_pretrained(
                    self.model,
                    config["peft_model_path"],
                    is_trainable=True,
                    adapter_name="trainedSFT",
                    torch_dtype=torch.bfloat16
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
        
        elif config["peft"]:
            if config["train"] == "train":
                self.model = get_peft_model(self.model, self.peft_config)
                print(self.model.print_trainable_parameters())
                #print(self.model)
            else:
                #merge model for faster inference
                self.model = PeftModel.from_pretrained(self.model, config["peft_model_path"], adapter_name=config["adapter_name"])
                #self.model = self.model.merge_and_unload()    

        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=75,
        )  

        return self.model


    def prepare_sft_data(self, data):
        pass


    def prepare_pref_data(self, data, ev_data=None):
        pass


    def calculate_metrics(self, input1, input2):
        pass


    def final_metrics(self, count):
        pass


    def load_data(self, path):
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
        else:
            evidence_data = []
            for i in range(self.config["sample_size"]):
                if path.endswith(".jsonl"):
                    newpath = path.replace(".jsonl",  str(i) + ".jsonl")
                    with open(newpath, "r") as efp: 
                        line = efp.readline()
                        while line:
                            conversation = json.loads(line)
                            evidence_data.append(conversation)
                            line = efp.readline()
                else:
                    newpath = path.replace(".json",  str(i) + ".json")
                    with open(newpath, "r") as efp:
                        evidence_data.extend(json.load(efp))
        return evidence_data
    

    def add_assistant_ouput(self,answer):
        return [{
                "role": "assistant",
                "content": self.COMP_ID + ": " + answer,
            }]


    def create_prompt(self, question, context, label=None):  
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
                    "content": self.CHAT_PROMPT_TEMPLATE.format(question=ex["question"], context=ex["context"]),
                })
                dialog.append({
                    "role": "assistant",
                    "content": self.COMP_ID + ": " + ex["answer"],
                })

        dialog.append({
            "role": "user",
            "content": self.CHAT_PROMPT_TEMPLATE.format(question=question, context=context),
        })

        if not label is None:
            if self.COMP_ID != "":
                labelString = self.COMP_ID + ": " + label 
            else: 
                labelString = label
            dialog.append({
                "role": "assistant",
                "content": labelString
            })
        return dialog


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
        #collect sampled output
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

    #SFT trainer
    def train(self, train_in_path, train_out_path, eval_in_path=None, eval_out_path=None):

        train_list = self.load_data(train_in_path)
        if not os.path.exists(train_out_path):
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
            if not os.path.exists(eval_out_path):
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

    #DPO trainer
    def train_dpo(self, train_in_path, train_out_path, eval_in_path=None, eval_out_path=None):
        logger.info(f"Cuda available: {torch.cuda.is_available()}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        # create dataset
        train_data = self.load_data(train_in_path)
        if not os.path.exists(train_out_path):
            train_list = self.prepare_pref_data(train_data)
            with open(train_out_path, "w") as f:
                json.dump(train_list, f)
        train_dataset = Dataset.from_list(train_list)
        del train_list
       
        dev_dataset = None
        if not eval_in_path is None:
            dev_list = self.load_data(eval_in_path)
            if not os.path.exists(eval_out_path):
                dev_list = self.prepare_pref_data(dev_list)
                with open(eval_out_path, "w") as f:
                    json.dump(dev_list, f)
            dev_dataset = Dataset.from_list(dev_list)
            del dev_list

        evalDuringTraining = True
        if dev_dataset is None:
            evalDuringTraining = "no"
        else:
            evalDuringTraining = "epoch"

        training_args = DPOConfig(
            output_dir=self.config["model_save_path"],
            learning_rate=self.config["model_learningrate"],#1e-6,
            warmup_steps=self.config["model_warmup_steps"],#150,  
            per_device_train_batch_size=self.config["model_batch_size"], #2
            per_device_eval_batch_size=self.config["model_eval_batch_size"],
            num_train_epochs=self.config["model_num_epochs"],
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            evaluation_strategy=evalDuringTraining,
            save_strategy="epoch",
            save_only_model=True,
            bf16=torch.cuda.is_bf16_supported(),
            beta=0.1
        )

        dpo_trainer = DPOTrainer(
            self.model,                  # base model from SFT pipeline
            ref_model=None,              # typically a copy of the SFT trained base model
           # beta=0.1,#0.5,              # temperature hyperparameter of DPO
            train_dataset=train_dataset, # dataset prepared above
            eval_dataset=None,           # eval dataset prepared above
            tokenizer=self.tokenizer,    # tokenizer
            args=training_args,          # training arguments e.g. batch size, lr, etc.
        )
     
        logger.info("Starting training now...")
        dpo_trainer.train()
        logger.info("Done with training!")


    