#!/usr/bin/env python
import yaml
from loguru import logger
import re
import sys
import json
sys.path.append("../EXPLAIGNN")
sys.path.append("../EXPLAIGNN/CLOCQ")


def retrieve_evidence_clocq_bm25(clocq, evidence_input_path, evidence_output_path, train=True, idx=""):
    clocq.inference_on_data_split(evidence_input_path, evidence_output_path, sources="kb_text_table_info", train=train,idx=idx)
   

def retrieve_evidence_per_question_clocq_bm25(clocq, rewritten_question):
    evidences, _ = clocq.evr.retrieve_evidences(rewritten_question, "kb_text_table_info")
    top_evidence = clocq.evs.get_top_evidences(rewritten_question, evidences)
    return top_evidence


def hasNumber(string):
    for ch in string:
        if ch.isdigit():
            return True
    return False

# return if the given string is a timestamp
def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    try:
        if not(pattern.match(timestamp)):
            return False
        else:
            return True
    except Exception as e:
        print("error for timestamp check: ", timestamp)
        return False

def convertTimestamp( timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    timesplits = timestamp.split("-")
    year = timesplits[0]
    if yearPattern.match(timestamp):
        return None, None, year
    month = convertMonth(timesplits[1])
    if monthPattern.match(timestamp):
        return None, month, year
    elif dayPattern.match(timestamp):
        day = timesplits[2].rsplit("T")[0]
        return day, month, year
   
    return None


def check_correct_time(answer, goldday, goldmonth, goldyear):
    if goldday.startswith("0"):
        goldday = goldday.replace("0", "",1)
    if goldyear in answer:
        if goldmonth is None:
            return True
        else:
            if goldmonth in answer.lower():
                if goldday is None or goldday in answer:
                    return True
    
    return False


# convert the given month to a number
def convertMonth( month):
    return{
        "01": "january",
        "02": "february",
        "03": "march",
        "04": "april",
        "05": "may",
        "06": "june",
        "07": "july",
        "08": "august",
        "09": "september", 
        "10": "october",
        "11": "november",
        "12": "december"
    }[month]



def sort_evs_by_score(dict):
    return dict['score']


def trim_text_to_max_tokens(tokenizer, text, max_num_tokens=75):
    """Trims the given text to the given maximum number of tokens for the tokenizer."""
    tokenized_prediction = tokenizer.encode(text)
    trimmed_tokenized_prediction = tokenized_prediction[: max_num_tokens]#tokenized_prediction[1: max_num_tokens + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def larger_max_tokens(tokenizer,text, max_num_tokens=75):
    """Trims the given text to the given maximum number of tokens for the tokenizer."""
    tokenized_prediction = tokenizer.encode(text)
    if len(tokenized_prediction) > max_num_tokens:
        return True
    return False 


def load_config(config_path):
    """Load config from yaml path."""
    logger.info(f"Loading config from: `{config_path}`")
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


