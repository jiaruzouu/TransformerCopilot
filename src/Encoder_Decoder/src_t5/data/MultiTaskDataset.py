import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.prompt import load_prompt_template, get_info_from_prompt,check_task_prompt
from utils import utils
from utils import indexing
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json

        
class MultiTaskDataset(Dataset):    
    def parse_dataset_args(parser):
        """
        parse dataset related command line arguments
        """
        parser.add_argument("--data_path", type=str, default='../data', help="data directory")
        parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method, including random, sequential and collaborative")
        parser.add_argument("--tasks", type=str, default='sequential,direct,straightforward', help="Downstream tasks, separate by comma")
        parser.add_argument("--datasets", type=str, default='Beauty', help="Dataset names, separate by comma")
        parser.add_argument("--prompt_file", type=str, default='../prompt_template.txt', help='the path of the prompt template file')
        
        
        # arguments related to item indexing methods
        parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during ')
        parser.add_argument("--collaborative_token_size", type=int, default=200, help='the number of tokens used for indexing')
        parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
        parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
        parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')
        
        # arguments related to sequential task
        parser.add_argument("--max_his", type=int, default=-1, help='the max number of items in history sequence, -1 means no limit')
        parser.add_argument("--his_prefix", type=int, default=1, help='whether add prefix in history')
        parser.add_argument("--his_sep", type=str, default=' , ', help='The separator used for history')
        parser.add_argument("--skip_empty_his", type=int, default=1, help='whether include data with empty history.')
        
        # arguments related for evaluation
        parser.add_argument("--valid_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
        parser.add_argument("--valid_prompt_sample", type=int, default=1, help='use sampled prompt for validation every epoch.')
        parser.add_argument("--valid_sample_num", type=str, default='3,3', help='the number of sampled data for each task')
        parser.add_argument("--test_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
        
        # arguments related to prompt sampling
        parser.add_argument("--sample_prompt", type=int, default=0, help='sample prompt or not')
        parser.add_argument("--sample_num", type=str, default='2,2,2', help='the number of sampled data for each task')
        
        return parser

    def __init__(self, args, dataset_name, mode):
        self.mode = mode
        self.data = []

        self.args = args
        # Load JSON data
        with open(dataset_name, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mode == "train":
            if self.args.datasets in ("mawps", "svamp", "gsm8k", "aqua", "boolq", "piqa", "hellaswag", "social_i_qa", "winogrande", "openbookqa"):
                if item["input"]:
                    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                        ### Instruction:
                        {item["instruction"]}

                        ### Input:
                        {item["input"]}

                        ### Response:"""
                else:
                    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                        ### Instruction:
                        {item["instruction"]}

                        ### Response:"""
                
                output_text = item["output"]

                return {
                    "input": prompt, 
                    "output": output_text
                }
            elif self.args.datasets in ("NA"):
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    
                    ### Input:
                    {item["sQuestion"]} Answer Choices: {item["options"]}

                    ### Response:"""         
                output_text = item["lSolutions"]
                answer = item["lSolutions"]
                


                return {
                    "input": prompt, 
                    "output": output_text,
                    "answer": answer
                }
                
        if self.mode == "test":
            if self.args.datasets in ("mawps", "svamp", "gsm8k", "aqua", "boolq", "piqa", "hellaswag", "social_i_qa", "winogrande", "openbookqa"):
                if item["input"]:
                    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                        ### Instruction:
                        {item["instruction"]}

                        ### Input:
                        {item["input"]}

                        ### Response:"""
                else:
                    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                        ### Instruction:
                        {item["instruction"]}

                        ### Response:"""
                
                output_text = item["answer"]

                return {
                    "input": prompt, 
                    "output": output_text
                }
                
            elif self.args.datasets in("NA"):
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Input:
                    {item["instruction"]}

                    ### Response:"""         
                output_text = item["answer"]
                answer = item["answer"]

                return {
                    "input": prompt, 
                    "output": output_text,
                    "answer": answer
                }
            