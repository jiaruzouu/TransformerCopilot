import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from pilot import LlamaForCausalLM
from copilot.modeling_llama_explore import New_LlamaForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    if args.logfile is not None:
        sys.stdout = open(args.logfile, "a", buffering=1)
        sys.stderr = sys.stdout
    
    print(args)
    print(args.explore_weight)

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        if args.explore_flag == 1:
            if args.decoding_method == 'greedy':
                generation_explore_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generation_mode="greedy",  # Set mode to greedy
                    num_beams=1,               # Ensure no beam search
                    do_sample=False,            # Disable sampling
                    **kwargs
                )
            elif args.decoding_method == 'beam_search':
                generation_explore_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    **kwargs,
                )
        
        with torch.no_grad():
            generation_output = model.generate(
                model_explore=model_explore if args.explore_flag == 1 else None,
                generation_explore_config=generation_explore_config if args.explore_flag == 1 else None,
                explore_weight=args.explore_weight if args.explore_flag == 1 else 0.0,
                top_k_explore_logits=args.top_k_explore_logits if args.explore_flag == 1 else -1,
                max_new_tokens_generation=max_new_tokens,
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        print(outputs)
        return outputs

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/meta-llama')
    
    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    
    if args.explore_flag == 1:
        tokenizer, model, tokenizer_explore, model_explore = load_model(args)
    else: tokenizer, model = load_model(args)
    
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print("MODEL_OUTPUT:", output)
            print('prediction:', predict)
            print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', choices=['lora', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--logfile', required=False)
    
    # New added argument for the copilot model
    parser.add_argument('--explore_model', required=False)
    parser.add_argument('--base_explore_model', required=False)
    parser.add_argument('--explore_lora_weights', required=False)
    parser.add_argument('--explore_flag', type=int, default=0)
    parser.add_argument('--explore_weight', type=float, default=0.0)
    parser.add_argument('--decoding_method', type=str, choices=['greedy', 'beam_search'],required=False)
    parser.add_argument('--explore_logits_factor', type=float, default=100.0)
    parser.add_argument('--top_k_explore_logits', type=int, default=-2)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if args.explore_flag == 1:
        base_explore_model = args.base_explore_model
    
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    
    if args.explore_flag == 1:
        if not base_explore_model:
            raise ValueError(f'can not find base model name by the value: {args.model}')
        
    lora_weights = args.lora_weights
    
    if args.explore_flag == 1:
        explore_lora_weights = args.explore_lora_weights
        if not explore_lora_weights:
            raise ValueError(f'can not find lora weight, the value is: {explore_lora_weights}')
    
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token_id = 0  # unk
        tokenizer.padding_side = "left"
        tokenizer.pad_token = "[PAD]"
        if args.explore_flag == 1:
            tokenizer_explore = AutoTokenizer.from_pretrained(base_explore_model)
            tokenizer_explore.pad_token_id = 0
            tokenizer_explore.padding_side = "left"
            tokenizer_explore.pad_token = "[PAD]"
            
    assert device == "cuda"
    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) 
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
        if args.explore_flag == 1:            
            model_hidden_size = model.config.hidden_size
            model_explore = New_LlamaForCausalLM.from_pretrained(
                base_explore_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                exploit_hidden_size=model_hidden_size,
                explore_logits_factor=args.explore_logits_factor,
            )

            model_explore = PeftModel.from_pretrained(
                model_explore,
                explore_lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0}
            )
            
        model.config.output_hidden_states = True
        
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    if args.explore_flag == 1:
        return tokenizer, model, tokenizer_explore, model_explore
    else: return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    sentence = sentence.lower()
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
