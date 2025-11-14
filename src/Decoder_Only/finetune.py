import os
import sys
from typing import List
import json
import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
import pdb 
from tqdm import tqdm
import copy
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import copy

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    LlamaTokenizer,
)

from pilot import LlamaForCausalLM  # noqa: F402
from copilot.modeling_llama_explore import New_LlamaForCausalLM  # noqa: F402

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass



def train_runner(
    model,
    model_explore,
    tokenizer,
    tokenizer_explore,
    train_data,
    val_data,
    micro_batch_size,
    gradient_accumulation_steps,
    num_epochs,
    learning_rate,
    warmup_steps,
    fp16,
    eval_step,
    save_step,
    output_dir,
    explore_output_dir,
    device,
    val_set_size,
    use_wandb,
    wandb_run_name,
    ddp,
    group_by_length,
    resume_from_checkpoint,
    explore_flag,
):
    """
    A customized training loop that replaces the original transformers.Trainer usage.
    Includes a tqdm progress bar for training.
    """

    # ---------------------------
    # 1. Setup DataLoaders
    # ---------------------------
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    if val_data is not None:
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=micro_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    else:
        val_loader = None

    # ---------------------------
    # 2. Setup Optimizer / Scheduler
    # ---------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if explore_flag:
        optimizer_explore = torch.optim.AdamW(model_explore.parameters(), lr=learning_rate)
        total_steps_explore = len(train_loader) * num_epochs
        scheduler_explore = get_linear_schedule_with_warmup(
            optimizer_explore,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_explore,
        )
    
    # Optionally set up gradient scaling for FP16
    scaler = None
    scaler_explore = None
    if fp16:
        scaler = torch.cuda.amp.GradScaler()

        if explore_flag:
            scaler_explore = torch.cuda.amp.GradScaler()


    # ---------------------------
    # 3. (Optional) Load from checkpoint if needed
    # ---------------------------
    if resume_from_checkpoint:
        print(f"[train_runner] Resuming from checkpoint: {resume_from_checkpoint}")
        # Potentially load the optimizer, scheduler state, etc. as well.

    # ---------------------------
    # 4. Training Loop
    # ---------------------------
    global_step = 0
    model.train()
    if explore_flag:
        model_explore.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_explore_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}", unit="batch") as pbar:
            for step, batch in enumerate(train_loader):
                for k in batch:
                    batch[k] = batch[k].to(device)

                with torch.cuda.amp.autocast(fp16):
                    exploit_outputs = model(**batch)
                    if explore_flag:
                        
                        exploit_input_ids = batch['input_ids'].detach()
                        exploit_attention_mask = batch['attention_mask'].detach()
                        exploit_labels = batch['labels'].detach()
                        
                        num_layers = len(exploit_outputs.hidden_states) 
                        bsz, seq_len = exploit_labels.size()
                        exploit_hidden_states = torch.cat(exploit_outputs.hidden_states,dim=1).detach() 
                        exploit_hidden_states = exploit_hidden_states.view(bsz, num_layers, seq_len, -1)
                        exploit_hidden_states = exploit_hidden_states.mean(dim=1)
                        
                        exploit_logits = exploit_outputs.logits.detach()
                        
                        exploit_probabilities = F.softmax(exploit_logits, dim=-1)
                        explore_labels_prob = torch.zeros_like(exploit_probabilities, device=device, dtype=exploit_hidden_states.dtype)
                        
                        explore_labels_prob[
                            torch.arange(exploit_labels.size()[0], device=device).unsqueeze(1).expand(-1, exploit_labels.size()[1]), 
                            torch.arange(exploit_labels.size()[1], device=device).unsqueeze(0).expand(exploit_labels.size()[0], -1), 
                            exploit_labels
                        ] = torch.tensor(1, device=device,dtype=exploit_hidden_states.dtype)
                        
                        
                        explore_labels_prob = explore_labels_prob - exploit_probabilities
                        
                        explore_outputs = model_explore(
                            attention_mask=exploit_attention_mask,
                            labels=explore_labels_prob,
                            exploit_hidden_states=exploit_hidden_states,
                        )
                        
                        explore_loss = explore_outputs.loss
                        explore_loss = explore_loss / gradient_accumulation_steps
                        
                    loss = exploit_outputs.loss
                    loss = loss / gradient_accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                    if explore_flag:
                        scaler_explore.scale(explore_loss).backward()
                else:
                    loss.backward()
                    if explore_flag:
                        explore_loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps
                if explore_flag:
                    epoch_explore_loss += explore_loss.item() * gradient_accumulation_steps 

                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                        if explore_flag:
                            scaler_explore.step(optimizer_explore)
                            scaler_explore.update()
                    else:
                        optimizer.step()
                        if explore_flag:
                            optimizer_explore.step()

                    optimizer.zero_grad()
                    if explore_flag:
                        optimizer_explore.zero_grad()
                    scheduler.step()
                    if explore_flag:
                        scheduler_explore.step()
                    global_step += 1

                    # Logging every 10 steps or so (just an example)
                    if global_step % 10 == 0:
                        current_loss = loss.item() * gradient_accumulation_steps
                        print(f"  [train_runner] step={global_step}, loss={current_loss:.4f}")
                        if explore_flag:
                            current_explore_loss = explore_loss.item() * gradient_accumulation_steps
                            print(f"  [train_runner] step={global_step}, explore_loss={current_explore_loss:.4f}")

                    # Evaluate if we have a val set and at correct step
                    if val_loader and (eval_step > 0) and (global_step % eval_step == 0):
                        evaluate_loss = 0.0
                        evaluate_explore_loss = 0.0
                        model.eval()
                        if explore_flag:
                            model_explore.eval()
                            
                        with torch.no_grad():
                            for val_batch in val_loader:
                                for k in val_batch:
                                    val_batch[k] = val_batch[k].to(device)
                                with torch.cuda.amp.autocast(fp16):
                                    val_out = model(**val_batch)
                                    if explore_flag:
                                        val_exploit_input_ids = val_batch['input_ids']
                                        val_exploit_attention_mask = val_batch['attention_mask']
                                        val_exploit_labels = val_batch['labels']
                                        
                                        val_num_layers = len(val_out.hidden_states) 
                                        val_bsz, val_seq_len = val_exploit_labels.size()
                                        val_exploit_hidden_states = torch.cat(val_out.hidden_states,dim=1).detach() 
                                        val_exploit_hidden_states = val_exploit_hidden_states.view(val_bsz, val_num_layers, val_seq_len, -1)
                                        val_exploit_hidden_states = val_exploit_hidden_states.mean(dim=1)
                                        
                                        val_exploit_logits = val_out.logits.detach() 
                                        
                                        val_exploit_probabilities = F.softmax(val_exploit_logits, dim=-1)
                                        val_explore_labels_prob = torch.zeros_like(val_exploit_probabilities, device=device, dtype=val_exploit_hidden_states.dtype)
                                        
                                        val_explore_labels_prob[
                                            torch.arange(val_exploit_labels.size()[0], device=device).unsqueeze(1).expand(-1, val_exploit_labels.size()[1]),
                                            torch.arange(val_exploit_labels.size()[1], device=device).unsqueeze(0).expand(val_exploit_labels.size()[0], -1),
                                            val_exploit_labels
                                        ] = torch.tensor(1, device=device,dtype=val_exploit_hidden_states.dtype)
                                        
                                        val_explore_labels_prob = val_explore_labels_prob - val_exploit_probabilities
                                        
                                        val_explore_out = model_explore(
                                            attention_mask=val_exploit_attention_mask,
                                            labels=val_explore_labels_prob,
                                            exploit_hidden_states=val_exploit_hidden_states,
                                        )
                                        evaluate_explore_loss += val_explore_out.loss.item()
                                evaluate_loss += val_out.loss.item()
                        evaluate_loss /= len(val_loader)
                        print(f"  [train_runner] Eval step {global_step} | val_loss={evaluate_loss:.4f}")
                        if explore_flag:
                            evaluate_explore_loss /= len(val_loader)
                            print(f"  [train_runner] Eval step {global_step} | val_explore_loss={evaluate_explore_loss:.4f}")
                        model.train()
                        if explore_flag:
                            model_explore.train()

                    # Save checkpoint at specified steps
                    if save_step > 0 and (global_step % save_step == 0):
                        ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_path, exist_ok=True)
                        print(f"  [train_runner] Saving checkpoint to {ckpt_path}")
                        model.save_pretrained(ckpt_path)
                        
                        if explore_flag:
                            ckpt_path_explore = os.path.join(explore_output_dir, f"checkpoint-{global_step}-explore")
                            os.makedirs(ckpt_path_explore, exist_ok=True)
                            print(f"  [train_runner] Saving explore_checkpoint to {ckpt_path_explore}")
                            model_explore.save_pretrained(ckpt_path_explore)

                # Update our training tqdm
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "explore_loss": f"{explore_loss.item() * gradient_accumulation_steps:.4f}" if explore_flag else "",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "explore_lr": f"{scheduler_explore.get_last_lr()[0]:.2e}" if explore_flag else "",
                    }
                )

        avg_loss = epoch_loss / len(train_loader)
        
        if explore_flag:
            avg_explore_loss = epoch_explore_loss / len(train_loader)
            print(f"  [train_runner] Epoch {epoch+1} done | avg_train_loss={avg_loss:.4f} | avg_explore_loss={avg_explore_loss:.4f}")
        else:
            print(f"  [train_runner] Epoch {epoch+1} done | avg_train_loss={avg_loss:.4f}")

    print("[train_runner] Training complete.")

def train(
        base_model: str = "", 
        explore_base_model: str = "",
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        explore_output_dir: str = "./lora-explore-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        dataset_name: str = "",
        test_data_path: str="",
        direct_test: bool = False,
        explore_flag: bool = False,
        explore_logits_factor: float = 100.0,
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"explore_base_model: {explore_base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"explore_output_dir: {explore_output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"dataset_name: {dataset_name}\n"
        f"test_data_path: {test_data_path}\n"
        f"direct_test: {direct_test}\n"
        f"explore_flag: {explore_flag}\n"
        f"explore_logits_factor: {explore_logits_factor}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    if explore_flag:
        assert (
            explore_base_model
        ), "Please specify a --explore_model , e.g. --explore_model='decapoda-research/llama-7b-hf'"
        
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # ---------------------
    # Load Model
    # ---------------------
    if load_8bit:
        assert load_8bit ==False
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
            
        )
        model_hidden_size = model.config.hidden_size
        if explore_flag:
            model_explore = New_LlamaForCausalLM.from_pretrained(
                explore_base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
                exploit_hidden_size=model_hidden_size,
                explore_logits_factor=explore_logits_factor,
            )
        
        
    if model.config.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
        if explore_flag:
            tokenizer_explore = AutoTokenizer.from_pretrained(explore_base_model, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    
    if explore_flag:
        tokenizer_explore.pad_token_id = 0
        tokenizer_explore.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not train_on_inputs:
            assert train_on_inputs == True 
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # Prepare model for int8 training if needed
    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if explore_flag:
        model_explore = prepare_model_for_int8_training(model_explore, use_gradient_checkpointing=use_gradient_checkpointing)
        
    # ---------------------
    # PEFT Config
    # ---------------------
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"target_modules for base model: {target_modules}")
        
        if explore_flag:
            target_explore_modules = copy.deepcopy(target_modules)
            target_explore_modules.append("fc_layer")
            if model.config.hidden_size != model_explore.config.hidden_size:
                target_explore_modules.append("match_hidden_states")
            
            print(f"target_modules for explore model: {target_explore_modules}")
            
            config_explore = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_explore_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config)
    if explore_flag:
        model_explore = get_peft_model(model_explore, config_explore)
    
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    # ---------------------
    # Load dataset
    # ---------------------
    if data_path.endswith(".json"): 
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        assert resume_from_checkpoint==False 
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA / adapter weights
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()
    if explore_flag:
        model_explore.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=train_val["train"].column_names)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=train_val["test"].column_names)
        )
    else:
        train_data = data["train"].shuffle().map(
            generate_and_tokenize_prompt, remove_columns=data["train"].column_names
        )
        val_data = None

    

    # Mark model to not use cache during training
    model.config.use_cache = False
    model.config.output_hidden_states = True
    
    if explore_flag:
        model_explore.config.use_cache = False
        
    # Keep a special state_dict that only saves the adapter's parameters
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    
    if explore_flag:
        old_state_dict_explore = model_explore.state_dict
        model_explore.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict_explore())
        ).__get__(model_explore, type(model_explore))

    # ---------------------
    # Call our custom training loop
    # ---------------------
    # Decide on the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not ddp:
        model.to(device)
        if explore_flag:
            model_explore.to(device)

    # Equivalent warmup_steps as we had in the Trainer setup
    warmup_steps = 100

    train_runner(
        model=model,
        model_explore=model_explore if explore_flag else None,
        tokenizer=tokenizer,
        tokenizer_explore=tokenizer_explore if explore_flag else None,
        train_data=train_data,
        val_data=val_data,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        fp16=True,  # from the original TrainingArguments(fp16=True)
        eval_step=eval_step if val_set_size > 0 else 0,
        save_step=save_step,
        output_dir=output_dir,
        explore_output_dir=explore_output_dir,
        device=device,
        val_set_size=val_set_size,
        use_wandb=use_wandb,
        wandb_run_name=wandb_run_name,
        ddp=ddp,
        group_by_length=group_by_length,
        resume_from_checkpoint=resume_from_checkpoint,
        explore_flag=explore_flag,
    )

    # ---------------------
    # After training, save final model
    # ---------------------
    model.save_pretrained(output_dir)
    if explore_flag:
        model_explore.save_pretrained(explore_output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)\n")

    # ---------------------
    # (Optional) Direct Test
    # ---------------------
    if direct_test:
        assert direct_test == False
        # The rest of the code remains the same as before...
        model.eval()
        def evaluate(
                instructions,
                input=None,
                temperature=0.01,
                top_p=0.95,
                top_k=40,
                num_beams=4,
                max_new_tokens=16,
                **kwargs,
        ):
            prompts = [generate_prompt_for_test(instr, input) for instr in instructions]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            s = generation_output.sequences
            outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
            outputs = [o.split("### Response:")[1].strip() for o in outputs]
            print(outputs)
            return outputs        

        save_file = f'experiment/{base_model}-{adapter_name}-{dataset_name}.json'
        create_dir('experiment/')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        dataset = load_data(test_data_path)
        batches = create_batch(dataset, batch_size)
        total = len(batches)
        correct = 0
        current = 0
        output_data = []
        pbar = tqdm(total=total)
        for idx, batch in enumerate(batches):
            current += len(batch)
            instructions = [d.get('instruction') for d in batch]

            outputs = evaluate(instructions)

            for data_, output in zip(batch, outputs):
                label = data_.get('answer')
                flag = False
                predict = extract_answer(dataset_name, output)
                if label == predict:
                    correct += 1
                    flag = True
                new_data = copy.deepcopy(data_)
                new_data['output_pred'] = output
                new_data['pred'] = predict
                new_data['flag'] = flag
                output_data.append(new_data)
                print(data_["instruction"])
                print("MODEL OUTPUT:", output)
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

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501

def load_data(file_path) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
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


def extract_answer(dataset, sentence: str) -> float:
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

def generate_prompt_for_test(instruction, input=None):
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

if __name__ == "__main__":
    fire.Fire(train)
