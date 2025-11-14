import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
from utils import utils
import utils.generation_trie as gt
import utils.evaluate as evaluate
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from processor.Collator import Collator, TestCollator
import time
import pdb
import random 
import wandb 
from utils.generation import Explore_GenerationMixin
import re

class SingleRunner:
    def parse_runner_args(parser):
        """
        parse dataset related command line arguments
        """
        parser.add_argument("--optim", type=str, default='AdamW', help='The name of the optimizer')
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--clip", type=float, default=1)
        parser.add_argument("--logging_step", type=int, default=100)
        parser.add_argument("--warmup_prop", type=float, default=0.05)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=2)
        parser.add_argument("--train", type=int, default=1, help='train or not')
        parser.add_argument("--backbone", type=str, default='t5-small', help='backbone model name')
        parser.add_argument("--metrics", type=str, default='hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics used for evaluation')
        parser.add_argument("--load", type=int, default=0, help='load model from model path or not.')
        parser.add_argument("--random_initialize", type=int, default=1, help='Randomly initialize number-related tokens.')
        parser.add_argument("--test_epoch", type=int, default=1, help='test once for how many epochs, 0 for no test during training.')
        parser.add_argument("--valid_select", type=int, default=0, help='use validation loss to select models')
        parser.add_argument("--test_before_train", type=int, default=1, help='whether test before training')
        parser.add_argument("--test_filtered", type=int, default=0, help='whether filter out the items in the training data.')
        parser.add_argument("--test_filtered_batch", type=int, default=1, help='whether testing with filtered data in batch.')
        
        return parser
    
    def __init__(self, model, model_explore, tokenizer, train_loader, valid_loader, test_loader, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.args = args
        self.regenerate_candidate = False
        self.reconstruct_data = False
        self.test_epoch = self.args.test_epoch
        self.valid_select = self.args.valid_select
        self.test_before_train = self.args.test_before_train
        self.test_filtered = self.args.test_filtered
        self.test_filtered_batch = self.args.test_filtered_batch
        
        # Set Copilot configurations here
        self.explore_weight = 0
        self.explore = False 
        self.f2_shuffle = False  
        self.f2_loss = 'rmse' 
        self.wandb = False 
        self.epoch_idx = 0
        if self.explore:
            self.model_explore = model_explore
        
        self.testloaders = []
        self.testloaders.append(test_loader)
        
        if args.train:
            self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()
            if self.explore:
                self.explore_optimizer, self.explore_scheduler = self.explore_create_optimizer_and_scheduler()
            
        self.metrics = args.metrics.split(',')
        self.generate_num = max([int(m.split('@')[1]) for m in self.metrics])
        
        if self.explore and self.f2_shuffle:
            self.hidden_states_lists = []
            self.attention_mask_lists = []
            self.decoder_hidden_states_lists = []
            self.labels_lists = []
            self.output_attention_lists = []
            self.lm_logits_lists = []
    
    def clear_explore_store_lists(self):
        if self.explore and self.f2_shuffle:
            self.hidden_states_lists.clear()
            self.attention_mask_lists.clear()
            self.decoder_hidden_states_lists.clear()
            self.labels_lists.clear()
            self.output_attention_lists.clear()
            self.lm_logits_lists.clear()
        return     
    def train(self):
        if self.wandb:
            wandb.init(project="explore_transformers", name="debug_test_2", config=self.args)
        self.model.zero_grad()
        if self.explore:
            self.model_explore.zero_grad()
        train_losses = []
        explore_train_losses = []
        valid_losses = []
        best_epoch = -1
        
        if self.test_before_train > 0:
            self.test()
        for epoch in range(self.args.epochs):
            self.epoch_idx += 1
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()
            logging.info(f"Start training for epoch {epoch+1}")
            self.model.train()
            if self.explore:
                self.model_explore.train()
            losses = []
            explore_losses = []
            i = -1
            self.clear_explore_store_lists()
            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                output_ids = batch[2].to(self.device)
                output_attention = batch[3].to(self.device)
                   
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=self.args.alpha,
                    return_dict=True,
                )
                
                loss = output["loss"]
                lm_logits = output["logits"].detach()
                encoder_hidden_states = output["encoder_hidden_states"].detach()
                encoder_attention_mask = output["encoder_attention_mask"].detach()
                sequence_output = output["sequence_output"]
                decoder_hidden_states = torch.cat(output["decoder_hidden_states"], dim=1).detach()
                
                lm_mask = output_attention != 0
                lm_mask = lm_mask.float()
                B, L = output_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()
                

                if self.explore:
                    
                    if self.f2_shuffle == True:

                        encoder_hidden_states = encoder_hidden_states.cpu()
                        encoder_attention_mask = encoder_attention_mask.cpu()

                        decoder_hidden_states = decoder_hidden_states.cpu()
                        output_ids = output_ids.cpu()
                        output_attention = output_attention.cpu()
                        lm_logits = lm_logits.cpu()
                        self.hidden_states_lists.append(encoder_hidden_states) 
                        self.attention_mask_lists.append(encoder_attention_mask)

                        self.decoder_hidden_states_lists.append(decoder_hidden_states) 
                        self.labels_lists.append(output_ids) 
                        self.output_attention_lists.append(output_attention) 
                        self.lm_logits_lists.append(lm_logits) 

                        list_length = len(self.hidden_states_lists)
                        random_idx = -1 
                        random_idx = random.randrange(list_length)
                        
                        encoder_hidden_states_ = self.hidden_states_lists[random_idx]
                        encoder_attention_mask_ = self.attention_mask_lists[random_idx]
                        decoder_hidden_states_ = self.decoder_hidden_states_lists[random_idx]

                        output_ids_ = self.labels_lists[random_idx]
                        output_attention_ = self.output_attention_lists[random_idx]
                        lm_logits_ = self.lm_logits_lists[random_idx]
                        
                        encoder_hidden_states_ = encoder_hidden_states_.to(self.device)
                        encoder_attention_mask_ = encoder_attention_mask_.to(self.device)
                        decoder_hidden_states_ = decoder_hidden_states_.to(self.device)
                        # last_layer_grad_ = last_layer_grad_.to(self.device)
                        output_ids_ = output_ids_.to(self.device)
                        output_attention_ = output_attention_.to(self.device)
                        lm_logits_ = lm_logits_.to(self.device)
                        
                        explore_output = self.model_explore(
                            backbone = self.args.backbone,
                            lm_logits=lm_logits_,
                            decoder_hidden_states=decoder_hidden_states_,
                            encoder_attention_mask=encoder_attention_mask_,
                            hidden_states=encoder_hidden_states_,
                            labels=output_ids_,  
                            output_attention=output_attention_,                  
                            return_dict=True,
                            loss_type=self.f2_loss
                        )

                    elif self.f2_shuffle == False:
                        explore_output = self.model_explore(
                            backbone = self.args.backbone,
                            lm_logits=lm_logits,
                            decoder_hidden_states=decoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            hidden_states=encoder_hidden_states,
                            labels=output_ids, 
                            output_attention=output_attention,                  
                            return_dict=True,
                            loss_type=self.f2_loss
                        )
                    
                    if self.f2_loss == 'ce':
                        if self.f2_shuffle :    
                            explore_lm_mask = output_attention_ != 0
                            explore_B, explore_L = output_ids_.size()
                        else: 
                            explore_lm_mask = output_attention != 0
                            explore_B, explore_L = output_ids.size()
                        explore_lm_mask = explore_lm_mask.float()
                        explore_loss = explore_output["explore_loss"]
                        explore_lm_logits = explore_output["logits"]
                        explore_loss = explore_loss.view(explore_B,explore_L) * explore_lm_mask
                        explore_loss = (explore_loss.sum(dim=1) / explore_lm_mask.sum(dim=1).clamp(min=1)).mean()
                    elif self.f2_loss == 'mse':
                        explore_loss = explore_output["explore_loss"]
                        explore_lm_logits = explore_output["logits"]
                    elif self.f2_loss == 'rmse':
                        explore_loss = explore_output["explore_loss"]
                    
                    
                loss.backward()
                if self.explore: 
                    explore_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                if self.explore:
                    torch.nn.utils.clip_grad_norm_(self.model_explore.parameters(), self.args.clip)
                self.optimizer.step()
                self.scheduler.step()
                if self.explore:
                    self.explore_optimizer.step()
                    self.explore_scheduler.step()
                
                self.model.zero_grad()
                if self.explore:
                    self.model_explore.zero_grad()
                losses.append(loss)
                if self.wandb:
                    wandb.log({"loss": loss.item()})
                if self.explore:
                    explore_losses.append(explore_loss) 
                    if self.wandb:
                        wandb.log({"explore_loss": explore_loss.item()})
                
                i += 1
                if i % 100 == 0:
                    train_batch_loss = sum(losses)/len(losses)
                    train_batch_explore_loss = sum(explore_losses)/len(explore_losses) if self.explore else 0 
                    print(f"Training loss for batch {i}: {train_batch_loss}; Training explore loss for batch {i}: {train_batch_explore_loss}")
                    if self.wandb:
                        wandb.log({"train_batch_loss": train_batch_loss})
                        if self.explore: wandb.log({"train_batch_explore_loss": train_batch_explore_loss})
            train_epoch_loss = sum(losses)/len(losses)
            train_losses.append(train_epoch_loss)
            
            if self.explore:
                train_epoch_explore_loss = sum(explore_losses)/len(explore_losses)
                explore_train_losses.append(train_epoch_explore_loss)
                logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}; The average training explore loss for epoch {epoch+1} is {train_epoch_explore_loss}")
            else:
                logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")
            
            self.test()
            
            if self.valid_select > 0:
                logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=self.args.alpha,
                            return_dict=True,
                        )
                        # compute loss masking padded tokens
                        loss = output["loss"]
                        lm_mask = output_attention != 0
                        lm_mask = lm_mask.float()
                        B, L = output_ids.size()
                        loss = loss.view(B, L) * lm_mask
                        loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                        losses.append(loss)
                    valid_epoch_loss = sum(losses)/len(losses)
                    valid_losses.append(valid_epoch_loss)
                    logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                    if valid_epoch_loss == min(valid_losses):
                        logging.info(f"The minimal validation loss so far.")
                        best_epoch = epoch + 1
                        utils.save_model(self.model, self.args.model_path)
                        logging.info(f"Save the current model to {self.args.model_path}")
                
            if self.test_epoch > 0:
                if (epoch + 1) % self.test_epoch == 0:
                    self.model.eval()
                    self.test()
            
            # dist.barrier()
        if self.valid_select > 0:
            if self.args.rank == 0:
                logging.info(f"The best validation at Epoch {best_epoch}")
        else:
            if self.args.rank == 0:
                torch.save(self.model.module.state_dict(), self.args.model_path)
                logging.info(f"Save the current model to {self.args.model_path}")
                

        return
    
    def create_optimizer_and_scheduler(self):
        if self.args.rank == 0:
            logging.info("Building Optimizer and Scheduler")
        batch_per_epoch = len(self.train_loader)
        total_steps = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_prop)
        
        if self.args.rank == 0:
            logging.info(f'Batch per epoch: {batch_per_epoch}')
            logging.info(f'Total steps: {total_steps}')
            logging.info(f'Warmup proportion: {self.args.warmup_prop}')
            logging.info(f'Warm up steps: {warmup_steps}')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.rank == 0:
            logging.info(f"Building Optimizer {self.args.optim}")
        
        if self.args.optim.lower() == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps)
        else:
            raise NotImplementedError
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return optimizer, scheduler
    
    def explore_create_optimizer_and_scheduler(self):
        if self.args.rank == 0:
            logging.info("Building Explore Model Optimizer and Scheduler")
        batch_per_epoch = len(self.train_loader)
        total_steps = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_prop)
        
        if self.args.rank == 0:
            logging.info(f'Explore: Batch per epoch: {batch_per_epoch}')
            logging.info(f'Explore:Total steps: {total_steps}')
            logging.info(f'Explore:Warmup proportion: {self.args.warmup_prop}')
            logging.info(f'Explore:Warm up steps: {warmup_steps}')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model_explore.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model_explore.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.rank == 0:
            logging.info(f"Building Explore Model Optimizer {self.args.optim}")
        
        if self.args.optim.lower() == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps)
        else:
            raise NotImplementedError
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return optimizer, scheduler
    
    def test(self, path=None, explore_path=None):
        self.model.eval()
        if self.explore:
            self.model_explore.eval()
        if path:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            if self.explore:
                self.model_explore.load_state_dict(torch.load(explore_path, map_location=self.device))
        for loader in self.testloaders:
            if self.test_filtered > 0:
                if self.test_filtered_batch > 0:
                    self.test_dataset_task_filtered_batch(loader)
                else:
                    assert self.args.eval_batch_size == 1
                    self.test_dataset_task_filtered(loader)
            else:
                self.test_dataset_task(loader)
            
            
    def test_dataset_task_filtered_batch(self, testloader):
        logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = batch[5].to(self.device)
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num + testloader.dataset.max_positive,
                        num_return_sequences=self.generate_num + testloader.dataset.max_positive,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results_filtered(testloader.dataset.positive, testloader.dataset.id2user, user_idx, \
                                                            self.generate_num+testloader.dataset.max_positive, \
                                                            generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
                
    def test_dataset_task_filtered(self, testloader):
        logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
            
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = int(batch[5][0])
                positive = testloader.dataset.positive[testloader.dataset.id2user[user_idx]]
                
                user_candidate = candidates - positive
                
                candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in user_candidate
                ]
                )
                prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]}')

    def test_dataset_task(self, testloader):
        
        logging.info(f'testing {self.args.datasets} dataset on {self.args.tasks} task')
        test_total = 0
        test_total_explore = 0
        with torch.no_grad():
            total_samples = 0 
            correct_answers = 0
            
            logging.info('##################################################start of prediction')
            for batch in tqdm(testloader):
                
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                output_ids = batch[2].to(self.device)
                output_attention = batch[3].to(self.device)
                
                
                prediction = self.model.generate( 
                        model_explore = self.model_explore if self.explore else None,
                        explore_weight = self.explore_weight,
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_length=256,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]

            
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                

                total_samples += 1
                
                if self.args.datasets == 'hellaswag':
                    
                    gold_answer = gold_sents[0].strip()
                    gold_answer = re.findall(r'ending1|ending2|ending3|ending4', gold_answer)[0]
                    
                    pred_answer = generated_sents[0].strip()
                    pred_answer = re.findall(r'ending1|ending2|ending3|ending4', pred_answer)
                    
                    if not pred_answer:
                        pred_answer = ''
                    else:
                        pred_answer = pred_answer[0]
                    
                    if gold_answer == pred_answer:
                        correct_answers += 1
                
                if self.args.datasets == 'boolq':
                    
                    gold_answer = gold_sents[0].strip()
                    gold_answer = re.findall(r'true|false', gold_answer)[0]
                    
                    pred_answer = generated_sents[0].strip()
                    pred_answer = re.findall(r'true|false', pred_answer)
                    
                    if not pred_answer:
                        pred_answer = ''
                    else:
                        pred_answer = pred_answer[0]
                    
                    if gold_answer == pred_answer:
                        correct_answers += 1
                
                if self.args.datasets in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
                    
                    gold_answer = gold_sents[0].strip()
                    gold_answer = re.findall(r'answer1|answer2|answer3|answer4|answer5', gold_answer)[0]
                    
                    pred_answer = generated_sents[0].strip()
                    pred_answer = re.findall(r'answer1|answer2|answer3|answer4|answer5', pred_answer)
                    
                    if not pred_answer:
                        pred_answer = ''
                    else:
                        pred_answer = pred_answer[0]
                    
                    if gold_answer == pred_answer:
                        correct_answers += 1
                
                if self.args.datasets == 'aqua':
                    
                    gold_answer = gold_sents[0].strip()
                    gold_answer = re.findall(r'A|B|C|D|E', gold_answer)[0]
                    
                    pred_answer = generated_sents[0].strip()
                    pred_answer = re.findall(r'A|B|C|D|E', pred_answer)
                    
                    if not pred_answer:
                        pred_answer = ''
                    else:
                        pred_answer = pred_answer[0]
                    
                    if gold_answer == pred_answer:
                        correct_answers += 1
                        
                if self.args.datasets in ["mawps", "svamp", "gsm8k"]:
                    print(f'Gold Answer: {gold_sents[0]}')
                    print(f'Pred Answer: {generated_sents[0]}')  
           
                    pred_answer = generated_sents[0].replace(',', '')
                    pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
                    if not pred_answer:
                        pred_answer = float('inf')
                    else: pred_answer = float(pred_answer[-1])
                    
                    gold_answer = gold_sents[0].replace(',', '')
                    gold_answer = [s for s in re.findall(r'-?\d+\.?\d*', gold_answer)]
                    gold_answer = float(gold_answer[-1])
                    
                    print(f'Gold Answer [After]: {gold_answer}')
                    print(f'Pred Answer [After]: {pred_answer}')
                    
                    if abs(pred_answer - gold_answer) < 0.01:
                        correct_answers += 1
                
                if total_samples % 100 == 0:
                    logging.info(f'Epoch: {self.epoch_idx} - Samples: {total_samples}, Correct answers: {correct_answers}, Accuracy: {correct_answers/total_samples:.4f}') 
                
            logging.info(f'Epoch: {self.epoch_idx} - Total samples: {total_samples}, Correct answers: {correct_answers}, Accuracy: {correct_answers/total_samples:.4f}')    
