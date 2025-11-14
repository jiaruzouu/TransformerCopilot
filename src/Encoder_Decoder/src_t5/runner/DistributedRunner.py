from runner.SingleRunner import SingleRunner
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
from utils import utils
import torch
import utils.generation_trie as gt
import utils.evaluate as evaluate
from torch.utils.data.distributed import DistributedSampler
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import numpy as np
import random
import wandb
import pdb
import re 

class DistributedRunner(SingleRunner):
    
    def __init__(self, model, model_explore, tokenizer, train_loader, valid_loader, test_loader, device, args, rank):
        super().__init__(model, model_explore, tokenizer, train_loader, valid_loader, test_loader, device, args)
        self.rank = rank
        self.model = DDP(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)
        if self.explore:
            self.model_explore = DDP(self.model_explore, device_ids=[self.args.gpu], find_unused_parameters=True)
        
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
            if self.rank == 0:
                logging.info(f"Start training for epoch {epoch+1}")
            
            if epoch > 4 and self.explore:
                self.explore_weight = 1
                
            dist.barrier()
            if self.regenerate_candidate:
                for ds in self.train_loader.dataset.datasets:
                    ds.generate_candidates()
                    ds.construct_sentence()
            elif self.reconstruct_data:
                for ds in self.train_loader.dataset.datasets:
                    ds.construct_sentence()
                    
            dist.barrier()
            
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
                
                output = self.model.module(
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

                        explore_loss = explore_loss.view(explore_B,explore_L) * explore_lm_mask
                        explore_loss = (explore_loss.sum(dim=1) / explore_lm_mask.sum(dim=1).clamp(min=1)).mean()
                    elif self.f2_loss == 'mse':
                        explore_loss = explore_output["explore_loss"]

                    elif self.f2_loss == 'rmse':
                        explore_loss = explore_output["explore_loss"]

                    
                # update
                loss.backward()
                if self.explore: 

                    explore_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                if self.explore:
                    torch.nn.utils.clip_grad_norm_(self.model_explore.parameters(), self.args.clip)
                dist.barrier()
                
                self.optimizer.step()
                self.scheduler.step()
                if self.explore:
                    self.explore_optimizer.step()
                    self.explore_scheduler.step()
                self.model.zero_grad()
                if self.explore:
                    self.model_explore.zero_grad()
                
                dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                
                if self.explore:
                    dist.all_reduce(explore_loss.detach(), op=dist.ReduceOp.SUM)
                    explore_loss /= dist.get_world_size()
                
                dist.barrier()
                
                if self.rank == 0:
                    losses.append(loss.detach())
                    if self.explore:
                        explore_losses.append(explore_loss.detach())

                i += 1
                if self.rank == 0 and i % 100 == 0:
                    train_batch_loss = sum(losses)/len(losses)
                    train_batch_explore_loss = sum(explore_losses)/len(explore_losses) if self.explore else 0
                    print(f"Training loss for batch {i}: {train_batch_loss}; Training explore loss for batch {i}: {train_batch_explore_loss}")
                    if i % 500 == 0:
                        logging.info(f"Training loss for batch {i}: {train_batch_loss}; Training explore loss for batch {i}: {train_batch_explore_loss}")
                    if self.wandb:
                        wandb.log({"train_batch_loss": train_batch_loss})
                        if self.explore: wandb.log({"train_batch_explore_loss": train_batch_explore_loss})
    
            if self.rank == 0:
                train_epoch_loss = sum(losses)/len(losses)
                train_losses.append(train_epoch_loss)
                if self.explore:
                    train_epoch_explore_loss = sum(explore_losses)/len(explore_losses)
                    explore_train_losses.append(train_epoch_explore_loss)
                    logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}; The average training explore loss for epoch {epoch+1} is {train_epoch_explore_loss}")
                else:
                    logging.info(f"The average training loss for epoch {epoch+1} is {train_epoch_loss}")
            
            if self.epoch_idx == 5 or self.epoch_idx == 10:
                self.test()
        
            if self.valid_select > 0:
                if self.rank == 0:
                    logging.info(f"Start validation for epoch {epoch+1}")
                losses = []
                self.model.eval()
                with torch.no_grad():
                    if self.args.valid_prompt_sample > 0:
                        for ds in self.valid_loader.dataset.datasets:
                            ds.construct_sentence()
                    for batch in tqdm(self.valid_loader):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        output_ids = batch[2].to(self.device)
                        output_attention = batch[3].to(self.device)

                        output = self.model.module(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=self.args.alpha,
                            return_dict=True,
                        )

                        loss = output["loss"]
                        lm_mask = output_attention != 0
                        lm_mask = lm_mask.float()
                        B, L = output_ids.size()
                        loss = loss.view(B, L) * lm_mask
                        loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                        dist.barrier()

                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()

                        dist.barrier()

                        if self.rank == 0:
                            losses.append(loss)

                    if self.rank == 0:
                        valid_epoch_loss = sum(losses)/len(losses)
                        valid_losses.append(valid_epoch_loss)
                        logging.info(f"The average valid loss for epoch {epoch+1} is {valid_epoch_loss}")

                        if valid_epoch_loss == min(valid_losses):
                            logging.info(f"The minimal validation loss so far.")
                            best_epoch = epoch + 1
                            torch.save(self.model.module.state_dict(), self.args.model_path)
                            logging.info(f"Save the current model to {self.args.model_path}")
                
            if self.test_epoch > 0:
                if (epoch + 1) % self.test_epoch == 0:
                    self.model.eval()
                    self.test()
            
            dist.barrier()
        if self.valid_select > 0:
            if self.rank == 0:
                logging.info(f"The best validation at Epoch {best_epoch}")
        else:
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), self.args.model_path)
                logging.info(f"Save the current model to {self.args.model_path}")
        
        return
    

                
    def test(self, path=None, explore_path=None):
        self.model.eval()
        if self.explore:
            self.model_explore.eval()
        if path:
            self.model.module.load_state_dict(torch.load(path, map_location=self.device))
            if self.explore:
                self.model_explore.module.load_state_dict(torch.load(explore_path, map_location=self.device))
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
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
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
                user_idx = batch[5]
        
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
                
                rel_results = evaluate.rel_results_filtered(testloader.dataset.positive, testloader.dataset.id2user, user_idx.detach().cpu().numpy(), \
                                                            self.generate_num+testloader.dataset.max_positive, \
                                                            generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
    
    def test_dataset_task_filtered(self, testloader):
        if self.rank == 0:
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
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
            
    def test_dataset_task(self, testloader):
        if self.rank == 0:
            logging.info(f'testing {self.args.datasets} dataset on {self.args.tasks} task')
        test_total = 0
        test_total_explore = 0
        dist.barrier()
        with torch.no_grad():
            total_samples = 0 
            correct_answers = 0
            
            if self.args.rank == 0:
                logging.info('##################################################start of prediction##################################################')
                logging.info(f"Explore_weight:{self.explore_weight}")
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                output_ids = batch[2].to(self.device)
                output_attention = batch[3].to(self.device)
                
                prediction = self.model.module.generate(
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
                    else:
                        pred_answer = float(pred_answer[-1])
                    
                    gold_answer = gold_sents[0].replace(',', '')
                    gold_answer = [s for s in re.findall(r'-?\d+\.?\d*', gold_answer)]
                    gold_answer = float(gold_answer[-1])
                    
                    print(f'Gold Answer [After]: {gold_answer}')
                    print(f'Pred Answer [After]: {pred_answer}')
                    
                    if abs(pred_answer - gold_answer) < 0.01:
                        correct_answers += 1
                    
            dist.barrier()
            total_samples_tensor = torch.tensor(total_samples, dtype=torch.float32, device=self.device)
            correct_answers_tensor = torch.tensor(correct_answers, dtype=torch.float32, device=self.device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_answers_tensor, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                logging.info(f'Epoch - {self.epoch_idx}, Total Samples: {total_samples_tensor.item()}; Correct Answers: {correct_answers_tensor.item()}; Accuracy: {correct_answers_tensor.item()/total_samples_tensor.item():.4f}')
