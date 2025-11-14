import torch
import os
import argparse
import logging
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from data.MultiTaskDataset import MultiTaskDataset
from runner.SingleRunner import SingleRunner
from runner.DistributedRunner import DistributedRunner
from processor.Collator import Collator, TestCollator
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
from processor.DistMultiDataTaskSampler import DistMultiDataTaskSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Config
from model.FLAN_T5 import P5_T5
from model.explore_net import exploration_net
from utils import utils
from utils import initialization
from torch.nn.parallel import DistributedDataParallel as DDP
import pdb
from datetime import timedelta
from datasets import load_dataset

def get_dataset(args):
    # init dataset
    
    train_all_datasets = []
    valid_all_datasets = []
    
    TrainDataset = MultiTaskDataset(args, args.generation_train_file, 'train')
    train_all_datasets.append(TrainDataset)
    if args.valid_select > 0:
        assert args.valid_select == 0
        ValidDataset = MultiTaskDataset(args,  args.generation_train_file, 'validation')
        valid_all_datasets.append(ValidDataset)
        
    TrainSet = ConcatDataset(train_all_datasets)
    if args.valid_select > 0:
        ValidSet = ConcatDataset(valid_all_datasets)
    else:
        ValidSet = None
    return TrainSet, ValidSet


def get_loader(args, tokenizer, TrainSet, ValidSet, rank=0):
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        args.distributed = True
    else:
        args.distributed = False
    
    collator = Collator(args,tokenizer)

    if args.distributed:
        train_sampler = DistributedSampler(TrainSet, num_replicas=ngpus_per_node, rank=rank, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        dataset=TrainSet, 
        batch_size=args.batch_size, 
        collate_fn=collator, 
        sampler=train_sampler,  # Use sampler instead of shuffle
        shuffle=False if args.distributed else True
    )

    if args.valid_select > 0:
        if args.distributed:
            valid_sampler = DistributedSampler(ValidSet, num_replicas=ngpus_per_node, rank=rank, shuffle=False)
        else:
            valid_sampler = None

        valid_loader = DataLoader(
            dataset=ValidSet, 
            batch_size=args.batch_size, 
            collate_fn=collator, 
            sampler=valid_sampler,
            shuffle=False
        )
    else:
        valid_loader = None
    
    return train_loader, valid_loader



def get_test_dataset(args):

    test_all_datasets = []
    
    TestDataset = MultiTaskDataset(args, args.generation_test_file, 'test')
    test_all_datasets.append(TestDataset)

    TestSet = ConcatDataset(test_all_datasets)
    return TestSet

def get_testloader(args, tokenizer, TestSet, rank=0):
    # Get the number of available GPUs
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = ngpus_per_node > 1  # Enable distributed training only if multiple GPUs are available
    
    collator = Collator(args,tokenizer)

    # Use DistributedSampler for multi-GPU inference
    if args.distributed:
        test_sampler = DistributedSampler(TestSet, num_replicas=ngpus_per_node, rank=rank, shuffle=False)
    else:
        test_sampler = None  # Sequential loading for single-GPU

    test_loader = DataLoader(
        dataset=TestSet, 
        batch_size=args.eval_batch_size, 
        collate_fn=collator, 
        sampler=test_sampler,  # Use sampler for distributed inference
        shuffle=False  # No shuffling for evaluation
    )

    return test_loader



def single_main():
    # running on single gpu.
    duplicate_test = False
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)
    parser = SingleMultiDataTaskSampler.parse_sampler_args(parser)
    parser = SingleRunner.parse_runner_args(parser)
    
    args, extras = parser.parse_known_args()
    
    utils.setup_logging(args)
    utils.setup_model_path(args)
    utils.set_seed(args.seed)
    
    args.rank = 0
    
    device = torch.device("cuda", int(args.gpu.split(',')[0]))
    
    logging.info("##Experiment: Test openbookqa on FlanT5 baseline; ##Key-Parameters: Set explore_weight=1 and explore=True")
           
    logging.info(vars(args))
    
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    
    # init dataset and dataloader
    TrainSet, ValidSet = get_dataset(args)
    train_loader, valid_loader = get_loader(args, tokenizer, TrainSet, ValidSet)
    
    TestSet = get_test_dataset(args)
    test_loader = get_testloader(args, tokenizer, TestSet)
    
    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        # increase layers from 6 to 12
        
        if duplicate_test:
            config.num_layers *= 2
            config.num_decoder_layers *= 2
        logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementError
        
    model = P5_T5.from_pretrained(args.backbone, config=config)
    model.to(device)
    model_explore = None
    if args.explore_flag:
        model_explore = exploration_net.from_pretrained(args.backbone, config=config)
        model_explore.to(device)
    
    model.resize_token_embeddings(len(tokenizer))
    if args.explore_flag:
        model_explore.resize_token_embeddings(len(tokenizer))
    
    if args.random_initialize == 1:
        logging.info("Random initialize number related tokens")
        initialization.random_initialization(model, tokenizer, args.backbone)
        if args.explore_flag:
            initialization.random_initialization(model_explore, tokenizer, args.backbone)
    
    if args.load:
        logging.info(f"Load model from {args.model_path}")
        model = utils.load_model(model, args.model_path, args)
        if args.explore_flag:
            model_explore = utils.load_model(model_explore, args.model_explore_path, args)
    
    runner = SingleRunner(model, model_explore, tokenizer, train_loader, valid_loader, test_loader, device, args)
    
    if args.train:
        logging.info("Start training")
        runner.train()
    
    logging.info(f"Load model from {args.model_path}")
    runner.test(args.model_path)
    
    
    
def distributed_launch():
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)
    parser = DistMultiDataTaskSampler.parse_sampler_args(parser)
    parser = DistributedRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()
    
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    
    
    
    mp.spawn(
        distributed_main, args=(args, ), nprocs=ngpus_per_node, join=True
    )
    
    
def distributed_main(local_rank, args):
    
    # distributed learning
    duplicate_test = False
    args.rank = local_rank
    utils.set_seed(args.seed)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", world_size=args.world_size, rank=local_rank, timeout=timedelta(seconds=5000)
    )
    utils.setup_logging(args)
    utils.setup_model_path(args)
    
    if args.rank == 0:
        logging.info(f"##Experiment: Test {args.datasets} on FlanT5 baseline; ##Key-Parameters: Set explore_weight=0 and explore=Flase")
        logging.info(vars(args))
    TrainSet, ValidSet = get_dataset(args)
    
    device = f"cuda:{local_rank}"
    args.gpu = local_rank
    
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    
    train_loader, valid_loader = get_loader(args, tokenizer, TrainSet, ValidSet, local_rank)
    
    TestSet = get_test_dataset(args)
    test_loader = get_testloader(args, tokenizer, TestSet)
    
    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        if duplicate_test:
            config.num_layers *= 2
            config.num_decoder_layers *= 2
        if local_rank == 0:
            logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementError
        
    
        
    model = P5_T5.from_pretrained(args.backbone, config=config)
    model.to(device)
    model_explore = None
    if args.explore_flag:
        model_explore = exploration_net.from_pretrained(args.backbone, config=config)
        model_explore.to(device)
    
    
        
    if args.item_indexing == 'collaborative':
        for ds in train_loader.dataset.datasets:
            tokenizer.add_tokens(ds.new_token)
    model.resize_token_embeddings(len(tokenizer))
    if args.explore_flag:
        model_explore.resize_token_embeddings(len(tokenizer))
    
    if args.random_initialize == 1:
        if local_rank == 0:
            logging.info("Random initialize number related tokens")
        initialization.random_initialization(model, tokenizer, args.backbone)
        if args.explore_flag:
            initialization.random_initialization(model_explore, tokenizer, args.backbone)
    
    if args.load:
        if local_rank == 0:
            logging.info(f"Load model from {args.model_path}")
        model = utils.load_model(model, args.model_path, args, loc=device)
        model.to(device)
        if args.explore_flag:
            model_explore = utils.load_model(model_explore, args.model_explore_path, args, loc=device)
            model_explore.to(device)
    # if local_rank == 0:
        
    runner = DistributedRunner(model, model_explore, tokenizer, train_loader, valid_loader, test_loader, device, args, local_rank)
    
    if args.train:
        if local_rank == 0:
            logging.info("Start training")
        runner.train()
    dist.barrier()
    
    if local_rank == 0:
        logging.info(f"Load model from {args.model_path}")
    runner.test(args.model_path)
         
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    init_args, extras = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = init_args.gpu
    ngpus_per_node = torch.cuda.device_count()
    if init_args.distributed and ngpus_per_node > 1:
        distributed_launch()
    else:
        single_main()