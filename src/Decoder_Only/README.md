# Decoder-Only Transformer Copilot

## Introduction
This code contains the implementation for our decoder-only transformer copilot on both finetuning and inference (generation). We use LLaMA3 and Qwen2.5 family models in our experiments. We evalaute on 12 datasets across commonsense reasoning, arithmetic reasoning, and downstream generation tasks.

## Environment

### Prerequisites
- Python 3.11.9
- PyTorch
- Transformers
- Other dependencies specified in `requirements.txt`

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Put the `./Data/Testing` from main root into `./dataset` folder, and `./Data/Training` from main root into `ft-training_set` folder.

## Runing
### Training(finetune.py)
In this file, we use a conventional training loop to incporate our Pilot Model with Copilot Model and adapt to our framework's fine-tuning procedure. We ensure the accuracy achieves the same comparied with Transformer trainer. 

The hyperparameter setup is detailed in our submission and codes. 

### Evaluation (evaluate.py)
In this file, we evaluate the Transformer Copilot across multiple reasoning and downstream tasks. We modify the original `peft` and `Transforme` package to adapt to our framework's inference regime. 

The hyperparameter setup is detailed in our submission and codes. 

### Getting Started
An end-to-end runnig shell is provided in the folder of `script`. 

*Example run:*
```bash
bash script/arithmetic.sh 
```

Note that the model saved checkpoints for both Pilot and Copilot model is in folder `trained_models`, the running logs is in folder `logs`, and the inference output is stored in the fodler `experiment`. 