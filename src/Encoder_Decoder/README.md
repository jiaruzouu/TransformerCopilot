# Encoder-Decoder Transformer Copilot 
## Introduction
This code contains the implementation for our encoder-decoder transformer copilot on both finetuning and inference (generation). We use FLAN-T5 and T5 series models in our experiments. We evalaute on 12 datasets across commonsense reasoning, arithmetic reasoning, and downstream generation tasks.

## Environment

```bash
bash environment.sh
# or 
pip3 install -r requirements.txt
```
## Usage

Put the `./Data/Testing` from main root into `./data` folder, and `./Data/Training` from main root into `ft-training_set` folder 

## Finetuning and Inference Running
The command can be found in `./command/command_t5_generation/` folder. Run the command such as:

```
cd command
bash command_t5_generation/commonsense.sh
```
Modify the correponding hyperparameters related to base model and Transformer Copilot in the bash files. The hyperparameter setup is detailed in our submission and codes. The experimental logs can be found in the 'logs' folder, and model saved checkpoints can be found in the 'model' folder. 

