## Overview
This is the code submission for the Transformer Copilot (T-Copilot). The project is structured into three main directories: `Encoder_Decoder`, `Decoder_Only`, and `Data`.

```markdown
src/ 
├── Encoder_Decoder/ 
│ ├── <files of encoder-decoder T-Copilot> 
├── Decoder_Only/ 
│ ├── <files of decoder-only T-Copilot> 
├── Data/ 
│ ├── <datasets used for fine-tuning and evaluation>
```

### 1. `Encoder_Decoder/`
This directory contains implementations of the Encoder-Decoder T-Copilot. We conduct experiments on FLAN-T5, T-5 Series model and additional baseline models. 

### 2. `Decoder_Only/`
This directory contains implementations of the Decoder-Only T-Copilot. We conduct experiments on LLaMA3 and Qwen2.5 family models such as LLaMA-3.1-8B, Qwen2.5-3B and additional baseline models such as LLaMA-Pro, MergeKit, etc. 

### 3. `Data/`
This folder contains datasets for both fine-tuning and inference.

## Getting Started
Please follow the `README.md` under each folder for further implementation details. 