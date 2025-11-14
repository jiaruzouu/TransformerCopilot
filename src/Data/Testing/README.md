# Evaluation Dataset

We use **12 publicly available datasets** across three categories—**Commonsense Reasoning, Arithmetic Reasoning, and Downstream Recommendation**—to evaluate the generation performance of **Transformer-Copilot** and baseline methods.

To begin, download the public datasets and organize them as described below.

## Dataset Categories

We use the following datasets for evaluation:

- **Commonsense Reasoning:** PIQA, HellaSwag, WinoGrande, BoolQ, SIQA, and OpenbookQA
- **Arithmetic Reasoning:** AQuA, GSM8K, MAWPS, and SVAMP
- **Downstream Recommendation:** Beauty and LastFM

## Dataset Structure

Once downloaded, arrange the datasets in the following directory structure:

```markdown
./Data/Testing/ 
<!-- commonsense -->
├── PIQA/ 
│ ├── test.json 
├── HellaSwag/ 
│ ├── test.json 
├── WinoGrande/ 
│ ├── test.json 
├── BoolQ/ 
│ ├── test.json 
├── SIQA/ 
│ ├── test.json 
├── OpenbookQA/ 
│ ├── test.json 

<!-- arithmetic -->
├── AQuA/ 
│ ├── test.json 
├── GSM8K/ 
│ ├── test.json 
├── MAWPS/ 
│ ├── test.json 
├── SVAMP/ 
│ ├── test.json 

<!-- downstream recommendation -->
├── Beauty/ 
│ ├── test.json 
├── LastFM/ 
│ ├── test.json 
```

## Dataset Format

For evaluation, we use each dataset's testing set and follows a structured JSON format where each data instance contains an **instruction**, an **input (if applicable)**, an **output**, and the corresponding **answer**.

### Example: `test.json` in the BoolQ dataset
```json
[
    {
        "instruction": "Please answer the following question with true or false, question: are billy and nora from the thundermans twins?\n\nAnswer format: true/false",
        "input": "",
        "output": "the correct answer is false",
        "answer": "false"
    },
    {
        "instruction": "Please answer the following question with true or false, question: is world at war part of black ops?\n\nAnswer format: true/false",
        "input": "",
        "output": "the correct answer is true",
        "answer": "true"
    },
    {
        "instruction": "Please answer the following question with true or false, question: is white christmas a sequel to holiday inn?\n\nAnswer format: true/false",
        "input": "",
        "output": "the correct answer is false",
        "answer": "false"
    },
    {
        "instruction": "Please answer the following question with true or false, question: is a descant recorder the same as a soprano?\n\nAnswer format: true/false",
        "input": "",
        "output": "the correct answer is true",
        "answer": "true"
    },
    // ... Additional data instances
]
```
