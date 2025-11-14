# Finetuning Dataset

We use the **COMMONSENSE_170K** and **MATH_10K** datasets for fine-tuning **Transformer-Copilot** and baseline methods.

To get started, download the public datasets and follow the directory structure detailed below.

## Dataset Structure

Once downloaded, the datasets should be placed in the following directory:


```markdown
./Data/Training/ 
├── commonsense_170k.json 
├── math_10k.json

```


## Dataset Format

For fine-tuning, each dataset follows a structured JSON format where each data instance contains an **instruction**, an **input (if applicable)**, an **output**, and the corresponding **answer**.

### Example: `commonsense_170k.json`
```json
[
    {
        "instruction": "Please answer the following question with true or false: Do Iran and Afghanistan speak the same language?\n\nAnswer format: true/false",
        "input": "",
        "output": "The correct answer is true.",
        "answer": "true"
    },
    {
        "instruction": "Please choose the correct answer to fill in the blank: Ian taught Nelson how to land a ferocious punch, so _ could fight like a monster.\n\nOption1: Ian  Option2: Nelson  \nAnswer format: option1/option2",
        "input": "",
        "output": "The correct answer is option2.",
        "answer": "option2"
    }
    // ... Additional data instances
]
```

### Example: `math_10k.json`
```json
[
    {
        "instruction": "At the arcade, Dave won 11 tickets. If he spent 5 tickets on a beanie and later won 10 more tickets, how many would he have?",
        "input": "",
        "output": "Dave initially had 11 tickets. He spent 5 tickets on a beanie, leaving him with:\n\n11 - 5 = 6 tickets\n\nHe later won 10 more tickets, so his total number of tickets would be:\n\n6 + 10 = 16 tickets\n\nTherefore, Dave would have 16 tickets after spending 5 tickets on a beanie and winning 10 more tickets. The answer in Arabic numerals is:\n\n16",
        "answer": "16.0"
    },
    {
        "instruction": "Anthony has 16 pets. This morning, he forgot to lock the door and lost 6 pets. After that, 1/5 of his remaining pets died from old age. How many pets does he have left?",
        "input": "",
        "output": "Step 1: Subtract the number of lost pets from the total number of pets to find how many Anthony has left: \n16 - 6 = 10\n\nStep 2: Find how many pets died from old age by multiplying the remaining pets by 1/5: \n10 × 1/5 = 2\n\nStep 3: Subtract the number of pets that died from old age from the remaining pets: \n10 - 2 = 8\n\nTherefore, Anthony has 8 pets left. The answer in Arabic numerals is:\n\n8",
        "answer": "8.0"
    }
    // ... Additional data instances
]
```