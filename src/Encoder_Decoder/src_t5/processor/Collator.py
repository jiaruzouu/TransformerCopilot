import torch
import pdb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class Generation_Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        output_ids = [torch.tensor(item["labels"]) for item in batch]

        # Dynamically pad each batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_masks = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "output_ids": output_ids,
            "attention_mask": attention_masks
        }
class Collator:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        
        input_texts = [input_text['input'] for input_text in batch]
        output_texts = [input_text['output'] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]

        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]
        
        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )
    
class TestCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        user_idx = [input_text['user_idx'] for input_text in batch]
        input_texts = [input_text['input'] for input_text in batch]
        output_texts = [input_text['output'] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
            torch.tensor(user_idx),
        )
