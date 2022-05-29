# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb

from torch import cuda
import logging

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)
cnt=0

class KGDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, target_len, model="t5", is_eval=False, batch_gen=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.text = self.data.head_event
        self.ctext = self.data.tail_event
        self.model = model
        self.is_eval = is_eval
        self.batch_gen = batch_gen

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        labels = []

        if self.model in ["t5", "bart"]:
            source = self.tokenizer.batch_encode_plus([text], padding='max_length', max_length=self.source_len, return_tensors='pt', truncation=True)
            target = self.tokenizer.batch_encode_plus([ctext], padding='max_length', max_length=self.target_len, return_tensors='pt', truncation=True)

        elif self.model == "gpt2":
            if self.is_eval:
                if not self.batch_gen:
                    source = self.tokenizer.batch_encode_plus([text], padding=False, max_length=self.source_len, return_tensors='pt', truncation=True)
                    target = self.tokenizer.batch_encode_plus([ctext], padding=False, max_length=self.target_len, return_tensors='pt', truncation=True)
                else:
                    self.tokenizer.padding_side = 'left'
                    source = self.tokenizer.batch_encode_plus([text], padding='max_length', max_length=self.source_len, return_tensors='pt', truncation=True)
                    self.tokenizer.padding_side = 'right'
                    target = self.tokenizer.batch_encode_plus([ctext], padding='max_length', max_length=self.target_len, return_tensors='pt', truncation=True)
            else:
                source = self.tokenizer.batch_encode_plus([text + ' ' + ctext], padding='max_length',
                                                          max_length=self.source_len + self.target_len,
                                                          return_tensors='pt', truncation=True,
                                                          return_overflowing_tokens=True)
                target = source
                head_relation = self.tokenizer.batch_encode_plus([text], padding='max_length', max_length=self.source_len + self.target_len, return_tensors='pt', truncation=True)
                # for computing loss on tail, 
                # label tokens from head and relation by -100, so GPT2LMHealModel will not compute loss on that
                labels = source['input_ids']*(1-head_relation['attention_mask']) - 100*head_relation['attention_mask']
        else:
            raise NotImplementedError

        # if index < 5:
        #     logger.info("Source: {}".format(self.tokenizer.batch_decode(source['input_ids'])))
        #     logger.info("Target: {}".format(self.tokenizer.batch_decode(target['input_ids'])))

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        is_abs = torch.tensor(int('[' in text.split('[GEN]')[0]))

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long),
            'labels': labels,
            'is_abs': is_abs
        }