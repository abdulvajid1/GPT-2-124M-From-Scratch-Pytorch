from datasets import load_dataset
import tiktoken
import numpy as np
import os

repo_name = "Elriggs/openwebtext-100k"

# data & tokenizer
ds = load_dataset(repo_name, split='train')
tokenizer = tiktoken.get_encoding('gpt2')

eot = tokenizer.eot_token
CONTEXT_LEN = 1024

def tokenize(doc):
    all_tokens = []
    input_tokens = []
    target = []
    token_lists = tokenizer.encode_ordinary_batch(doc['text'])
    for tokens in token_lists:
        all_tokens.extend([eot] + tokens)
        
    for i in range(0, len(all_tokens) - CONTEXT_LEN, CONTEXT_LEN):
        input_tokens.append(all_tokens[i : i + CONTEXT_LEN])
        target.append(all_tokens[i+1 :i + CONTEXT_LEN + 1])
        
    return {'input': input_tokens, "labels": target}

ds_map = ds.map(tokenize, batch_size=100000, batched=True, remove_columns=['text'])

filepath = os.path.join(os.path.dirname(__file__), 'pretrain_data', )
os.makedirs(filepath, exist=True)

ds_map.save_to_disk('./pretrain_data/openwebtext_tokenized')


