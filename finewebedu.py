import os 
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


local_dir = 'edu_fineweb10B'
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards


# Create the cache the local directory if it does not exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split='train[:5%]')

# init tokenizer 
tokenizer = tiktoken.get_encoding('gpt2')
eot = tokenizer._special_tokens['<|endoftext|>']

def tokenize(doc):
    # tokenize a single document and return numpy array of uint16
    tokens = [eot]
    tokens.extend(tokenizer.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dict is too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16
    

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)
    

nprocs = max(1, os.cpu_count()//2)

with mp.Pool(nprocs) as pool:
    shard_index = 0
    
    # preallocate buffer to hold current shards
    all_tokens_np = np.empty(shape=(shard_size, ), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for  tokens in pool.imap(func=tokenize, iterable=fw, chunksize=16):
        
        if token_count + len(tokens) < shard_size: # if shard_size is not full, add next shard to current shard index
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count += len(tokens)
        else:
            # save current shard
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(os.path.dirname(__file__), f"edufineweb_{split}_{shard_index: 06d}")
            remider = shard_size - token_count
            all_tokens_np[token_count: token_count + remider] = tokens[:remider]
            progress_bar.update(remider)
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            
            # send remaining data with next buffer
            all_tokens_np[0: len(tokens)-remider] = tokens[remider:]
            token_count = len(tokens) - remider
            
            
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(os.path.dirname(__file__), f"edufineweb_{split}_{shard_index: 06d}")
        write_datafile(filename, all_tokens_np)

                        
            
    