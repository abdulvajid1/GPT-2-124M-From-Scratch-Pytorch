from datasets import load_dataset
import tiktoken
import numpy as np
import os
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_dataset', type=str, default="Elriggs/openwebtext-100k")
    return parser.parse_args()



def main():
    args = get_argparser()
    repo_name = args.hf_dataset

    # data & tokenizer
    ds = load_dataset(repo_name, split='train')
    print(f'Num rows in hf dataset {len(ds)}')
    
    tokenizer = tiktoken.get_encoding('gpt2')
    eot = tokenizer.eot_token
    CONTEXT_LEN = 1024

    def tokenize(doc):
        all_tokens = []
        input_tokens = []
        target = []
        token_lists = tokenizer.encode_ordinary_batch(doc['text'])
        
        # add eot token to each doc and extend all to get single list of batch doc wit eot token at the end of each doc
        for tokens in token_lists:
            all_tokens.extend([eot] + tokens)
        
        # Slide through the singl list to get input and target and create new dict input-target    
        for i in range(0, len(all_tokens) - CONTEXT_LEN, CONTEXT_LEN):
            input_tokens.append(all_tokens[i : i + CONTEXT_LEN])
            target.append(all_tokens[i+1 :i + CONTEXT_LEN + 1])
            
        return {'input': input_tokens, "labels": target}

    # apply function to data
    ds_map = ds.map(tokenize, batch_size=10000, batched=True, remove_columns=['text'])

    # Create a folder for pretrained data
    filepath = os.path.join(os.path.dirname(__file__), 'pretrain_data')
    os.makedirs(filepath, exist=True)

    ds_map.save_to_disk('./pretrain_data/openwebtext_tokenized')
    

if __name__ == "__main__":
    main()


