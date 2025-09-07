from config import GptConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_data_loader
from model import GPT
import tiktoken
from utils import load_checkpoint, save_optimizer
import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
from huggingface_hub import login
from dotenv import load_dotenv
import argparse
from huggingface_hub import snapshot_download
from save_checkpoint_hf_hub import save_to_hf

hf_save_step = 500

torch.set_float32_matmul_precision('high') # all matmul become fast (not how weigts store)

WARMUP_STEPS = 50
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.10
MAX_STEPS = 20000
SAVE_STEP = 100
SAVE_PATH = os.path.join(os.getcwd(), 'checkpoints')
logging_path = os.path.join(os.getcwd(), 'runs')

# Create the checkpoint Dir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)
    
# Mkdir Logging dir
if not os.path.exists(logging_path):
    os.makedirs(logging_path, exist_ok=True)

# HF Hub Checkpoint
repo_id = "Abdulvajid/gpt2-from-scratch"  # Replace with the desired model's repo ID

def train(model, optimizer, config: GptConfig, loader, epoch, grad_accumulation_step, writer, SAVE_STEP, SAVE_PATH, max_step, global_step=None):
    progress_bar = tqdm.tqdm(range(max_step), leave=True, desc=f'Epoch {epoch}: ', total=len(range(max_step)), dynamic_ncols=True)
    loader_iter = iter(loader)
    
    for _ in progress_bar:
        loss_accm = 0.0
    
        # Gradient Accumulation   
        for _ in tqdm.trange(grad_accumulation_step, desc='Grad Accm Steps', leave=True, dynamic_ncols=True):
            
            # Get Next batch, if iteration over start new
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
                
            x, y = batch['input'].to(config.device), batch['labels'].to(config.device)
            
            
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                # t1 = time.time()
                output = model(x, y)
                logits, loss = output['logits'], output['loss']
                
                # torch.cuda.synchronize()
                # t2 = time.time()
                # time_took = (t2 - t1) * 1000 # time will be in milliseconds
                # token_per_sec = (config.batch_size * config.context_len * grad_accumulation_step)/ (t2 - t1)
                # print(f"token per second {token_per_sec} and time took {time_took}")
                
                # import sys; sys.exit(0);
                # import code; code.interact(local=locals())
            
            loss = loss / grad_accumulation_step    
            loss.backward()
            loss_accm += loss.detach()
        
        # Weight clipping     
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        
            
        
        # Update learning rate    
        lr = get_lr(global_step)
        for params in optimizer.param_groups:
            params['lr'] = lr
            
        # Tensor boared Logging    
        writer.add_scalar('Training loss', loss_accm.item(), global_step)
        writer.add_scalar('Gradient norm', norm.item(), global_step)
        writer.add_scalar('Learning rate Decay', lr, global_step)
        
        # Save Checkpoint
        if (global_step) % SAVE_STEP == 0:
            print(f'Saving the model after {global_step} steps')
            
            # Create new checkpoint path for updated model
            checkpoint_path = os.path.join(SAVE_PATH,f'ckpt_{global_step}')
            save_path, _ = os.path.split(checkpoint_path)
            os.makedirs(checkpoint_path, exist_ok=True)
            
            model.save_pretrained(config=config, save_directory=checkpoint_path)
            save_optimizer(path=save_path,
                optimizer=optimizer,
                global_step=global_step)
        
        # Save to hf
        if (global_step) % hf_save_step == 0:
            save_to_hf(global_step)
        
        
        progress_bar.set_postfix(loss=f"{loss_accm.item(): .6f}",
                                 norm=f"{norm: .4f}",
                                 new_lr=f"{lr: .5f}",
                                 global_step=global_step)




def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * ((it + 1) / WARMUP_STEPS)
    
    if it > MAX_STEPS:
        return MIN_LR
    
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS) #initially small number then become close to 1
    assert 0<=decay_ratio<=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)        
    
    

def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'
         
    print(f'Using Device {device}')
    return device



def calc_grad_accumulation_step(desired_batch_size, micro_batch_size_token):
    
    assert desired_batch_size % micro_batch_size_token == 0
    grad_accumulation_step = desired_batch_size // micro_batch_size_token # Calc grad accumulation step needed.
    print(f'Desired total batch_size {desired_batch_size}')
    print(f'Max Micro Batch size that can fit on gpu : {micro_batch_size_token}')
    print(f'Grad accumualtion step needed for Desired batch size (MAX_BATCH/MICRO BATCH) : {grad_accumulation_step}')
    return grad_accumulation_step

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--download_checkpoint", action='store_true', default=False)
    parser.add_argument("--hf_login", type=str, default=None)
    return parser.parse_args()
    

def main():
    args = get_argparser()
    device = get_device()
    config = GptConfig(vocab_size=50304, d_model=1024, n_layers=32, n_heads=8, device=device)
    
    if args.hf_login:
        login(args.hf_login)
    
    if args.download_checkpoint:
        snapshot_download(repo_id=repo_id, local_dir=SAVE_PATH)
        
    
    # Gradient Accumulation Step
    max_batch_size_token = 524288 # a good power of two number close 0.5M token batch size according gpt paper
    micro_batch_size_token = (config.batch_size * config.context_len)
    grad_accumulation_step = calc_grad_accumulation_step(max_batch_size_token, micro_batch_size_token)
    
    # Model & tokenizer 
    tokenizer = tiktoken.get_encoding('gpt2')

    # Loader
    loader = get_data_loader(config.batch_size, num_workers=2, pin_memory=True)
    print(f'Total number of batches: {len(loader)}')
    
    # Tensorboard Login
    writer = SummaryWriter(log_dir='runs')
    
    # Load Model if exist
    if args.load_checkpoint != None:
        
        # create full path
        checkpoint_path = os.path.join(SAVE_PATH, args.load_checkpoint)
        assert os.path.exists(checkpoint_path), f'{checkpoint_path} file does not exist {os.path.exists(checkpoint_path)}' # checks if file exist
        
        model, optimizer, global_step = load_checkpoint(config=config, 
                                      checkpoint_path=checkpoint_path,  
                                      device=device) # Load model, optimizer return global step if exist else return 0
        model = torch.compile(model)
        print(f"Loading model from step {global_step}")
        
    else:
        # initialize model
        model = GPT(config).to(config.device)
        optimizer = model.configure_optimizer(MAX_LR, weight_decay=config.weight_decay)
        model = torch.compile(model)
        global_step = 0
        print('Starting Fresh training from global step 0')
        
        
    # Train 
    for epoch in range(config.n_epoch):
        train(model, optimizer,config, loader, epoch, grad_accumulation_step, writer, SAVE_STEP, SAVE_PATH, max_step=MAX_STEPS, global_step=global_step) # will change max_step to num accumualted batches (data is insufficient to set such amount of batch size)
        
        # generate sample after each epoch
        sample_input = ['you are a'] * 5
        out_list = model.generate(sample_input, temperature=1.0)
        for out in out_list:
            print('#'*50)
            print(out)
            
        if MAX_STEPS:
            break





if __name__ == "__main__":
    main()
            