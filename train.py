from config import GptConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_data_loader
from model import GPT
import tiktoken
from utils import load_checkpoint, save_checkpoint
import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import math
torch.set_float32_matmul_precision('high') # all matmul become fast (not how weigts store)

WARMUP_STEPS = 10
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.10
MAX_STEPS = 100


def train(model, optimizer, config: GptConfig, loader, epoch, grad_accumulation_step, writer, save_step, save_path, max_step):
    progress_bar = tqdm.tqdm(range(max_step), leave=True, desc=f'Epoch {epoch}: ', total=len(range(max_step)), dynamic_ncols=True)
    loader_iter = iter(loader)
    
    for step in progress_bar:
        loss_accm = 0.0
    
        # Gradient Accumulation   
        for _ in tqdm.trange(grad_accumulation_step, desc='Grad Accm Steps', leave=True, dynamic_ncols=True):
            
            # if itration ends start new loader
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)   # restart loader
                x, y = next(loader_iter)
            x, y = x.to(config.device), y.to(config.device)
            
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
             
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        
        global_step = epoch*len(loader) + step
            
        
        # Update learning rate    
        lr = get_lr(step)
        for params in optimizer.param_groups:
            params['lr'] = lr
            
        # Tensor boared Logging    
        writer.add_scalar('Training loss', loss_accm.item(), global_step)
        writer.add_scalar('Gradient norm', norm.item(), global_step)
        
        # Checkpointing
        if (step+1) % save_step == 0:
            save_checkpoint(path=save_path,
                model=model,
                optimizer=optimizer)
        
        
        progress_bar.set_postfix(loss=f"{loss_accm.item(): .6f}",
                                 norm=f"{norm: .4f}")




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

    # Calc grad accumulation step needed.
    grad_accumulation_step = desired_batch_size // micro_batch_size_token
    
    print(f'Desired total batch_size {desired_batch_size}')
    print(f'Max Micro Batch size that can fit on gpu : {micro_batch_size_token}')
    print(f'Grad accumualtion step needed for Desired batch size (MAX_BATCH/MICRO BATCH) : {grad_accumulation_step}')
    return grad_accumulation_step
    

def main():
    config = GptConfig(vocab_size=50304)
    device = get_device()
    
    # Gradient Accumulation Step
    max_batch_size_token = 524288 # a good power of two number close 0.5M token batch size according gpt paper
    micro_batch_size_token = (config.batch_size * config.context_len)
    grad_accumulation_step = calc_grad_accumulation_step(max_batch_size_token, micro_batch_size_token)
    
    # Model & tokenizer 
    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPT(config).to(config.device)
    optimizer = model.configure_optimizer(MAX_LR, weight_decay=config.weight_decay)
    model = torch.compile(model)
    
    loader = get_data_loader(tokenizer, config.context_len, config.batch_size, num_workers=2, pin_memory=True)
    print(f'Total number of samples: {len(loader)}')
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(.9, 0.95), weight_decay=0.1)
    
    save_path = 'checkpoints'
    save_step = 500
    save_path = 'checkpoints/ckpt.pt'
    writer = SummaryWriter(log_dir='runs')
    
    if config.load_checkpoint:
        load_checkpoint(path=save_path, model=model, optimizer=optimizer)
    
    for epoch in range(config.n_epoch):
        train(model, optimizer,config, loader, epoch, grad_accumulation_step, writer, save_step, save_path, max_step=MAX_STEPS) # will change max_step to num accumualted batches (data is insufficient to set such amount of batch size)
        
        # generate sample after each epoch
        sample_input = ['you are a'] * 5
        out_list = model.generate(sample_input, temperature=1.0)
        for out in out_list:
            print(out)

if __name__ == "__main__":
    main()
            