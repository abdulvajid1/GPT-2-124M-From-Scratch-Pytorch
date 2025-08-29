from config import GptConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_data_loader
from model import GPT
import tiktoken
from utils import load_checkpoint, save_checkpoint
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

torch.set_float32_matmul_precision('high') # all matmul become fast (not how weigts store)

def train(model, optimizer,config, loader, epoch, writer, save_step, save_path):
    progress_bar = tqdm(enumerate(loader), leave=True, desc=f'Epoch {epoch}: ', total=len(loader), dynamic_ncols=True)
    for step, (x, y) in progress_bar:
        global_step = epoch*len(loader) + step
        x, y = x.to(config.device), y.to(config.device)
        
        # t1 = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
            output = model(x, y)
            logits, loss = output['logits'], output['loss']
            # import code; code.interact(local=locals())
        
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # torch.cuda.synchronize()
        # t2 = time.time()
        # time_took = (t2 - t1) * 1000 # time will be in milliseconds
        # token_per_sec = (config.batch_size * config.context_len)/ (t2 - t1)
        # print(f"token per second {token_per_sec} and time took {time_took}")
        
        progress_bar.set_postfix(loss=loss.item(), grad_norm=norm.item())
        writer.add_scalar('Training loss', loss.item(), global_step)
        writer.add_scalar('Gradient norm', norm.item(), global_step)
        
        if (step+1) % save_step == 0:
            save_checkpoint(path=save_path,
                model=model,
                optimizer=optimizer)
        
        

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'

    print(f'Using Device: {device}')
    
    config = GptConfig(vocab_size=50304)
    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPT(config).to(config.device)
    model = torch.compile(model)
    loader = get_data_loader(tokenizer, config.context_len, config.batch_size, num_workers=4, pin_memory=True)
    print(f'Total number of samples: {len(loader)}')
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(.9, 0.95), weight_decay=0.1)
    save_path = 'checkpoints'
    save_step = 500
    save_path = 'checkpoints/ckpt.pt'
    writer = SummaryWriter(log_dir='runs')
    
    if config.load_checkpoint:
        load_checkpoint(path=save_path, model=model, optimizer=optimizer)
    
    for epoch in range(config.n_epoch):
        train(model, optimizer,config, loader, epoch, writer, save_step, save_path)
        
        # generate sample after each epoch
        sample_input = ['you are a'] * 5
        out_list = model.generate(sample_input, temperature=1.0)
        for out in out_list:
            print(out)

if __name__ == "__main__":
    main()
            