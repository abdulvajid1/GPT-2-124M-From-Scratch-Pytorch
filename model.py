import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GptConfig
import math 
from huggingface_hub import PyTorchModelHubMixin
import tiktoken
import inspect
import torch.optim as optim
from transformers import PreTrainedModel

import config

# Model architecture
# 2025-09-04-19-34-32.png

# Rotory positional encoding
# 2025-09-06-16-43-19.png
# the rotary matrix with pre-defined parameters Θ = {θi = 10000^(−2(i−1)/d),i∈[1,2,...,d/2]}. 

class RotoryPositionalEncoding(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.config = config
        self.precompute_thetas()
        
    def precompute_thetas(self):
        head_dim = self.config.d_model // self.config.n_heads # 128
        thetas = 1 / (torch.pow(10000, torch.arange(0, head_dim , 2)/head_dim))  # (64)
        token_positions = torch.arange(self.config.context_len) # (1024) $$\theta$$
        pos_thetas = torch.outer(token_positions, thetas) # (1024, 64) = ($$m_\theta$$) each pos head_dim have a theta
        self.register_buffer('cos_thetas', pos_thetas.cos().unsqueeze(0).unsqueeze(-2)) 
        self.register_buffer('sin_thetas', pos_thetas.sin().unsqueeze(0).unsqueeze(-2)) # (1, seq_len, 1, 64) for broadcasting
        
    def forward(self, x):
        bcz, seq_len, _, _ = x.size() # (bcz, seq, heads, head_dim)
        
        even_x = x[..., 0::2] # (bcz, seq, head, head_dim//2)
        odd_x = x[..., 1::2] # (bcz, seq, head, head_dim//2)
        
        # rotate
        even_rot = even_x * self.cos_thetas[..., :seq_len, :, :] - odd_x * self.sin_thetas[..., :seq_len, :, :]
        odd_rot = odd_x * self.cos_thetas[..., :seq_len, :, :] + even_x * self.sin_thetas[..., :seq_len, :, :]
        
        x_stacked = torch.stack((even_rot, odd_rot), dim=-1) # bcz,seq,d_model,2

        return x_stacked.flatten(-2)
    
  
class FeedForward(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.intermidiate_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.intermidiate_size, config.d_model)    
        )
        
    def forward(self, x):
        return self.feed_forward(x)
 
    
class RMSNorm(nn.Module):
    def __init__(self, config: GptConfig, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return x_norm * self.scale



class GPTAttention(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()

        self.config = config
        self.Q_w = nn.Linear(config.d_model, config.d_model)
        self.K_w = nn.Linear(config.d_model, config.d_model)
        self.V_w = nn.Linear(config.d_model, config.d_model)
        self.O_w = nn.Linear(config.d_model, config.d_model)
        self.rope = RotoryPositionalEncoding(config)
        
    def forward(self, x: torch.Tensor):
        self.batch_size, self.seq_len = x.shape[0], x.shape[1]
        assert self.config.d_model % self.config.n_heads == 0, "d_model should divisible by n_heads"
        self.head_dim = self.config.d_model // self.config.n_heads
        
        query = self.Q_w(x)
        key = self.K_w(x)
        value = self.V_w(x)
        
        # (b, s, d_model) -> view -> (batch_size, seq_len, num_head, head_dim) -> permute ->(b, num_head, s, head_dim)
        query = query.view(self.batch_size, self.seq_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(self.batch_size, self.seq_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(self.batch_size, self.seq_len, self.config.n_heads, self.head_dim).transpose(1, 2)

        query = self.rope(query)
        key = self.rope(key) 
        
        ctx_embd = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.05)
        ctx_embd = ctx_embd.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.config.n_heads * self.head_dim)
        return self.O_w(ctx_embd)


class DecoderBlock(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        
        self.attention = GPTAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
    

class GPT(PreTrainedModel):
    config_class = GptConfig
    
    def __init__(self, config: GptConfig):
        super().__init__(config)
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_rms_norm = RMSNorm(config)
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    
    def forward(self, x, target=None):
     
        batch_size, seq_len = x.size()
        assert seq_len <= self.config.context_len, "seq length should be less than context length"

        # token_embedding -> decoder_layers -> final_layernorm -> fin_layer
        x = self.tok_embed(x)
        for block in self.decoder_blocks:
            x = block(x)
            
        x = self.final_rms_norm(x)
        logits = self.final_layer(x) # (b, seq, vocab_size)

        loss = None
        if target != None:
            loss = F.cross_entropy(logits.view(batch_size*seq_len, -1), target.view(-1))
        
        return {'logits' : logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, text, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        idx = torch.tensor(self.tokenizer.encode_batch(text)).to(self.config.device)
        max_len = max_new_tokens if max_new_tokens else self.config.context_len
        
        for _ in range(max_len):
            logits = self(idx, target=None)['logits']
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)  
            if top_k is not None:
                topk_probs, topk_vocab_indieces = torch.topk(probs, top_k, dim=-1)
                next_token_idx = torch.multinomial(topk_probs, num_samples=1)
                next_token = torch.gather(topk_vocab_indieces, -1, next_token_idx)
                idx = torch.cat([idx, next_token], dim=-1)
            else: 
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=-1)
        idx = idx.detach().cpu().tolist()
        out = self.tokenizer.decode_batch(idx)
        return out
    
    def configure_optimizer(self, lr, weight_decay):
        
        params = {n: p for n, p in self.named_parameters()}
        params = {n: p for n, p in params.items() if p.requires_grad}

        decay_params = [p for n, p in params.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params.items() if p.dim() < 2]

        param_grps = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"total num of decay parms :{num_decay_params} with num tensor of {len(decay_params)}")
        print(f"total num of no decay parms :{num_nodecay_params} with num tensor of {len(nodecay_params)}")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in self.config.device
        print(f'Using fused: {use_fused}')
        optimizer = optim.AdamW(param_grps, lr=lr, betas=(.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
    
    def num_parameters(self):
        trainable_params = 0
        non_trainable_params = 0
        
        for params in self.parameters():
            if params.requires_grad:
                trainable_params+=params.numel()
            else:
                non_trainable_params+=params.numel()
            
        print(f"""Trainable Parameteres {trainable_params}
            Non-Trainable Parameteres {non_trainable_params}""")
    
def main():
    config = GptConfig()
    gpt = GPT(config)
    gpt.eval()
    gpt.to(config.device)
    sample_input = ['hai what is'] * 5
    out_list = gpt.generate(sample_input, max_new_tokens=15)
    for out in out_list:
        print(out)
    

if __name__ == '__main__':
    main()

