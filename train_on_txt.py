import wandb
import tiktoken
import torch.nn as nn
import tiktoken
import random
import time
import time
import torch
import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from dotenv import load_dotenv
from modules import GPT, GPTConfig
load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("Logged in to Weights & Biases.")
else:
    print("WANDB_API_KEY not found. Please check your .env file.")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DataLoaderLite:
    def __init__(self, B, T, split, data_dir='edu_fineweb10B/'):
        self.B = B  # batch size
        self.T = T  # sequence length
        assert split in {'train', 'val'}
        self.split = split
        self.data_dir = data_dir
        self.starting_position = 0

        # Get the shard filenames
        shards = [f for f in os.listdir(data_dir) if f.startswith(f'edufineweb_{split}_') and f.endswith('.npy')]
        self.shards = sorted([os.path.join(data_dir, s) for s in shards])

        assert len(self.shards) > 0, f"No shards found for split {split} in {data_dir}"
        print(f"Found {len(self.shards)} shards for split {split}")

        self.reset()

    def reset(self):
        # Initialize at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def load_tokens(self, shard_path):
        npt = np.load(shard_path)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            if self.current_shard == len(self.shards) - 1:
                print("-----===q---=====-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
                print("-----===q---=====-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
                print(f"Reached end of shard {self.current_shard}, resetting to shard 0")
                print("-----===q---=====-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
                print("-----===q---=====-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
                self.starting_position = 256
            self.current_position = self.starting_position
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
        return x, y


    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()




def get_lr(it, warmup_steps, max_steps, max_lr, min_lr, oscillation_period=2000):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    
    # Basic cosine annealing
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Add oscillation
    oscillation = 0#0.2 * math.sin(2 * math.pi * it / oscillation_period)
    
    # Combine cosine annealing with oscillation
    lr = min_lr + (coeff + oscillation) * (max_lr - min_lr)
    
    # Ensure learning rate stays within bounds
    return max(min(lr, max_lr), min_lr)

def average_absolute_value(tensor):
    return torch.mean(torch.abs(tensor)).item()

def max_absolute_value(tensor):
    return torch.max(torch.abs(tensor)).item()

project_name = "jul_prev_512_layers6_mlp4_prev0"
print(f"Project name: {project_name}")

def train(model_a, train_loader, val_loader, optimizer, criterion, device, config, compile=False, wandb_logging=False, save_model=True):
    if wandb_logging:
        wandb.init(project=project_name, config=config)
    if compile:
        model = torch.compile(model_a)
    else:
        model = model_a
    model.to(device)
    t0 = time.time()
    n2, rl = 0, 0
    for step in range(config.max_steps):
        # Update learning rate
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for param_group in optimizer.param_groups:
            multiplier = param_group.get('lr_multiplier', 1.0)
            param_group['lr'] = lr*multiplier
            if step%10000==19999:
                if param_group.get('named') == 'other':
                    param_group['multiplier'] = max(0.6*multiplier,0.1)
        if step % config.val_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(config.val_steps):
                    x, y = next(val_loader)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits = model(x)
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
            val_loss /= config.val_steps
            print(f"Step {step}: Validation Loss: {val_loss:.4f}")
            if wandb_logging:
                wandb.log({"val_loss": val_loss, "step": step})
        if step > 0 and step % config.generate_interval == 0:
            if compile:
                modelg = GPT(cfg.to_dict())
                modelg.load_state_dict(model_a.state_dict())
                modelg.to(device)
                for i in range(4):
                    textg = generate_text(modelg, device, config, enc)
                    print(f"Generated text: {textg}")
                del modelg
            else:
                textg = generate_text(model, device, config, enc)
                print(f"Generated text: {textg}")
            if wandb_logging:
                wandb.log({"generated_text": textg, "step": step})
        model.train()
        optimizer.zero_grad()
        
        # Gradient accumulation
        accumulated_loss = 0
        for _ in range(config.accumulation_steps):
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / config.accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        n2 = 0.99 * n2 + 0.01 * norm
        rl = 0.99 * rl + 0.01 * accumulated_loss
        optimizer.step()
        t1 = time.time()
        dt = t1 - t0
        if step % config.log_interval == 0:
            print(f"Step {step}: Loss: {accumulated_loss:.4f}, LR: {lr:.6f}, Time: {dt:.2f}s, norm = {norm:.4f}, n2 = {n2:.4f}, rl = {rl:.4f}")

            if wandb_logging:
                wandb.log({
                    "train_loss": accumulated_loss,
                    "learning_rate": lr,
                    "grad_norm": norm,
                    "n2": n2,
                    "rl": rl,
                    "step": step
                })
        if step > 3000 and step % 4000 == 3999 and save_model:
            try:
                if step> 12000:
                    model_a.push_to_hub("noen", config=cfg)
                else:
                    model_a.push_to_hub("none", config=cfg)
            except Exception as e:
                print(f"Error pushing to hub: {e}")
        if step > 500 and step % 1500 == 0:
            config.accumulation_steps = min(config.accumulation_steps*2, config.max_accumulation_steps)
    return model


def generate_text(model, device, config, enc):
    model.eval()
    if random.random() < 0.1:
        start_sentence = "Once upon a time, in a far away land "
    elif random.random() < 0.15:
        start_sentence = "On september 1, 1939, Germany"
    elif random.random() < 0.25:
        start_sentence = "Albert Einstein is famous for"
    elif random.random() < 0.3:
        start_sentence = "The capital of France is"
    elif random.random() < 0.35:
        start_sentence = "In theory, the Earth revolves around"
    elif random.random()<0.5:
        start_sentence = "Hello, I'm a language model"
    else:
        start_sentence = "The quick brown fox jumps over the lazy dog "
    start_tokens = enc.encode(start_sentence)
    context = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(random.randint(1,43))  # Add a seed to config for reproducibility

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(context)  # (B, T, vocab_size)

            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            top_k = getattr(config, 'top_k', 45)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            next_token = torch.gather(topk_indices, -1, ix)  # (B, 1)
            context = torch.cat([context, next_token], dim=1)

    generated_text = enc.decode(context[0].tolist())
    return generated_text

def get_grouped_params(model, tritt_ratio=0.1):
    trittention_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'trittention' in name:
            trittention_params.append(param)
        else:
            other_params.append(param)
    
    return [
        {'params': other_params, 'lr_multiplier': 1.0, 'named': 'other'},
        {'params': trittention_params, 'lr_multiplier': tritt_ratio, 'named': 'trittention'}  
    ]

B, T = 48, 512
cfg = GPTConfig(
    block_size = T,
    vocab_size = 50304,
    n_layer = 6,
    n_head = 8,
    n_embd = 512,
    mlp_ratio = 4,
    n_prev = 0
)

model = GPT(cfg.to_dict())
#model = Transformer.from_pretrained("Gusanidas/mixed-512-6", config=cfg.to_dict())
param_groups = get_grouped_params(model, tritt_ratio=1)

optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), lr=1e-5, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoaderLite(B=B, T=T, split="train")
#train_loader.current_shard = 0
#train_loader.current_position = 12
val_loader = DataLoaderLite(B=B, T=T, split="val")
class TrainConfig:
  max_steps = 16_000
  warmup_steps = 3000
  min_lr = 3e-5
  max_lr = 3e-4
  val_interval = 500
  val_steps = 30
  generate_interval = 2000
  log_interval = 250
  max_grad_norm = 0.9
  start_tokens = [1]  # <|endoftext|>
  max_new_tokens = 150
  accumulation_steps = 1
  max_accumulation_steps = 8


config = TrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"model = {model}") 
print("---====----====++++----")
print(f"number of parameters = {count_parameters(model)}")
train(model, train_loader, val_loader, optimizer, criterion, device, config, compile = False, wandb_logging=True, save_model=False)