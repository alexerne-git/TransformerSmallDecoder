import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
import math
import random
import wandb 
import matplotlib.pyplot as plt
import numpy as np

os.environ['WANDB_API_KEY'] = ''

wandb.init(project="Final_Wandb_Colab", config={
    "batch_size": 128,
    "block_size": 128,
    "max_iters": 600,
    "learning_rate": 3e-4,
    "eval_iters": 50,
    "n_embd": 512,
    "n_head": 8,
    "n_layer": 6,
    "dropout": 0.2,
})
config = wandb.config
batch_size = config.batch_size
block_size = config.block_size
max_iters = config.max_iters
learning_rate = config.learning_rate
eval_iters = config.eval_iters
n_embd = config.n_embd
n_head = config.n_head
n_layer = config.n_layer
dropout = config.dropout


torch.manual_seed(42)
random.seed(42)

file_path = "./dataset.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

lines = data.splitlines(keepends=True)
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
train_index = int(len(lines) * train_ratio)
val_index = train_index + int(len(lines) * val_ratio)
train_text = "".join(lines[:train_index])
val_text = "".join(lines[train_index:val_index])
test_text = "".join(lines[val_index:])

text = "".join(lines)
chars = sorted(list(set(text)))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

class CharDataset(Dataset):
    def __init__(self, text):
        self.data = torch.tensor(encode(text), dtype=torch.long)
    def __len__(self):
        return len(self.data) - config.block_size

def get_sliding_window_batch(dataset, step_size):
    data = dataset.data
    windows, targets = [], []
    for i in range(0, len(data) - config.block_size, step_size):
        windows.append(data[i:i + config.block_size])
        targets.append(data[i + 1:i + config.block_size + 1])
    windows = torch.stack(windows)
    targets = torch.stack(targets)
    indices = torch.randperm(len(windows))[:config.batch_size]
    return windows[indices].to(device), targets[indices].to(device)


# Transformer code 
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) 
        q = self.query(x) 
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) 
        self.proj = nn.Linear(head_size * num_heads, n_embd) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out 

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  
            nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ln1 = nn.LayerNorm(n_embd)  
        self.ffwd = FeedForward(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd) 

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  
        x = x + self.ffwd(self.ln2(x)) 
        return x 

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size) 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape 
        tok_emb = self.token_embedding_table(index) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond) 
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index
# Transformer code 


train_dataset = CharDataset(train_text)
val_dataset = CharDataset(val_text)
test_dataset = CharDataset(test_text)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

best_val_loss = float('inf')
start_time = time.time()

for iter in range(config.max_iters):
    if iter % config.eval_iters == 0:
        model.eval()
        with torch.no_grad():
            train_loss, val_loss = 0.0, 0.0
            for split, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
                total_loss = 0.0
                for _ in range(config.eval_iters):
                    xb, yb = get_sliding_window_batch(dataset, step_size=config.block_size // 2)
                    logits, loss = model(xb, yb)
                    total_loss += loss.item()
                avg_loss = total_loss / config.eval_iters
                wandb.log({f"{split}_loss": avg_loss, "iteration": iter})
                if split == 'train':
                    train_loss = avg_loss
                else:
                    val_loss = avg_loss
                    if avg_loss < best_val_loss:
                        best_val_loss = avg_loss
                        torch.save(model.state_dict(), "best_model.pth")
        print(f"Iteration {iter}: Train Loss {train_loss:.3f}, Val Loss {val_loss:.3f}")

    model.train()
    xb, yb = get_sliding_window_batch(train_dataset, step_size=config.block_size // 2)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    wandb.log({"train_loss_step": loss.item(), "iteration": iter})


window_losses = []
model.eval()
for start in range(0, len(test_dataset.data) - block_size, block_size):
    xb = test_dataset.data[start:start + block_size].unsqueeze(0).to(device)
    yb = test_dataset.data[start + 1:start + block_size + 1].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, loss = model(xb, yb)
        window_losses.append(loss.item())

avg_test_loss = np.mean(window_losses)
std_test_loss = np.std(window_losses)
perplexity = math.exp(avg_test_loss)
perplexity_std = math.exp(std_test_loss)
wandb.log({"test_perplexity": perplexity, "test_perplexity_std": perplexity_std})

print(f"Test Perplexity: {perplexity:.3f} ± {perplexity_std:.3f}")

seed = "O God, O God!"
context = torch.tensor(encode(seed), dtype=torch.long).unsqueeze(0).to(device)
with torch.no_grad():
    generated_tokens = model.generate(context, max_new_tokens=500)
    generated_text = decode(generated_tokens[0].tolist())
    print(f"Generated Text:\n{generated_text}")
    wandb.log({"generated_text": generated_text})

end_time = time.time()
training_time = (end_time - start_time) / 60
print(f"Total training time: {training_time:.2f} minutes")
wandb.log({"training_time_minutes": training_time})

wandb.finish()