import torch
import os
from model import GPTLanguageModel
from torch.optim import AdamW
from utils import *

with open("data/cricket_laws.txt", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = max(1, int(0.9 * len(data)))  # safe split even if tiny
train_data, val_data = data[:n], data[n:]

# ---- FIX: ensure dataset is large enough ----
if len(train_data) <= BLOCK_SIZE:
    print(f"Dataset too small ({len(train_data)} chars). Lowering BLOCK_SIZE to {len(train_data)-1}")
    BLOCK_SIZE = len(train_data) - 1

def get_batch(split):
    d = train_data if split == "train" else val_data
    max_index = len(d) - BLOCK_SIZE
    if max_index <= 0:
        x = d[:BLOCK_SIZE].unsqueeze(0)
        y = d[1:BLOCK_SIZE+1].unsqueeze(0)
        return x.to(DEVICE), y.to(DEVICE)
    ix = torch.randint(0, max_index, (BATCH_SIZE,))
    x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = GPTLanguageModel(vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

print("Starting training...\n")

for i in range(5000):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f"[TRAIN] step={i} loss={loss.item():.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print(f"\nTraining complete! Model saved → {MODEL_OUT}")
