import torch
from model import GPTLanguageModel
from utils import *

# Load dataset to rebuild vocab
with open("data/cricket_laws.txt", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def decode(indices):
    return ''.join(itos[i] for i in indices)

# Load trained model
model = GPTLanguageModel(vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
model.eval()

print("\n🏏 Cricket GPT is ready!")
print("Type your prompt and press ENTER (type 'exit' to stop)\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        print("Goodbye! 👋")
        break

    # Encode only known characters
    encoded = [stoi[c] for c in prompt if c in stoi]
    if not encoded:
        print("⚠ Prompt contains no valid characters from vocab!")
        continue

    idx = torch.tensor(encoded).unsqueeze(0).to(DEVICE)
    out = model.generate(idx, 200)
    response = decode(out[0].tolist())

    print("\n🤖 Cricket GPT:\n")
    print(response)
    print("\n" + "-"*50 + "\n")
