import torch
from src.model import Transformer

# Toy example: reversing a padded sequence
src = torch.tensor([[1, 2, 3, 4, 0, 0]])  # 0 = padding
tgt = torch.tensor([[4, 3, 2, 1, 0, 0]])

model = Transformer(src_vocab=5, tgt_vocab=5, d_model=32, num_heads=4, num_layers=2)

output = model(src, tgt)
print("Model output shape:", output.shape)
