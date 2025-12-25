# 🧠 Transformer Neural Network (From Scratch — PyTorch)

This repository contains a **minimal, educational, encoder-decoder Transformer implementation** written from scratch in **PyTorch**, along with everything needed to run it locally on Windows.

It is built to help you deeply understand how Transformers work internally — including attention, multi-head communication, positional structure, and layer stacking.

---

## 📌 What `venv` Does in This Project
A **virtual environment (venv)** creates an isolated Python runtime so that:
- Installed packages stay inside this project, not globally
- Dependency versions don’t conflict with other AI/ML work
- Your setup can be reproduced using `requirements.txt`


---

## ⚙️ Setup & Run (Windows)

```bash
# 1. Activate the virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model test
python train.py


How the Transformer Works (Core Concepts)
1️⃣ Embedding Layer (nn.Embedding)

Converts token IDs → dense vectors of length d_model

These vectors are trainable and form the model’s semantic space

Similar words/tokens cluster naturally after training

2️⃣ Positional Encoding

Adds order information because the model reads all tokens at once

Uses sin/cos waves so each position is unique and smooth for neighbors

Added to embeddings without training extra parameters

3️⃣ Attention

Each token produces:

Vector	Meaning
Query (Q)	What the token is looking for
Key (K)	What information the token contains
Value (V)	The actual information to share

Then computes:

Attention Scores = Q × Kᵀ / √d_head
Weights = softmax(scores)
Output = weights × V


This allows direct token-to-token communication.

4️⃣ Multi-Head Attention

Splits embedding dims across multiple heads

Each head learns a different pattern (grammar, structure, meaning, coreference, etc.)

Heads run in parallel, then are concatenated and projected back

5️⃣ Feed-Forward Network

Expands features (4× usually) → applies ReLU/GELU → compresses back

Adds non-linearity and learning capacity

Runs on each token independently at every position

6️⃣ Residual + LayerNorm

Why residual?

x = x + sublayer(x)


✔ Prevents vanishing gradients
✔ Allows deep stacking

Why LayerNorm?

x = norm(x)


✔ Stabilizes training per token
✔ Works better than BatchNorm for NLP

7️⃣ Encoder vs Decoder
Encoder	Decoder
Bidirectional self-attention	Masked self-attention (no future leak)
Builds understanding	Generates outputs
Produces memory	Queries memory using cross-attention

Cross-attention intuition:

Encoder creates knowledge → Decoder retrieves what it needs.

8️⃣ Output Layer
Linear(d_model → vocab_size)


Produces logits, not probabilities

Softmax is applied only during inference or loss calculation

🧪 Toy Runner Goal (train.py)

This script does not perform real training — it simply verifies execution.

Example test:

Input:  [1, 2, 3, 4, 0, 0]
Target: [4, 3, 2, 1, 0, 0]
Task: Reverse while ignoring padding

⚠ Notes

The model is not pre-trained (weights are random)

Padding masks and causal masks are not used in this toy test

This is the classic 2017 encoder-decoder Transformer, not GPT style

The network runs fully offline, without external APIs

🔜 Possible Extensions

You can build on this repo by adding:

Tokenizer + padding masks

Causal (future-blocking) masks

Real text dataset training

Saving/loading model weights

Converting API-generated text into training data

✨ Final Takeaway

A Transformer is built on:

Attention = communication,
FFN = computation,
Positional encoding = structure,
Residual + Norm = stability,
Layer stacking = progressive reasoning.
