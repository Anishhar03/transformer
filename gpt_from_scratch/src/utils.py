import os
from dotenv import load_dotenv
load_dotenv()

DEVICE = os.getenv("DEVICE", "cpu")
EPOCHS = int(os.getenv("EPOCHS", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", 256))
LR = float(os.getenv("LEARNING_RATE", 5e-4))
MODEL_OUT = os.getenv("MODEL_OUT", "models/cricket_gpt.pt")
