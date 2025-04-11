# Mini GPT-style Language Model with LoRA Fine-tuning on Synthetic Character Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import string
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType

# ----------------------------
# 1. Data Generation
# ----------------------------
def generate_pretrain_data(num_sequences=10000, seq_length=10):
    data = []
    for _ in range(num_sequences):
        seq = ''.join(random.choices(string.ascii_letters, k=seq_length))
        data.append(seq)
    return data

def generate_finetune_data(num_sequences=1000, seq_length=10):
    data = []
    for _ in range(num_sequences):
        chars = []
        prev_val = random.randint(32, 126 - seq_length)  # Start with printable ASCII
        for _ in range(seq_length):
            chars.append(chr(prev_val))
            # Ensure next char has higher ASCII value but stays within printable range
            next_min = prev_val + 1
            next_max = min(prev_val + 10, 126)
            if next_min <= next_max:  # Only generate if valid range exists
                prev_val = random.randint(next_min, next_max)
            else:
                break  # Stop if we can't generate a higher value
        if len(chars) == seq_length:  # Only add complete sequences
            data.append(''.join(chars))
    return data

# ----------------------------
# 2. Dataset and Tokenizer
# ----------------------------
class CharTokenizer:
    def __init__(self):
        chars = string.printable
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char2idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices])

tokenizer = CharTokenizer()

class CharDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.sequences[idx])
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

# ----------------------------
# 3. Mini GPT Model
# ----------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim=16, n_layers=3):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(10, dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=64)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
        # Add config attribute with dictionary-like behavior
        class Config:
            def __init__(self):
                self.use_return_dict = False
                self.vocab_size = vocab_size
                self.hidden_size = dim
                self.num_hidden_layers = n_layers
                self.num_attention_heads = 4
                self.tie_word_embeddings = False
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.config = Config()

    def forward(self, *args, **kwargs):
        # Handle both positional and keyword arguments
        if len(args) > 0:
            x = args[0]
        elif 'input_ids' in kwargs:
            x = kwargs['input_ids']
        elif isinstance(kwargs.get('inputs', None), dict):
            x = kwargs['inputs']['input_ids']
        else:
            raise ValueError("No input provided to forward method")
        
        B, T = x.size()
        x = self.token_embed(x) + self.pos_embed[:T]
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.head(x)

# ----------------------------
# 4. Training Function
# ----------------------------
def train_model(model, dataloader, optimizer, epochs=5, is_lora=False):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if is_lora:
            visualize_lora_params(model, epoch+1)
    return losses

# ----------------------------
# 5. Visualize LoRA Parameters
# ----------------------------
def visualize_lora_params(model, epoch):
    lora_weights = []
    for name, param in model.named_parameters():
        if 'lora' in name and 'weight' in name:
            lora_weights.append(param.detach().cpu().flatten())

    if lora_weights:
        flat = torch.cat(lora_weights)
        plt.figure(figsize=(10, 2))
        plt.hist(flat.numpy(), bins=50, color='blue')
        plt.title(f"LoRA Parameters at Epoch {epoch}")
        plt.show()

# ----------------------------
# 6. Main Script
# ----------------------------

# Pretraining
pretrain_sequences = generate_pretrain_data()
pretrain_dataset = CharDataset(pretrain_sequences, tokenizer)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)
# Save pretrain sequences to file for reproducibility
with open('random_char_data.txt', 'w') as f:
    for seq in pretrain_sequences:
        f.write(seq + '\n')

model = MiniGPT(vocab_size=tokenizer.vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_model(model, pretrain_loader, optimizer, epochs=5)
# Generate some sample outputs from pretrained model
print("\nSample outputs from pretrained model:")
model.eval()
with torch.no_grad():
    for i in range(10):
        # Start with a random character as seed
        seed = random.choice(string.ascii_letters)
        x = torch.tensor([tokenizer.encode(seed)], dtype=torch.long)
        output = []
        
        # Generate 9 more characters
        for _ in range(9):
            logits = model(x)
            next_token = torch.argmax(logits[0, -1]).item()
            output.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
        
        generated = seed + tokenizer.decode(output)
        print(f"Sample {i+1}: {generated}")
model.train()

# Debug: Print model parameter names
print("Model parameter names:")
for name, _ in model.named_parameters():
    print(name)

# LoRA Fine-tuning
finetune_sequences = generate_finetune_data()
finetune_dataset = CharDataset(finetune_sequences, tokenizer)
finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True)
# Save finetune sequences to file for reproducibility
with open('finetune_char_data.txt', 'w') as f:
    for seq in finetune_sequences:
        f.write(seq + '\n')

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=['layers.0.linear1', 'layers.0.linear2', 'head'],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model_lora = get_peft_model(model, lora_config)
optimizer_lora = torch.optim.Adam(model_lora.parameters(), lr=1e-3)
train_model(model_lora, finetune_loader, optimizer_lora, epochs=5, is_lora=True)

