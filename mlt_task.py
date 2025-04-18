#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


#Task 1: Sentence Transformer Model Implementation

# Load the pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample of input sentences
sentences = [
    "Machine learning is transforming the world.",
    "I love exploring deep learning models.",
    "I enjoy going for Marathon but this weather does not allow me to do so."
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Display embeddings
for i, emb in enumerate(embeddings):
    print(f"Sentence {i+1}: {sentences[i]}")
    print(f"Embedding (first 5 values): {emb[:5]}")
    print(f"Total length: {len(emb)}\n")


# In[3]:


#Task#2 to Task#4 
# --- Model --------------------------------------------------------------------
# First Initialize /register initial parameters, load sentence transformer and extract HF model
# Linear layer considering 5 examples; minimizes overfitting risk
# Unfreeze last two transformer layers only. Unfreeze all layers- risk to over-fitting
# mean_pool for single 384 dim. vector per sentence while removing padding noise
# Inference logic of the model

class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name="all-MiniLM-L6-v2", n_topic=3, n_sent=3):
        super().__init__()
        self.backbone = SentenceTransformer(model_name)
        self.encoder  = self.backbone._first_module().auto_model
        dim           = self.backbone.get_sentence_embedding_dimension()

        for p in self.encoder.parameters():
            p.requires_grad = False

        for blk in self.encoder.encoder.layer[-2:]:
            for p in blk.parameters():
                p.requires_grad = True
        self.topic_head = nn.Linear(dim, n_topic)
        self.sent_head  = nn.Linear(dim, n_sent)

    def _mean_pool(self, token_emb, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (token_emb * mask).sum(1)
        counted = mask.sum(1).clamp(min=1e-9)
        return summed / counted

    def forward(self, sentences, task):
        feats = self.backbone.tokenize(sentences)
        ids = feats["input_ids"].to(device)
        att = feats["attention_mask"].to(device)
        out = self.encoder(input_ids=ids, attention_mask=att)
        embed = self._mean_pool(out.last_hidden_state, att)
        if task == "A":
            return self.topic_head(embed)
        if task == "B":
            return self.sent_head(embed)
        raise ValueError("task must be 'A' or 'B'")

# --- Tiny dataset -------------------------------------------------------------
# Minimal PyTorch Dataset class

class MultiTaskDataset(Dataset):
    sents = [
        "The movie was fantastic and thrilling.",
        "The market is crashing rapidly.",
        "Python is a great programming language.",
        "The match ended in a draw.",
        "Inflation is rising due to fuel prices."
    ]
    lbls = {
        "A": [1, 2, 0, 1, 2],  # topic
        "B": [2, 0, 2, 1, 0]   # sentiment
    }
    def __init__(self, task):
        self.task = task
    def __len__(self):
        return len(self.sents)
    def __getitem__(self, idx):
        return self.sents[idx], self.lbls[self.task][idx]

# --- Training -----------------------------------------------------------------
# Updates only last two layers (including heads) during training

model = MultiTaskTransformer().to(device)
opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=5e-5
)

#opt   = torch.optim.AdamW(model.parameters(), lr=5e-5)
lossf = nn.CrossEntropyLoss()
la = DataLoader(MultiTaskDataset("A"), batch_size=2, shuffle=True)
lb = DataLoader(MultiTaskDataset("B"), batch_size=2, shuffle=True)

model.train()
for epoch in range(30):
    for (sa, ya), (sb, yb) in zip(la, lb):
        ya, yb = ya.to(device), yb.to(device)
        pa = model(sa, "A")
        pb = model(sb, "B")
        loss = lossf(pa, ya) + lossf(pb, yb)
        opt.zero_grad(); loss.backward(); opt.step()

# --- Evaluation ---------------------------------------------------------------
model.eval()
train_sents = MultiTaskDataset.sents[:3]
extra_sents = [
    "Oil prices across the world are hurting everyone's wallet.",
    "Mission Impossible latest movie was superb!",
    "I love building AI solution and agents which addresses key society challenges."
]
lookup_topic = {0:"Technology",1:"Entertainment",2:"Finance"}
lookup_sent  = {0:"Negative",1:"Neutral",2:"Positive"}

for group, label in [(train_sents, "TRAIN"), (extra_sents, "UNSEEN")]:
    print(f"\n=== {label} ===")
    with torch.no_grad():
        ta = model(group, "A").argmax(1)
        tb = model(group, "B").argmax(1)
    for s, t, b in zip(group, ta, tb):
        print(f"{s}\n→ Topic: {lookup_topic[t.item()]} | Sentiment: {lookup_sent[b.item()]}\n")

