import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import re
from sklearn.utils import shuffle

obama = Path("obama_speeches")
obama_files = obama.glob("*.txt")

obama_all = ""

for file in obama_files:
    text = file.read_text(encoding="utf-8", errors="ignore")
    obama_all += text

trump = Path("trump_speeches")
trump_files = trump.glob("*.txt")

trump_all = ""

for file in trump_files:
    text = file.read_text(encoding="utf-8", errors="ignore")
    trump_all += text

custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['Applause'])

wc_obama = WordCloud(width=600, height=400, 
                     background_color="white",
                     stopwords=custom_stopwords).generate(obama_all)

wc_trump = WordCloud(width=600, height=400, 
                     background_color="white",
                     stopwords=custom_stopwords).generate(trump_all)

fig, axes = plt.subplots(1,2, figsize=(12,6))
axes[0].imshow(wc_obama, interpolation='bilinear')
axes[0].axis("off")
axes[0].set_title("Obama")
axes[1].imshow(wc_trump, interpolation='bilinear')
axes[1].axis("off")
axes[1].set_title("Trump")

# create labelling by phrase structure

def corpus(data):
    sentences = re.split(r"[.!?]", data)
    for i,sentence in enumerate(sentences):
        s = sentence
        if sentence.startswith(" "):
            s = sentence[1:]
        if sentence.endswith(" "):
            s = sentence[:-1]
        if sentence.startswith("\n"):
            s = sentence.lstrip("\n")
        if sentence.endswith("\n"):
            s = sentence.rstrip("\n")
        sentences[i] = s

    sentences = [sentence.lower().split(" ") for sentence in sentences]
    
    return sentences

trump_sentences = corpus(trump_all)
obama_sentences = corpus(obama_all)

all_words = trump_sentences + obama_sentences
all_words = list(set([word for sentence in all_words for word in sentence]))

main_dict = {word: (i+1) for i, word in enumerate(all_words)}

def to_numeric(dict, data):
    main = []
    for sentence in data:
        s = [dict[word] for word in sentence]
        main.append(s)

    return main

trump_coded = to_numeric(main_dict, trump_sentences)
obama_coded = to_numeric(main_dict, obama_sentences)

TRUMP = 0
OBAMA = 1
trump_labelled = [(vec, TRUMP) for vec in trump_coded]
obama_labelled = [(vec, OBAMA) for vec in obama_coded]

all_labelled = trump_labelled + obama_labelled 

X = [item1 for item1, _ in all_labelled]
y = [item2 for _, item2 in all_labelled]

X, y = shuffle(X, y, random_state=10)

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

# pad
MAXLEN = 20

for i,sentence in enumerate(X):
    if len(sentence) > MAXLEN:
        s = sentence[:MAXLEN]
    if len(sentence) < MAXLEN:
        s = sentence + [0]*(MAXLEN-len(sentence))
    X[i] = s


X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.float)

NUMSAMPLES = X.shape[1]
BATCHSIZE = 4

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
dataset = TextDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        emb = emb*mask
        lengths = mask.sum(dim=1).clamp(min=1)
        avg = emb.sum(dim=1)/lengths
        logits = self.fc(avg)

        return logits.squeeze(1)
    

VOCABSIZE = max(list(main_dict.values())) + 1

model = TextClassifier(VOCABSIZE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_acc = (preds == y_batch).float().mean().item()
            total_correct += (preds==y_batch).sum().item()
            total_samples += y_batch.size(0)

    epoch_acc = total_correct / total_samples
    print(f"Epoch accuracy: {epoch_acc}")

    
def predict(model, sentence):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([sentence], dtype=torch.long)
        logit = model(x)
        prob = torch.sigmoid(logit).item()
        return prob

reverse_dict = {v:k for k, v in main_dict.items()}

test_trump = [main_dict[word] for word in trump_sentences[5]]
test_obama = [main_dict[word] for word in obama_sentences[5]]

print(predict(model, test_trump))
print(predict(model, test_obama))









