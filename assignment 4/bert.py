# pip install transformers torch scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 1. LOAD DATA
# ----------------------------
print("Loading data...")
train_path = "train_data.csv"
test_path = "test_data.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Make sure train_data.csv and test_data.csv exist in the directory.")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ----------------------------
# 2. HANDLE COLUMN NAMES & SAMPLING
# ----------------------------
def get_cols(df):
    text_col, label_col = None, None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["text", "review", "comment"]):
            text_col = col
        if any(keyword in col.lower() for keyword in ["label", "sentiment", "target"]):
            label_col = col
    
    if text_col is None: text_col = df.columns[0]
    if label_col is None: label_col = df.columns[1]
    return text_col, label_col

train_text_col, train_label_col = get_cols(train_df)
test_text_col, test_label_col = get_cols(test_df)

# Sampling 100 for verification, can increase to 5000/1000 later
train_df = train_df[[train_text_col, train_label_col]].rename(columns={train_text_col: "text", train_label_col: "label"}).sample(min(100, len(train_df)))
test_df = test_df[[test_text_col, test_label_col]].rename(columns={test_text_col: "text", test_label_col: "label"}).sample(min(20, len(test_df)))

# Convert labels to int
train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce').fillna(0).astype(int)
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').fillna(0).astype(int)

train_df.dropna(subset=['text'], inplace=True)
test_df.dropna(subset=['text'], inplace=True)

print(f"Sampled Dataset: Train({len(train_df)}), Test({len(test_df)})")

# ----------------------------
# 3. PREPROCESSING
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)  # Replace HTML tags with space
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# ----------------------------
# 4. SPLIT TRAIN → TRAIN + VAL
# ----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.1, random_state=42
)

test_texts = test_df['text']
test_labels = test_df['label']

# ----------------------------
# 5. TOKENIZATION
# ----------------------------
print("Tokenizing data...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# ----------------------------
# 6. DATASET CLASS
# ----------------------------
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# ----------------------------
# 7. MODEL SETUP
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# 8. TRAIN FUNCTION (WITH OPTIMIZER PARAM)
# ----------------------------
def train_model(model, train_loader, optimizer, epochs=1):
    losses = []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return losses

# ----------------------------
# 9. EVALUATION FUNCTION (CPU SAFE)
# ----------------------------
def evaluate(model, data_loader):
    model.eval()
    model.to(device)
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels_cpu)

    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )

    return acc, precision, recall, f1, predictions, true_labels

# ----------------------------
# 10. RUN EXPERIMENTS
# ----------------------------

print("\n--- Standard BERT Fine-Tuning ---")
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
losses = train_model(model, train_loader, optimizer, epochs=1)

# Plot Loss
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Standard Training Loss")
plt.savefig("loss_plot.png")
print("Saved loss_plot.png")

# Test Evaluation
acc, p, r, f, preds, true = evaluate(model, test_loader)
print(f"Results: Acc={acc:.2f}, P={p:.2f}, R={r:.2f}, F1={f:.2f}")

# Confusion Matrix
cm = confusion_matrix(true, preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# 11. EXPERIMENT 1 (FREEZE BERT BACKBONE)
print("\n--- Experiment 1: Frozen BERT ---")
model_frozen = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
for param in model_frozen.bert.parameters():
    param.requires_grad = False

optimizer_f = AdamW(model_frozen.parameters(), lr=2e-5)
train_model(model_frozen, train_loader, optimizer_f, epochs=1)

# 12. EXPERIMENT 2 (UNFREEZE LAST 2 LAYERS)
print("\n--- Experiment 2: Partial Fine-Tuning ---")
model_partial = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
for name, param in model_partial.bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer_p = AdamW(model_partial.parameters(), lr=2e-5)
train_model(model_partial, train_loader, optimizer_p, epochs=1)

print("\nAll experiments completed successfully.")
