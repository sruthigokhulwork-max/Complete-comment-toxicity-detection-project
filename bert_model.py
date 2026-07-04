# bert_model.py
# Fine-tunes DistilBERT for multi-label toxicity detection
# Optimized for Apple M4 with MPS acceleration

# Step 1: Imports
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Step 2: Device setup — M4 Mac uses MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M4 MPS acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Step 3: Constants
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128       # shorter than LSTM's 200 — BERT handles context better
BATCH_SIZE   = 16        # safe for 16GB M4
EPOCHS       = 3         # 3 epochs enough for fine-tuning pre-trained BERT
LR           = 2e-5      # standard BERT fine-tuning learning rate
LABEL_COLS   = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

# Step 4: Load and prepare data
print("\nLoading data...")
train_df = pd.read_csv("train.csv")

# Use 20% of data for faster fine-tuning on M4
# Full dataset would take 8+ hours; 20% gives strong results in 1-2 hours
train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
print(f"Training on {len(train_df):,} samples (20% of full dataset)")
print(f"Label distribution:\n{train_df[LABEL_COLS].sum()}")

# Split into train and validation
train_data, val_data = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42
)
print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

# Step 5: Tokenizer
print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


# Step 6: Dataset class
class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(),
            "attention_mask" : encoding["attention_mask"].squeeze(),
            "labels"         : torch.tensor(
                self.labels[idx], dtype=torch.float
            )
        }


# Step 7: Create datasets and dataloaders
train_dataset = ToxicityDataset(
    texts     = train_data["comment_text"].values,
    labels    = train_data[LABEL_COLS].values,
    tokenizer = tokenizer,
    max_len   = MAX_LEN
)

val_dataset = ToxicityDataset(
    texts     = val_data["comment_text"].values,
    labels    = val_data[LABEL_COLS].values,
    tokenizer = tokenizer,
    max_len   = MAX_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches  : {len(val_loader)}")

# Step 8: Load DistilBERT model
print("\nLoading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    problem_type="multi_label_classification"
)
model = model.to(device)
print("Model loaded and moved to device")

# Step 9: Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# Loss function — binary cross entropy for multi-label
pos_weight = torch.tensor(
    [1.0, 5.0, 1.0, 20.0, 1.0, 8.0]
).to(device)  # toxic, severe_toxic, obscene, threat(20x), insult, identity_hate(8x)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Step 10: Training function
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"  Batch {batch_idx}/{len(loader)} "
                f"— Loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


# Step 11: Evaluation function
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Sigmoid to convert logits to probabilities
            preds = torch.sigmoid(outputs.logits)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate AUC per label
    auc_scores = {}
    for i, label in enumerate(LABEL_COLS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores[label] = round(float(auc), 4)
        except Exception:
            auc_scores[label] = 0.0

    mean_auc = np.mean(list(auc_scores.values()))
    return auc_scores, mean_auc


# Step 12: Training loop
print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Device: {device}")
print("=" * 50)

best_auc  = 0
best_path = "bert_best_model"

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)

    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    print(f"Train Loss: {train_loss:.4f}")

    auc_scores, mean_auc = evaluate(model, val_loader, device)
    print(f"Val Mean AUC: {mean_auc:.4f}")
    for label, auc in auc_scores.items():
        print(f"  {label:<20} {auc:.4f}")

    # Save best model
    if mean_auc > best_auc:
        best_auc = mean_auc
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        print(f" New best model saved — AUC: {best_auc:.4f}")

print("\n" + "=" * 50)
print(f"Training complete. Best Mean AUC: {best_auc:.4f}")

# Step 13: Save final metrics
bert_metrics = auc_scores.copy()
bert_metrics["mean_auc"] = round(best_auc, 4)

with open("bert_metrics.json", "w") as f:
    json.dump(bert_metrics, f, indent=4)
print("Saved: bert_metrics.json")

