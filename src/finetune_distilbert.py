"""
Fine-tune DistilBERT on WELFake titles for fake news classification.
Uses titles only for consistency with prediction input.
Saves fine-tuned model to models/distilbert_finetuned/
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import os

# ── Config ──
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 128
SAVE_DIR = "models/distilbert_finetuned"


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def main():
    print("=" * 60)
    print("  Fine-tuning DistilBERT on WELFake Titles")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load data
    print("\n[1/4] Loading WELFake dataset...")
    df = pd.read_csv("data/WELFake_Dataset.csv")
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]

    texts = df["title"].tolist()
    labels = df["label"].tolist()  # WELFake: 0=Real, 1=Fake
    print(f"  {len(texts)} titles loaded")
    print(f"  Real(0): {labels.count(0)} | Fake(1): {labels.count(1)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # Tokenizer & Model
    print("\n[2/4] Loading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Datasets
    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(X_val, y_val, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training
    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += len(labels_batch)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["label"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += len(labels_batch)

        val_acc = val_correct / val_total * 100
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # Save
    print(f"\n[4/4] Saving fine-tuned model to {SAVE_DIR}...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("  Model saved successfully!")

    print("\n" + "=" * 60)
    print("  Fine-tuning Complete!")
    print(f"  Final Val Accuracy: {val_acc:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
