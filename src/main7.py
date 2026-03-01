"""
Step 2: Extract embeddings from fine-tuned DistilBERT, then train XGBoost.
Uses the [CLS] token embedding from the fine-tuned model as features.
"""
import pandas as pd
import numpy as np
import torch
import joblib
import os
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

FINETUNED_DIR = "models/distilbert_finetuned"
MAX_LEN = 128
BATCH_SIZE = 64


def extract_embeddings_batch(texts, tokenizer, model, device, max_len=128, batch_size=64):
    """Extract [CLS] embeddings from fine-tuned DistilBERT in batches."""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # [CLS] token is the first token — use its hidden state as embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def main():
    print("=" * 60)
    print("  XGBoost with Fine-tuned DistilBERT Embeddings")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load fine-tuned model (base model for embeddings, not classification head)
    print("\n[1/5] Loading fine-tuned DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(FINETUNED_DIR)
    model = DistilBertModel.from_pretrained(FINETUNED_DIR)
    model.to(device)
    print("  Model loaded!")

    # Load data
    print("\n[2/5] Loading WELFake dataset...")
    df = pd.read_csv("data/WELFake_Dataset.csv")
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    texts = df["title"].tolist()
    labels = df["label"].values  # 0=Real, 1=Fake
    print(f"  {len(texts)} titles loaded")

    # Extract embeddings
    print("\n[3/5] Extracting fine-tuned embeddings...")
    embeddings = extract_embeddings_batch(texts, tokenizer, model, device, MAX_LEN, BATCH_SIZE)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Train XGBoost
    print("\n[4/5] Training XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    xgb_model.save_model('models/xgboost_finetuned.json')
    print("  XGBoost model saved to models/xgboost_finetuned.json")

    # Evaluate
    print("\n[5/5] Evaluating...")
    y_pred = xgb_model.predict(X_test)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Fine-tuned DistilBERT + XGBoost)")
    plt.tight_layout()
    plt.savefig('confusion_matrix_finetuned.png')
    print("  Confusion matrix saved to confusion_matrix_finetuned.png")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
