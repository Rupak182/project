"""
Predict using fine-tuned DistilBERT embeddings + XGBoost.
"""
import torch
import xgboost as xgb
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

FINETUNED_DIR = "models/distilbert_finetuned"
MAX_LEN = 128


def load_models():
    print("Loading fine-tuned DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(FINETUNED_DIR)
    bert_model = DistilBertModel.from_pretrained(FINETUNED_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    print("Loading XGBoost model...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_finetuned.json")

    print("Models loaded!\n")
    return tokenizer, bert_model, xgb_model, device


def predict_news(text, tokenizer, bert_model, xgb_model, device):
    # 1. Tokenize
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 2. Get fine-tuned [CLS] embedding
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 3. XGBoost prediction
    pred = xgb_model.predict(cls_embedding)
    proba = xgb_model.predict_proba(cls_embedding)

    # WELFake: 0 = Real, 1 = Fake
    label = "Real" if pred[0] == 0 else "Fake"
    confidence = proba[0][pred[0]] * 100
    return label, confidence


def main():
    tokenizer, bert_model, xgb_model, device = load_models()

    print("--- Fake News Detection (Fine-tuned DistilBERT + XGBoost) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            text = input("Enter news text: ")
            if text.lower() in ['exit', 'quit']:
                break
            if not text.strip():
                print("Empty input. Try again.")
                continue

            prediction, confidence = predict_news(text, tokenizer, bert_model, xgb_model, device)
            print(f"\n=> Predicted Label: ** {prediction.upper()} ** (Confidence: {confidence:.1f}%)\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
