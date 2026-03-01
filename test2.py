"""
Cross-dataset generalizability test.
Model was trained on WELFake titles only.
Tests on ISOT, GossipCop, PolitiFact to check if it generalizes.
Saves separate confusion matrix for each dataset.
"""
import torch
import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

FINETUNED_DIR = "models/distilbert_finetuned"
MAX_LEN = 128
BATCH_SIZE = 64


def extract_embeddings(texts, tokenizer, model, device):
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Extracting"):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch, max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            all_embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embs)


def test_dataset(name, texts, true_labels, tokenizer, model, xgb_model, device, print_samples=False):
    print(f"\n{'='*60}")
    print(f"  Testing on: {name} ({len(texts)} samples)")
    print(f"{'='*60}")

    embs = extract_embeddings(texts, tokenizer, model, device)
    preds = xgb_model.predict(embs)

    if print_samples:
        print("\n  [Detailed Predictions]")
        for text, true_lbl, pred_lbl in zip(texts, true_labels, preds):
            status = " CORRECT  " if true_lbl == pred_lbl else " INCORRECT"
            true_str = "Fake" if true_lbl == 1 else "Real"
            pred_str = "Fake" if pred_lbl == 1 else "Real"
            print(f"  {status} | True: {true_str:4s} | Pred: {pred_str:4s} | Text: {text}")

    correct = (preds == true_labels).sum()
    total = len(true_labels)
    acc = correct / total * 100

    # Per-class accuracy
    real_mask = true_labels == 0
    fake_mask = true_labels == 1
    real_acc = (preds[real_mask] == 0).sum() / real_mask.sum() * 100 if real_mask.sum() > 0 else 0
    fake_acc = (preds[fake_mask] == 1).sum() / fake_mask.sum() * 100 if fake_mask.sum() > 0 else 0

    print(f"  Overall Accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"  Real Accuracy:    {real_acc:.1f}% ({(preds[real_mask]==0).sum()}/{real_mask.sum()})")
    print(f"  Fake Accuracy:    {fake_acc:.1f}% ({(preds[fake_mask]==1).sum()}/{fake_mask.sum()})")

    # Save confusion matrix
    cm = confusion_matrix(true_labels, preds)
    filename = f"cm_{name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Confusion matrix saved to {filename}")

    return acc


def main():
    print("=" * 60)
    print("  Cross-Dataset Generalizability Test")
    print("  Model: Fine-tuned DistilBERT + XGBoost (trained on WELFake)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load models
    print("\nLoading models...")
    tokenizer = DistilBertTokenizer.from_pretrained(FINETUNED_DIR)
    model = DistilBertModel.from_pretrained(FINETUNED_DIR)
    model.to(device)
    model.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_finetuned.json")
    print("  Models loaded!")

    results = {}

    # --- Test 1: ISOT Dataset ---
    try:
        isot_true = pd.read_csv("data/ISOT_TRUE.csv")
        isot_fake = pd.read_csv("data/iSOT_FAKE.csv")
        # Sample 500 from each for speed
        isot_true_sample = isot_true.sample(min(500, len(isot_true)), random_state=42)
        isot_fake_sample = isot_fake.sample(min(500, len(isot_fake)), random_state=42)

        texts = isot_true_sample["title"].tolist() + isot_fake_sample["title"].tolist()
        labels = np.array([0]*len(isot_true_sample) + [1]*len(isot_fake_sample))  # 0=Real, 1=Fake
        results["ISOT"] = test_dataset("ISOT (titles)", texts, labels, tokenizer, model, xgb_model, device)
    except Exception as e:
        print(f"  ISOT error: {e}")

    # # --- Test 2: GossipCop Dataset ---
    # try:
    #     gc_real = pd.read_csv("data/gossipcop_real.csv")
    #     gc_fake = pd.read_csv("data/gossipcop_fake.csv")
    #     gc_real_sample = gc_real.dropna(subset=["title"]).sample(min(500, len(gc_real)), random_state=42)
    #     gc_fake_sample = gc_fake.dropna(subset=["title"]).sample(min(500, len(gc_fake)), random_state=42)

    #     texts = gc_real_sample["title"].tolist() + gc_fake_sample["title"].tolist()
    #     labels = np.array([0]*len(gc_real_sample) + [1]*len(gc_fake_sample))
    #     results["GossipCop"] = test_dataset("GossipCop (titles)", texts, labels, tokenizer, model, xgb_model, device)
    # except Exception as e:
    #     print(f"  GossipCop error: {e}")

    # --- Test 3: PolitiFact Dataset ---
    try:
        pf_real = pd.read_csv("data/politifact_real.csv")
        pf_fake = pd.read_csv("data/politifact_fake.csv")
        pf_real_sample = pf_real.dropna(subset=["title"]).sample(min(500, len(pf_real)), random_state=42)
        pf_fake_sample = pf_fake.dropna(subset=["title"]).sample(min(500, len(pf_fake)), random_state=42)

        texts = pf_real_sample["title"].tolist() + pf_fake_sample["title"].tolist()
        labels = np.array([0]*len(pf_real_sample) + [1]*len(pf_fake_sample))
        results["PolitiFact"] = test_dataset("PolitiFact (titles)", texts, labels, tokenizer, model, xgb_model, device)
    except Exception as e:
        print(f"  PolitiFact error: {e}")

    typed_texts = [
        # === FAKE (15) ===
        # Political conspiracy
        "Hillary Clinton caught running secret email server from underground bunker",
        "Trump secretly signed deal with Russia to rig upcoming elections",
        "Obama administration exposed for secretly spying on political opponents",
        "Democrats caught running illegal voter registration scheme in swing states",
        "Republican senators admit they rigged the election with foreign agents",
        # Health misinformation
        "Drinking bleach cures COVID according to anonymous doctors",
        "Scientists confirm eating chocolate every day completely eliminates cancer cells",
        "Vaccines proven to cause autism by leaked government documents",
        "New study reveals that 5G towers are causing brain tumors in children",
        "Doctors expose secret cure for diabetes that big pharma is hiding from public",
        # Science/conspiracy
        "Earth proven to be flat by leaked NASA internal documents",
        "Bill Gates implanting microchips through vaccines to control population",
        "Elon Musk confirms aliens are living inside Mars",
        "Government whistleblower reveals birds are actually surveillance drones",
        "Moon landing was filmed in Hollywood studio and astronauts confessed",
        # === REAL (15) ===
        # Politics
        "Senate approved new cybersecurity legislation to protect government agencies",
        "Supreme Court ruled in favor of expanding voting rights protections",
        "President signed executive order to address climate change and emissions",
        "Congress passed a bipartisan infrastructure bill after months of debate",
        "Federal Reserve raised interest rates by quarter percentage point citing inflation",
        # Economy
        "Unemployment rate dropped to four percent as economy shows recovery",
        "Stock market closed at record high as tech sector rallied on strong earnings",
        "India GDP growth rate reached seven percent in the latest fiscal quarter",
        "Global oil prices declined after OPEC announced increased production targets",
        "Amazon reported strong quarterly revenue driven by cloud computing growth",
        # Health/Science
        "World Health Organization issued new guidelines for pandemic preparedness",
        "Researchers at MIT developed a new battery technology for electric vehicles",
        "India successfully launched its space mission to study the surface of the sun",
        "Scientists discovered high levels of microplastics in major river systems worldwide",
        "Clinical trials showed promising results for new Alzheimer drug treatment",
        # ===  FAKE (10) ===
        "Secret documents reveal world leaders planning to replace cash with digital currency to track citizens",
        "Whistleblower confirms pharmaceutical companies deliberately spreading diseases for profit",
        "Anonymous sources reveal social media platforms secretly recording private conversations",
        "Leaked report shows climate change is a hoax invented by scientists to get funding",
        "Breaking news reveals US military has been hiding alien spacecraft for decades",
        "Canadian government secretly planning to ban all religious practices by next year",
        "Scientists paid by government to fake data about global warming for political agenda",
        "Massive cover up exposed as hospitals admit to injecting patients with tracking chips",
        "Leaked emails prove that tech companies are controlling peoples minds through smartphones",
        "Underground network of politicians caught selling state secrets to foreign governments",
        # ===  REAL (10) ===
        "Reserve Bank of India maintained repo rate at six point five percent for fourth time",
        "European Union reached agreement on new regulations for artificial intelligence technology",
        "United Nations Security Council held emergency meeting to discuss humanitarian crisis",
        "Government announced new policy to increase renewable energy production by thirty percent",
        "Major airlines reported increase in passenger traffic as international travel restrictions eased",
        "Japan earthquake measured six point two on Richter scale with no casualties reported",
        "Central government allocated additional funding for rural healthcare infrastructure development",
        "New trade agreement between India and Australia expected to boost bilateral exports",
        "Global semiconductor shortage continued to impact automobile production in several countries",
        "International Olympic Committee announced new host city for upcoming summer games",
    ]
    typed_labels = np.array([1]*15 + [0]*15 + [1]*10 + [0]*10)  # Matches text order: 15F, 15R, 10F, 10R
    results["Hand-typed"] = test_dataset("Hand-typed Examples", typed_texts, typed_labels, tokenizer, model, xgb_model, device, print_samples=True)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        bar = "█" * int(acc / 2) + "░" * (50 - int(acc / 2))
        print(f"  {name:15s} | {bar} | {acc:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
