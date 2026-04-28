"""
Cross-dataset generalizability test for the updated pipeline (TF-IDF + RoBERTa + XGBoost).
Tests on ISOT, PolitiFact to check if it generalizes.
Saves separate confusion matrix for each dataset.
"""
import torch
import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from tqdm import tqdm
import os
import sys

# Add src to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import apply_text_cleaning
from embedder import get_roberta_embedding, extract_embeddings
from feature_combiner import combine_features

def test_dataset(name, texts, true_labels, tfidf_vectorizer, xgb_model, print_samples=False):
    print(f"\n{'='*60}")
    print(f"  Testing on: {name} ({len(texts)} samples)")
    print(f"{'='*60}")

    # Create dummy dataframe to use apply_text_cleaning
    df = pd.DataFrame({"text": texts})
    df = apply_text_cleaning(df, text_column='text')
    
    clean_texts = df['clean_text'].tolist()

    # Get TF-IDF
    print("  Computing TF-IDF...")
    tfidf_matrix = tfidf_vectorizer.transform(clean_texts).toarray()

    # Get Embeddings
    print("  Extracting Embeddings...")
    embeddings = extract_embeddings(clean_texts)

    # Combine features
    print("  Combining Features...")
    X = combine_features(tfidf_matrix, embeddings, metadata=None)

    # Predict
    preds = xgb_model.predict(X)

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
    filename = f"cm_pipeline_{name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {name} (Pipeline)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Confusion matrix saved to {filename}")

    return acc

def main():
    print("=" * 60)
    print("  Cross-Dataset Generalizability Test (Main Pipeline)")
    print("  Model: TF-IDF + RoBERTa Embeddings + XGBoost")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    try:
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    except Exception as e:
        print("Failed to load tfidf_vectorizer.pkl. Did you run src/main.py?")
        return

    xgb_model = xgb.XGBClassifier()
    try:
        xgb_model.load_model("models/xgboost_model.json")
    except Exception as e:
        print("Failed to load xgboost_model.json. Did you run src/main.py?")
        return
        
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
        results["ISOT"] = test_dataset("ISOT (titles)", texts, labels, tfidf_vectorizer, xgb_model)
    except Exception as e:
        print(f"  ISOT error: {e}")

    # --- Test 3: PolitiFact Dataset ---
    try:
        pf_real = pd.read_csv("data/politifact_real.csv")
        pf_fake = pd.read_csv("data/politifact_fake.csv")
        pf_real_sample = pf_real.dropna(subset=["title"]).sample(min(500, len(pf_real)), random_state=42)
        pf_fake_sample = pf_fake.dropna(subset=["title"]).sample(min(500, len(pf_fake)), random_state=42)

        texts = pf_real_sample["title"].tolist() + pf_fake_sample["title"].tolist()
        labels = np.array([0]*len(pf_real_sample) + [1]*len(pf_fake_sample))
        results["PolitiFact"] = test_dataset("PolitiFact (titles)", texts, labels, tfidf_vectorizer, xgb_model)
    except Exception as e:
        print(f"  PolitiFact error: {e}")

    typed_texts = [
        # === FAKE (15) ===
        "Hillary Clinton caught running secret email server from underground bunker",
        "Trump secretly signed deal with Russia to rig upcoming elections",
        "Obama administration exposed for secretly spying on political opponents",
        "Democrats caught running illegal voter registration scheme in swing states",
        "Republican senators admit they rigged the election with foreign agents",
        "Drinking bleach cures COVID according to anonymous doctors",
        "Scientists confirm eating chocolate every day completely eliminates cancer cells",
        "Vaccines proven to cause autism by leaked government documents",
        "New study reveals that 5G towers are causing brain tumors in children",
        "Doctors expose secret cure for diabetes that big pharma is hiding from public",
        "Earth proven to be flat by leaked NASA internal documents",
        "Bill Gates implanting microchips through vaccines to control population",
        "Elon Musk confirms aliens are living inside Mars",
        "Government whistleblower reveals birds are actually surveillance drones",
        "Moon landing was filmed in Hollywood studio and astronauts confessed",
        # === REAL (15) ===
        "Senate approved new cybersecurity legislation to protect government agencies",
        "Supreme Court ruled in favor of expanding voting rights protections",
        "President signed executive order to address climate change and emissions",
        "Congress passed a bipartisan infrastructure bill after months of debate",
        "Federal Reserve raised interest rates by quarter percentage point citing inflation",
        "Unemployment rate dropped to four percent as economy shows recovery",
        "Stock market closed at record high as tech sector rallied on strong earnings",
        "India GDP growth rate reached seven percent in the latest fiscal quarter",
        "Global oil prices declined after OPEC announced increased production targets",
        "Amazon reported strong quarterly revenue driven by cloud computing growth",
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
    results["Hand-typed"] = test_dataset("Hand-typed Examples", typed_texts, typed_labels, tfidf_vectorizer, xgb_model, print_samples=True)

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
