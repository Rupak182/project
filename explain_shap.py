"""
KernelSHAP Explainability for DistilBERT + XGBoost Fake News Classifier
Runs on 5 FAKE + 5 REAL hand-typed samples.
Generates: waterfall plots per sample, aggregate importance plots.

Note: shap.plots.beeswarm/bar require fixed-length features. Since each text
has a different token count, we manually aggregate token-level SHAP values
across samples instead.
"""

from collections import defaultdict
import torch
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel

# ── Config ────────────────────────────────────────────────────────────────────
FINETUNED_DIR = "models/distilbert_finetuned"
MAX_LEN = 128
BATCH_SIZE = 16   # smaller batch for occlusion passes

# ── 5 FAKE + 5 REAL samples ───────────────────────────────────────────────────
SELECTED_TEXTS = [
    # FAKE (5)
    "Vaccines proven to cause autism by leaked government documents",
    "Bill Gates implanting microchips through vaccines to control population",
    "Moon landing was filmed in Hollywood studio and astronauts confessed",
    "New study reveals that 5G towers are causing brain tumors in children",
    "Scientists paid by government to fake data about global warming for political agenda",
    # REAL (5)
    "Senate approved new cybersecurity legislation to protect government agencies",
    "World Health Organization issued new guidelines for pandemic preparedness",
    "Federal Reserve raised interest rates by quarter percentage point citing inflation",
    "Researchers at MIT developed a new battery technology for electric vehicles",
    "European Union reached agreement on new regulations for artificial intelligence technology",
]
SELECTED_LABELS = ["FAKE"] * 5 + ["REAL"] * 5


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(FINETUNED_DIR)
    model = DistilBertModel.from_pretrained(FINETUNED_DIR)
    model.to(device).eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_finetuned.json")

    return tokenizer, model, xgb_model, device


# ── Embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(texts, tokenizer, model, device):
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = list(texts[i:i + BATCH_SIZE])
        enc = tokenizer(batch, max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            all_embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embs)


# ── KernelSHAP pipeline function ──────────────────────────────────────────────
def make_pipeline_fn(tokenizer, model, xgb_model, device):
    def pipeline_predict(texts):
        """Takes a list/array of strings, returns FAKE probability (shape: [n])."""
        texts = list(texts)
        embs = extract_embeddings(texts, tokenizer, model, device)
        return xgb_model.predict_proba(embs)[:, 1]  # P(FAKE)
    return pipeline_predict


# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_waterfall(shap_values, idx, label, text):
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    short = text[:60] + "..." if len(text) > 60 else text
    plt.title(f"[{label}] {short}", fontsize=9, pad=10)
    plt.tight_layout()
    fname = f"shap_waterfall_{idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def _aggregate_token_shap(shap_values, skip_special=True):
    """Aggregate mean |SHAP| per token across all samples."""
    SPECIAL = {'[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'}
    token_vals = defaultdict(list)
    for i in range(len(shap_values)):
        tokens = shap_values[i].data
        vals   = shap_values[i].values
        for tok, val in zip(tokens, vals):
            if skip_special and tok in SPECIAL:
                continue
            token_vals[tok].append(val)   # keep signed value
    # mean signed value + mean abs value for sorting
    mean_signed = {t: float(np.mean(v)) for t, v in token_vals.items()}
    mean_abs    = {t: float(np.mean(np.abs(v))) for t, v in token_vals.items()}
    return mean_signed, mean_abs


def save_summary_importance(shap_values):
    """Horizontal bar chart of top tokens sorted by mean absolute SHAP value (importance)."""
    mean_signed, mean_abs = _aggregate_token_shap(shap_values)
    top = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)[:20]
    tokens_sorted = [t for t, _ in top][::-1]   # bottom → top
    abs_vals      = [mean_abs[t] for t in tokens_sorted]
    signed_vals   = [mean_signed[t] for t in tokens_sorted]
    colors = ["#e74c3c" if v > 0 else "#2980b9" for v in signed_vals]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(tokens_sorted, abs_vals, color=colors)
    ax.set_xlabel("Mean |SHAP value|  (red = toward FAKE, blue = toward REAL)")
    ax.set_title("Global Token Importance (Mean Absolute SHAP)", fontsize=12)
    ax.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig("shap_summary_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary_importance.png")


def save_summary_direction(shap_values):
    """Vertical bar chart of top tokens showing mean signed SHAP value (direction)."""
    mean_signed, mean_abs = _aggregate_token_shap(shap_values)
    top = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)[:20]
    tokens_sorted = [t for t, _ in top]
    values = [mean_signed[t] for t in tokens_sorted]
    colors = ["#e74c3c" if v > 0 else "#2980b9" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tokens_sorted, values, color=colors)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Mean SHAP value  (↑ toward FAKE, ↓ toward REAL)")
    ax.set_title("Global Token Direction (Mean Signed SHAP)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("shap_summary_direction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary_direction.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  KernelSHAP Explainability — DistilBERT + XGBoost")
    print("=" * 60)

    tokenizer, model, xgb_model, device = load_models()
    pipeline_fn = make_pipeline_fn(tokenizer, model, xgb_model, device)

    print("\nSetting up KernelSHAP explainer...")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(pipeline_fn, masker)

    print(f"Running KernelSHAP on {len(SELECTED_TEXTS)} texts...")
    print("(This may take several minutes on GPU)\n")
    shap_values = explainer(SELECTED_TEXTS, batch_size=4)

    print("\nGenerating plots...")

    # Waterfall per text
    for i, (text, label) in enumerate(zip(SELECTED_TEXTS, SELECTED_LABELS)):
        save_waterfall(shap_values, i, label, text)

    # Aggregate importance (Horizontal)
    save_summary_importance(shap_values)

    # Aggregate direction (Vertical)
    save_summary_direction(shap_values)

    print("\n" + "=" * 60)
    print("  Done! Plots saved:")
    print("  shap_waterfall_0.png ... shap_waterfall_9.png")
    print("  shap_summary_importance.png")
    print("  shap_summary_direction.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
