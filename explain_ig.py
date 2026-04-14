"""
explain_ig.py — Integrated Gradients Token Attribution
========================================================
Uses Captum's IntegratedGradients to compute token importance
directly from DistilBert's fine-tuned classification head.

Key difference from SHAP:
  - No token masking → no OOD inputs → no saturation problem
  - Baseline = [PAD] token embedding (all zeros effectively)
  - Gradients flow through DistilBERT's classification head
  - Both FAKE and REAL samples get meaningful attribution values

Note: This explains DistilBertForSequenceClassification directly.
XGBoost is not involved here — we use the fine-tuned classification
head that was trained alongside the DistilBERT backbone.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import IntegratedGradients, TokenReferenceBase, visualization

FINETUNED_DIR = "models/distilbert_finetuned"
MAX_LEN = 128
N_STEPS = 50       # interpolation steps (higher = more accurate, slower)

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


# ── Model wrapper ─────────────────────────────────────────────────────────────
class DistilBertIGWrapper(torch.nn.Module):
    """
    Wraps DistilBertForSequenceClassification so that IntegratedGradients
    can perturb the INPUT EMBEDDINGS directly rather than token IDs.

    IG requires a differentiable function. Token IDs are discrete (not
    differentiable), so we must work in embedding space instead.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_embeds, attention_mask):
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )
        # Return P(FAKE) — class 1 probability via softmax
        return torch.softmax(outputs.logits, dim=-1)[:, 1]


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(FINETUNED_DIR)

    # Load the CLASSIFICATION model (with head), not just the base encoder
    model = DistilBertForSequenceClassification.from_pretrained(
        FINETUNED_DIR, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device).eval()
    return tokenizer, model, device


# ── Integrated Gradients for a single text ───────────────────────────────────
def compute_ig_attributions(text, tokenizer, model, wrapper, ig, device):
    """
    Returns:
        tokens     : list of string tokens (without [CLS]/[SEP]/[PAD])
        scores     : numpy array of attribution scores per token
        pred_prob  : float, P(FAKE) for this text
    """
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Get actual input embeddings
    input_embeds = model.distilbert.embeddings(input_ids)  # (1, seq_len, 768)

    # Baseline: all [PAD] token embeddings
    pad_id = tokenizer.pad_token_id
    pad_ids = torch.full_like(input_ids, pad_id)
    baseline_embeds = model.distilbert.embeddings(pad_ids)

    # Run IG
    attributions, delta = ig.attribute(
        input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(attention_mask,),
        n_steps=N_STEPS,
        return_convergence_delta=True
    )

    # Summarise: L2 norm of attribution across embedding dim → one score per token
    attr_scores = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    # Preserve direction: positive if attribution pushed toward FAKE
    attr_sum = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()
    signed_scores = np.sign(attr_sum) * attr_scores

    # Get prediction
    with torch.no_grad():
        pred_prob = wrapper(input_embeds, attention_mask).item()

    # Decode tokens, remove special tokens
    token_ids = input_ids.squeeze(0).tolist()
    tokens_all = tokenizer.convert_ids_to_tokens(token_ids)
    special = {tokenizer.cls_token, tokenizer.sep_token,
               tokenizer.pad_token, '[CLS]', '[SEP]', '[PAD]'}

    tokens, scores = [], []
    for tok, sc in zip(tokens_all, signed_scores):
        if tok not in special and tok != tokenizer.pad_token:
            tokens.append(tok)
            scores.append(sc)

    return tokens, np.array(scores), pred_prob, float(delta.mean().abs())


# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_ig_waterfall(tokens, scores, pred_prob, label, text, idx):
    """Horizontal bar chart showing token attributions (positive=FAKE, negative=REAL)."""
    # Sort by absolute score, take top 15
    order = np.argsort(np.abs(scores))[::-1][:15]
    top_tokens = [tokens[i] for i in order][::-1]   # bottom→top
    top_scores = [scores[i] for i in order][::-1]
    colors = ["#e74c3c" if s > 0 else "#2980b9" for s in top_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_tokens, top_scores, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Attribution Score  (red=toward FAKE, blue=toward REAL)")

    short = text[:70] + "..." if len(text) > 70 else text
    ax.set_title(
        f"[{label}]  P(FAKE)={pred_prob:.4f}\n"
        f"Integrated Gradients — Token Attribution\n{short}",
        fontsize=9, pad=10
    )

    # Annotate bar values
    for bar, val in zip(bars, top_scores):
        xpos = val + 0.001 if val >= 0 else val - 0.001
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    fname = f"shap_ig_waterfall_{idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}  |  P(FAKE)={pred_prob:.4f}")


def save_ig_summary(all_tokens, all_scores):
    """Global aggregate: mean absolute + signed attribution per word."""
    from collections import defaultdict
    token_map = defaultdict(list)
    for tokens, scores in zip(all_tokens, all_scores):
        for tok, sc in zip(tokens, scores):
            token_map[tok].append(sc)

    mean_abs    = {t: float(np.mean(np.abs(v))) for t, v in token_map.items()}
    mean_signed = {t: float(np.mean(v))          for t, v in token_map.items()}

    top = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)[:20]
    toks = [t for t, _ in top][::-1]
    abs_vals = [mean_abs[t] for t in toks]
    colors   = ["#e74c3c" if mean_signed[t] > 0 else "#2980b9" for t in toks]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(toks, abs_vals, color=colors)
    ax.set_xlabel("Mean |Attribution|  (red=FAKE signal, blue=REAL signal)")
    ax.set_title("Global Token Importance — Integrated Gradients", fontsize=12)
    ax.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig("shap_ig_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_ig_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Integrated Gradients — Token Attribution (Option B)")
    print("  Model: DistilBertForSequenceClassification (fine-tuned)")
    print("  No masking, no OOD inputs, no XGBoost involved")
    print("=" * 65)

    tokenizer, model, device = load_models()
    wrapper = DistilBertIGWrapper(model)

    # Set up IG — target=None because our wrapper already returns a scalar
    ig = IntegratedGradients(wrapper)

    print(f"\nRunning IG on {len(SELECTED_TEXTS)} texts ({N_STEPS} steps each)...\n")

    all_tokens, all_scores = [], []

    for i, (text, label) in enumerate(zip(SELECTED_TEXTS, SELECTED_LABELS)):
        print(f"  [{i+1}/{len(SELECTED_TEXTS)}] [{label}] {text[:60]}...")
        tokens, scores, pred_prob, delta = compute_ig_attributions(
            text, tokenizer, model, wrapper, ig, device
        )
        print(f"         convergence delta (lower=better): {delta:.5f}")
        save_ig_waterfall(tokens, scores, pred_prob, label, text, i)
        all_tokens.append(tokens)
        all_scores.append(scores)

    print("\nSaving global summary...")
    save_ig_summary(all_tokens, all_scores)

    print("\n" + "=" * 65)
    print("  Done! Outputs:")
    print("  shap_ig_waterfall_0.png ... shap_ig_waterfall_9.png")
    print("  shap_ig_summary.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
