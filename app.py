import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from xgboost import XGBClassifier
from transformers import DistilBertModel

from explain_ig import (
    load_models,
    DistilBertIGWrapper,
    compute_ig_attributions,
    MAX_LEN,
    N_STEPS
)
from captum.attr import IntegratedGradients

st.set_page_config(layout="wide", page_title="Fake News Expainability App")

# 1. Model Loading (Cached)
@st.cache_resource
def load_all_models():
    """Loads DistilBertForSequenceClassification + IG wrapper (for explanations)."""
    tokenizer, model, device = load_models()
    wrapper = DistilBertIGWrapper(model)
    ig = IntegratedGradients(wrapper)
    return tokenizer, model, device, wrapper, ig


@st.cache_resource
def load_xgb_pipeline():
    """Loads DistilBertModel (base, for CLS embeddings) + trained XGBoost model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from explain_ig import FINETUNED_DIR
    base_model = DistilBertModel.from_pretrained(FINETUNED_DIR)
    base_model.to(device).eval()
    xgb_model = XGBClassifier()
    xgb_model.load_model("models/xgboost_finetuned.json")
    return base_model, xgb_model, device

# Text highlighting function
def get_html_highlighted_text(tokens, scores):
    html = "<div style='line-height: 1.6; font-size: 16px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: white; color: black; max-width: 100%; white-space: pre-wrap; word-wrap: break-word;'>"
    
    max_score = np.max(np.abs(scores)) if len(scores) > 0 else 1.0
    if max_score == 0:
        max_score = 1.0
    
    for tok, score in zip(tokens, scores):
        # Normalize score to [0, 1] for alpha
        alpha = abs(score) / max_score
        
        # Clip alpha just to be safe
        alpha = min(max(alpha, 0), 1)
        
        # Red = FAKE (score > 0), Blue = REAL (score < 0)
        # Using rgba for background color
        if score > 0:
            color = f"rgba(255, 99, 71, {alpha})" # Tomato Red
        elif score < 0:
            color = f"rgba(100, 149, 237, {alpha})" # Cornflower Blue
        else:
            color = "transparent"
            
        # Clean up subword tokens (##)
        display_tok = tok.replace('##', '') if tok.startswith('##') else f" {tok}"
        
        # Add to html
        html += f"<span style='background-color: {color}; padding: 0.1em; border-radius: 3px;'>{display_tok}</span>"
        
    html += "</div>"
    return html

# Matplotlib Waterfall
def plot_waterfall(tokens, scores, pred_prob, text):
    order = np.argsort(np.abs(scores))[::-1][:20]
    top_tokens = [str(tokens[i])[:20] for i in order][::-1]   # bottom→top
    top_scores = [scores[i] for i in order][::-1]
    colors = ["#e74c3c" if s > 0 else "#2980b9" for s in top_scores]

    fig, ax = plt.subplots(figsize=(6, 8))
    bars = ax.barh(top_tokens, top_scores, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Attribution Score (red=toward FAKE, blue=toward REAL)")

    short = text[:70] + "..." if len(text) > 70 else text
    ax.set_title(
        f"P(FAKE)={pred_prob:.4f}\nIntegrated Gradients — Token Attribution\n{short}",
        fontsize=10, pad=10
    )

    max_val = np.max(np.abs(top_scores)) if len(top_scores) > 0 else 1.0
    if max_val == 0: max_val = 1.0
    offset = max_val * 0.05

    # Annotate bar values
    for bar, val in zip(bars, top_scores):
        xpos = val + offset if val >= 0 else val - offset
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=8)

    # Use layout adjustment that won't throw tight_layout constraint errors
    try:
        fig.tight_layout()
    except UserWarning:
        pass
    
    return fig

# Batch prediction using DistilBERT [CLS] embeddings → XGBoost
@torch.no_grad()
def batch_predict(texts, tokenizer, base_model, xgb_model, device):
    """Extracts [CLS] embeddings from base DistilBertModel, then predicts with XGBoost."""
    all_embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = base_model(input_ids, attention_mask=attention_mask)
        # [CLS] token hidden state — shape: (batch, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    import numpy as np
    all_embeddings = np.vstack(all_embeddings)
    # XGBoost predict_proba returns [[P(REAL), P(FAKE)], ...]
    probs_fake = xgb_model.predict_proba(all_embeddings)[:, 1].tolist()
    return probs_fake

def main():
    st.title("Fake News Expainability App")
    st.markdown("Upload your fake/real news data, perform predictions, and select samples to investigate why the model made a certain decision through token attribution.")

    with st.spinner("Loading models..."):
        try:
            tokenizer, model, device, wrapper, ig = load_all_models()
        except Exception as e:
            st.error(f"Error loading IG models. Have you trained them? Error: {e}")
            return

    with st.spinner("Loading XGBoost model..."):
        try:
            base_model, xgb_model, xgb_device = load_xgb_pipeline()
        except Exception as e:
            st.error(f"Error loading XGBoost model. Have you run main7.py? Error: {e}")
            return

    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or TXT File", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                lines = uploaded_file.getvalue().decode("utf-8").splitlines()
                df = pd.DataFrame({"text": lines})
                
            st.sidebar.success(f"Loaded {len(df)} samples.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # Find text column
        text_cols = [col for col in df.columns if col.lower() in ["text", "title", "content", "news"]]
        if text_cols:
            text_col = text_cols[0]
        else:
            text_col = df.columns[0]
            st.warning(f"No standard 'text' column found. Defaulting to first column: '{text_col}'")

        # Session state for predictions
        if "df_preds" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
            texts = df[text_col].astype(str).tolist()
            with st.spinner("Running XGBoost batch predictions..."):
                try:
                    probs = batch_predict(texts, tokenizer, base_model, xgb_model, xgb_device)
                    df["P(FAKE)"] = probs
                    df["P(REAL)"] = 1.0 - df["P(FAKE)"]
                    df["Prediction"] = df["P(FAKE)"].apply(lambda p: "FAKE" if p >= 0.5 else "REAL")
                    st.session_state.df_preds = df
                    st.session_state.last_uploaded = uploaded_file.name
                except Exception as e:
                    st.error(f"Error in XGBoost batch predictions: {e}")
                    return
        
        df_display = st.session_state.df_preds

        st.header("2. Data & Predictions")
        st.markdown("Here is the raw data sorted by prediction probability. Select a **Row Index** below for explanation.")
        st.dataframe(df_display, width=800)

        st.header("3. Explanations (Integrated Gradients)")
        row_idx = st.selectbox("Select sample index to explain:", df_display.index)

        if row_idx is not None:
            sample_text = df_display.loc[row_idx, text_col]
            sample_prob = df_display.loc[row_idx, "P(FAKE)"]
            
            st.write(f"**Selected Text:** {sample_text}")
            st.write(f"**Predicted P(FAKE):** {sample_prob:.4f}")

            with st.spinner("Computing Integrated Gradients... (this takes a moment)"):
                try:
                    tokens, scores, pred_prob, delta = compute_ig_attributions(
                        str(sample_text), tokenizer, model, wrapper, ig, device
                    )
                except Exception as e:
                    st.error(f"Error computing IG: {e}")
                    return

            st.markdown(f"*Convergence Delta: {delta:.5f} (lower is better)*")
            
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Attribution Visualization")
                # Need to use a lambda or pass text differently
                fig = plot_waterfall(tokens, scores, pred_prob, str(sample_text))
                st.pyplot(fig)  # Avoiding `use_container_width` for now

            with col2:
                st.subheader("Highlighted Text")
                st.markdown("> **Red** = Pushed prediction toward FAKE. **Blue** = Pushed prediction toward REAL.")
                html_code = get_html_highlighted_text(tokens, scores)
                st.markdown(html_code, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
