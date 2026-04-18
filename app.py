import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    tokenizer, model, device = load_models()
    wrapper = DistilBertIGWrapper(model)
    ig = IntegratedGradients(wrapper)
    return tokenizer, model, device, wrapper, ig

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
    top_tokens = [tokens[i] for i in order][::-1]   # bottom→top
    top_scores = [scores[i] for i in order][::-1]
    colors = ["#e74c3c" if s > 0 else "#2980b9" for s in top_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_tokens, top_scores, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Attribution Score (red=toward FAKE, blue=toward REAL)")

    short = text[:70] + "..." if len(text) > 70 else text
    ax.set_title(
        f"P(FAKE)={pred_prob:.4f}\nIntegrated Gradients — Token Attribution\n{short}",
        fontsize=10, pad=10
    )

    # Annotate bar values
    for bar, val in zip(bars, top_scores):
        xpos = val + 0.001 if val >= 0 else val - 0.001
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    return fig

# Basic batch prediction without IG for the table using pure torch
@torch.no_grad()
def batch_predict(texts, tokenizer, wrapper, device):
    probs = []
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
        input_embeds = wrapper.model.distilbert.embeddings(input_ids)
        
        preds = wrapper(input_embeds, attention_mask).cpu().tolist()
        probs.extend(preds)
    return probs

def main():
    st.title("Fake News Expainability App")
    st.markdown("Upload your fake/real news data, perform predictions, and select samples to investigate why the model made a certain decision through token attribution.")

    with st.spinner("Loading models..."):
        try:
            tokenizer, model, device, wrapper, ig = load_all_models()
        except Exception as e:
            st.error(f"Error loading models. Have you trained them? Error: {e}")
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
            with st.spinner("Running batch predictions..."):
                try:
                    probs = batch_predict(texts, tokenizer, wrapper, device)
                    df["P(FAKE)"] = probs
                    df["P(REAL)"] = 1.0 - df["P(FAKE)"]
                    st.session_state.df_preds = df
                    st.session_state.last_uploaded = uploaded_file.name
                except Exception as e:
                    st.error(f"Error in batch predictions: {e}")
                    return
        
        df_display = st.session_state.df_preds

        st.header("2. Data & Predictions")
        st.markdown("Here is the raw data sorted by prediction probability. Select a **Row Index** below for explanation.")
        st.dataframe(df_display, use_container_width=True)

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
                st.pyplot(fig)

            with col2:
                st.subheader("Highlighted Text")
                st.markdown("> **Red** = Pushed prediction toward FAKE. **Blue** = Pushed prediction toward REAL.")
                html_code = get_html_highlighted_text(tokens, scores)
                st.components.v1.html(html_code, height=400, scrolling=True)

if __name__ == "__main__":
    main()
