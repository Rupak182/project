import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.to(device)
model.eval()

def get_roberta_embedding(text):
    """Get mean pooled RoBERTa embedding for a single text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def extract_embeddings(texts):
    """Extract embeddings for a list of texts."""
    return np.vstack([get_roberta_embedding(text) for text in tqdm(texts, desc="Extracting RoBERTa embeddings")])