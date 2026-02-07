from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Load the optimized model (approx 30M params vs 125M for RoBERTa)
# This model is specifically trained for semantic similarity and clustering
model = SentenceTransformer('all-MiniLM-L6-v2') 

def get_embedding(text):
    """
    Get embedding for a single text using SentenceTransformer.
    Returns a 1D numpy array (384-dim for MiniLM).
    """
    return model.encode(text, convert_to_numpy=True)

def extract_embeddings(texts):
    """
    Extract embeddings for a list of texts efficiently.
    Batch processing is handled automatically by the library.
    """
    # show_progress_bar=True gives a nice tqdm output automatically
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings