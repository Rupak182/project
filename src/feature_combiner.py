import numpy as np

def combine_features(tfidf_matrix, roberta_embeddings, metadata=None):
    """Concatenate all feature sets into one final matrix."""
    if metadata is not None and len(metadata) > 0:
        return np.hstack((tfidf_matrix, roberta_embeddings, metadata))
    return np.hstack((tfidf_matrix, roberta_embeddings))