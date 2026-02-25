from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(corpus,max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(corpus).toarray()
    return tfidf_matrix, tfidf