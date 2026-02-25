import pandas as pd
import joblib
import os
from preprocessing import apply_text_cleaning, normalize_metadata, extract_url_metadata
from tfidf_vectorizer import compute_tfidf
from embedder import extract_embeddings
from feature_combiner import combine_features
from classifier import train_xgboost


def main():

    print("Loading data...")
    try:
        df = pd.read_csv("data/combined_news.csv")
        print("Data loaded successfully.")


        df = apply_text_cleaning(df,text_column='text')
        print("Text cleaned successfully.")

        # Metadata normalization skipped as we aren't using metadata currently
        # df = normalize_metadata(df,metadata_columns)
        # print("Metadata normalized successfully.")

        
        # Extract URL metadata
        df = extract_url_metadata(df, url_column='news_url')
        print("URL metadata extracted.")

        # Normalize metadata
        metadata_cols = ['url_length', 'is_https', 'url_special_chars', 'url_digits']
        df = normalize_metadata(df, metadata_cols)
        print("Metadata normalized successfully.")

        tfidf_matrix, tfidf_vectorizer = compute_tfidf(df['clean_text'],max_features=5000)
        os.makedirs('models', exist_ok=True)
        joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        print("TF-IDF matrix computed successfully and vectorizer saved to models/tfidf_vectorizer.pkl.")

        embeddings = extract_embeddings(df['clean_text'])
        print("Embeddings computed successfully.")

        # Combine features with metadata
        # Convert metadata DataFrame to numpy array
        metadata_features = df[metadata_cols].to_numpy()
        
        X = combine_features(tfidf_matrix,embeddings,metadata=metadata_features)
        print("Features combined successfully.")

        y = df['label']
        print("Label extracted successfully.")

        train_xgboost(X,y)
        print("Model trained successfully.")

    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return


if __name__ == "__main__":
    main()



