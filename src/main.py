import pandas as pd
from preprocessing import apply_text_cleaning,normalize_metadata
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

        
        tfidf_matrix = compute_tfidf(df['clean_text'],max_features=5000)
        print("TF-IDF matrix computed successfully.")

        embeddings = extract_embeddings(df['clean_text'])
        print("Embeddings computed successfully.")

        X = combine_features(tfidf_matrix,embeddings,metadata=None)
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



