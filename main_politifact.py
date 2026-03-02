import pandas as pd
import joblib
import os
import sys
import numpy as np

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import apply_text_cleaning
from tfidf_vectorizer import compute_tfidf
from embedder import extract_embeddings
from feature_combiner import combine_features
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
try:
    from matplotlib import pyplot as plt
    import seaborn as sns
except ImportError:
    pass

def main():
    print("Loading PolitiFact data...")
    try:
        pf_real = pd.read_csv("data/politifact_real.csv")
        pf_fake = pd.read_csv("data/politifact_fake.csv")
    except FileNotFoundError as e:
        print(f"File not found. Please ensure data/politifact_real.csv and data/politifact_fake.csv exist. Error: {e}")
        return

    # Drop rows where 'title' is missing
    pf_real = pf_real.dropna(subset=["title"]).copy()
    pf_fake = pf_fake.dropna(subset=["title"]).copy()
    
    # Add labels: 0 for Real, 1 for Fake
    pf_real['label'] = 0
    pf_fake['label'] = 1
    
    # Combine
    df = pd.concat([pf_real, pf_fake], ignore_index=True)
    
    print(f"Data loaded successfully. Total samples: {len(df)}")

    # Apply text cleaning on the 'title' column since PolitiFact dataset uses 'title'
    df = apply_text_cleaning(df, text_column='title')
    print("Text cleaned successfully.")
    
    tfidf_matrix, tfidf_vectorizer = compute_tfidf(df['clean_text'], max_features=5000)
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer_pf.pkl')
    print("TF-IDF matrix computed and vectorizer saved to models/tfidf_vectorizer_pf.pkl.")

    embeddings = extract_embeddings(df['clean_text'].tolist())
    print("Embeddings computed successfully.")

    X = combine_features(tfidf_matrix, embeddings, metadata=None)
    print("Features combined successfully.")

    y = df['label'].values
    print("Label extracted successfully.")

    print("Training XGBoost model...")
    # Train XGBoost locally to save with different name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    model.save_model('models/xgboost_model_pf.json')
    print("XGBoost model saved to models/xgboost_model_pf.json")
    
    y_pred = model.predict(X_test)
    print("-" * 60)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Real','Fake']))
    print("-" * 60)
    
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - PolitiFact Only Model")
        plt.tight_layout()
        plt.savefig('confusion_matrix_pf.png')
        plt.close()
        print("Confusion matrix saved to confusion_matrix_pf.png")
    except NameError:
        print("matplotlib/seaborn not installed, skipping confusion matrix plot.")

if __name__ == "__main__":
    main()
