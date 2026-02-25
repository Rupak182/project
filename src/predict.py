import joblib
import xgboost as xgb
import numpy as np
import os
from preprocessing import clean_text
from embedder import get_embedding
from feature_combiner import combine_features

def load_models():
    if not os.path.exists('models/tfidf_vectorizer.pkl') or not os.path.exists('models/xgboost_model.json'):
        print("Models not found in 'models/' directory. Please run 'python src/main.py' first to train and save the models.")
        return None, None
    
    print("Loading TF-IDF vectorizer...")
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_model.json')
    
    print("Models loaded successfully.")
    return tfidf_vectorizer, model

def predict_news(text, tfidf_vectorizer, model):
    # 1. Clean the text
    cleaned = clean_text(text)
    
    # 2. Extract TF-IDF feature
    tfidf_feature = tfidf_vectorizer.transform([cleaned]).toarray()
    
    # 3. Extract embedding feature
    embedding_feature = get_embedding(cleaned).reshape(1, -1)
    
    # 4. Combine features
    X = combine_features(tfidf_feature, embedding_feature, metadata=None)
    
    # 5. Make prediction
    pred = model.predict(X)
    
    # Assume 0 is Fake and 1 is Real based on target_names in classifier.py
    return "Fake" if pred[0] == 0 else "Real"

def main():
    tfidf_vectorizer, model = load_models()
    
    if tfidf_vectorizer is None or model is None:
        return

    print("\n--- Fake News Detection System ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            text = input("Enter news text: ")
            if text.lower() in ['exit', 'quit']:
                break
            
            if not text.strip():
                print("Empty input. Please try again.")
                continue
                
            prediction = predict_news(text, tfidf_vectorizer, model)
            print(f"\n=> Predicted Label: ** {prediction.upper()} **\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
