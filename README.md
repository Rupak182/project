# Project Overview

This project is a machine learning classifier designed to detect fake news using a fine-tuned DistilBERT model paired with XGBoost.

## Setup Instructions

### Data Preparation
The datasets and trained models used in this project are very large and are strictly ignored by Git to prevent repository size issues.

1. Create a `data/` directory in the root of the project.
2. Place all your raw datasets `.csv` files inside the `data/` directory (e.g., `WELFake_Dataset.csv`, `ISOT_TRUE.csv`, `iSOT_FAKE.csv`).
3. Create a `models/` directory.
4. Trained models (like `.safetensors` or `.json` xgboost models) should be saved either in `models/` or in explicitly ignored model folders like `distilbert_finetuned/`.



Datasets:
https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news
https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset