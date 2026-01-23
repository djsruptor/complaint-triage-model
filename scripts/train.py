import os
import time
import joblib
import argparse
import numpy as np

from src.processing import preprocess_data, split_data
from src.config import MODEL_DIR, MODEL_FILE

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def precision_at_k(y_true, y_score, k_frac=0.10):
    k = max(1, int(k_frac * len(y_true)))
    top_idx = np.argsort(y_score)[-k:]
    return float(y_true.iloc[top_idx].mean())

def train_model(input_path: str) -> None:
    print('Loading and preprocessing data...')
    df = preprocess_data(input_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    print('Training model...')
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            max_features=20000
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=3
        ))
    ])

    pipeline.fit(X_train, y_train)

    print('Evaluating model...')
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    val_p10 = precision_at_k(y_val, y_val_proba, 0.10)
    test_p10 = precision_at_k(y_test, y_test_proba, 0.10)

    print(f'Validation AUC: {val_auc:.4f} | Precision@10%: {val_p10:.4f}')
    print(f'Test AUC:       {test_auc:.4f} | Precision@10%: {test_p10:.4f}')

    print('Saving model...')
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f'model saved to {MODEL_FILE}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train complaint priority classifier')
    parser.add_argument(
        '--input',
        required=True,
        help='Path to raw CFPB complaints CSV downloaded from the CFPB portal'
    )
    args = parser.parse_args()

    start = time.time()
    train_model(args.input)
    print(f'Training time: {time.time() - start:.2f}s')