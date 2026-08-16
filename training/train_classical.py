"""
Classical Model Training for Thai Flood Relief NLP Pipeline

Trains and evaluates classical ML models:
- Naive Bayes with BoW
- SVM with TF-IDF
- SVM with Word2Vec

Compares all 24 pipeline-representation combinations.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    TRAIN_CSV_PATH, TEST_CSV_PATH, MODELS_DIR,
    TFIDF_VEC_PATH, SVM_MODEL_PATH, W2V_MODEL_PATH,
    PREPROCESSING_PIPELINES, PreprocessConfig, TrainingConfig
)
from utils.features import (
    build_bow_vectorizer, build_tfidf_vectorizer,
    Word2VecFeatures, save_sklearn_vectorizer, save_word2vec_model
)


# =============================================================================
# Data Loading
# =============================================================================
def load_train_test_data(
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """
    Load training and test data
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract text and labels
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['label'].astype(int).values
    
    X_test = test_df['text'].astype(str).tolist()
    y_test = test_df['label'].astype(int).values
    
    print(f"Loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Train labels: {np.bincount(y_train)}")
    print(f"Test labels: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test


# =============================================================================
# Model Training Functions
# =============================================================================
def train_bow_nb(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig
) -> Dict[str, Any]:
    """
    Train Naive Bayes with Bag-of-Words
    """
    # Build vectorizer and transform
    vectorizer = build_bow_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'bow_nb',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'vectorizer': vectorizer,
        'vocab_size': len(vectorizer.vocabulary_),
    }
    
    return results


def train_tfidf_svm(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig
) -> Dict[str, Any]:
    """
    Train Linear SVM with TF-IDF
    """
    # Build vectorizer and transform
    vectorizer = build_tfidf_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'tfidf_svm',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'vectorizer': vectorizer,
        'vocab_size': len(vectorizer.vocabulary_),
    }
    
    return results


def train_tfidf_lr(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig
) -> Dict[str, Any]:
    """
    Train Logistic Regression with TF-IDF
    """
    # Build vectorizer and transform
    vectorizer = build_tfidf_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'tfidf_lr',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'vectorizer': vectorizer,
        'vocab_size': len(vectorizer.vocabulary_),
    }
    
    return results


def train_w2v_svm(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig,
    vector_size: int = 200
) -> Dict[str, Any]:
    """
    Train Linear SVM with Word2Vec embeddings
    """
    # Build Word2Vec and transform
    w2v = Word2VecFeatures(vector_size=vector_size, config=config)
    X_train_vec = w2v.fit_transform(X_train)
    X_test_vec = w2v.transform(X_test)
    
    # Train model
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'w2v_svm',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'w2v_model': w2v,
        'vector_size': vector_size,
    }
    
    return results


# =============================================================================
# Run All Pipeline Combinations
# =============================================================================
def run_all_experiments(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    save_best: bool = True
) -> pd.DataFrame:
    """
    Run all 24 pipeline-representation combinations
    
    6 pipelines × 4 representations = 24 experiments
    """
    all_results = []
    best_f1 = 0
    best_result = None
    
    representations = ['bow_nb', 'tfidf_svm', 'tfidf_lr', 'w2v_svm']
    
    print("\n" + "="*60)
    print("Running All Pipeline-Representation Combinations")
    print("="*60)
    
    for pipe_name, config in PREPROCESSING_PIPELINES.items():
        print(f"\n--- {pipe_name}: {config} ---")
        
        for rep in representations:
            try:
                print(f"  Training {rep}...", end=" ")
                
                if rep == 'bow_nb':
                    result = train_bow_nb(X_train, y_train, X_test, y_test, config)
                elif rep == 'tfidf_svm':
                    result = train_tfidf_svm(X_train, y_train, X_test, y_test, config)
                elif rep == 'tfidf_lr':
                    result = train_tfidf_lr(X_train, y_train, X_test, y_test, config)
                elif rep == 'w2v_svm':
                    result = train_w2v_svm(X_train, y_train, X_test, y_test, config)
                
                result['pipeline_name'] = pipe_name
                
                print(f"F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}")
                
                # Track best result
                if result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_result = result.copy()
                
                # Store results (without model objects for DataFrame)
                all_results.append({
                    'pipeline': pipe_name,
                    'config': str(config),
                    'method': result['method'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1'],
                })
                
            except Exception as e:
                print(f"Error: {e}")
                all_results.append({
                    'pipeline': pipe_name,
                    'config': str(config),
                    'method': rep,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'error': str(e),
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_path = os.path.join(MODELS_DIR, 'classical_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(results_df.sort_values('f1', ascending=False).head(10).to_string())
    
    # Save best model
    if save_best and best_result:
        print(f"\n--- Best Model: {best_result['method']} ({best_result['pipeline_name']}) ---")
        print(f"F1: {best_result['f1']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        
        # Save best vectorizer and model
        if 'vectorizer' in best_result:
            save_sklearn_vectorizer(best_result['vectorizer'], TFIDF_VEC_PATH)
            print(f"Saved vectorizer to: {TFIDF_VEC_PATH}")
        
        if 'model' in best_result:
            joblib.dump(best_result['model'], SVM_MODEL_PATH)
            print(f"Saved model to: {SVM_MODEL_PATH}")
        
        if 'w2v_model' in best_result:
            save_word2vec_model(best_result['w2v_model'], W2V_MODEL_PATH)
            print(f"Saved Word2Vec to: {W2V_MODEL_PATH}")
    
    return results_df


# =============================================================================
# Quick Training (Single Best Config)
# =============================================================================
def train_best_classical_model(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train the recommended classical model (TF-IDF + SVM with Pipeline 4)
    """
    # Use Pipeline 4: clean + normalize + stopwords
    config = PREPROCESSING_PIPELINES['pipeline_4']
    
    print("Training TF-IDF + SVM with Pipeline 4...")
    result = train_tfidf_svm(X_train, y_train, X_test, y_test, config)
    
    print(f"\n--- Results ---")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print(f"Vocabulary Size: {result['vocab_size']}")
    
    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_sklearn_vectorizer(result['vectorizer'], TFIDF_VEC_PATH)
    joblib.dump(result['model'], SVM_MODEL_PATH)
    
    print(f"\nSaved:")
    print(f"  Vectorizer: {TFIDF_VEC_PATH}")
    print(f"  Model: {SVM_MODEL_PATH}")
    
    return result


# =============================================================================
# Prediction Function
# =============================================================================
def predict_with_classical(
    texts: List[str],
    vectorizer_path: str = TFIDF_VEC_PATH,
    model_path: str = SVM_MODEL_PATH
) -> np.ndarray:
    """
    Make predictions using saved classical model
    """
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    
    return predictions


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train classical ML models")
    parser.add_argument("--all", action="store_true",
                       help="Run all 24 pipeline experiments")
    parser.add_argument("--quick", action="store_true",
                       help="Train single best model quickly")
    parser.add_argument("--train-path", type=str, default=TRAIN_CSV_PATH,
                       help="Path to training data")
    parser.add_argument("--test-path", type=str, default=TEST_CSV_PATH,
                       help="Path to test data")
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.train_path) or not os.path.exists(args.test_path):
        print("Training/Test data not found!")
        print("Please run prepare_dataset.py first to create the data.")
        print(f"  python training/prepare_dataset.py --sample")
        sys.exit(1)
    
    # Load data
    X_train, y_train, X_test, y_test = load_train_test_data(
        args.train_path, args.test_path
    )
    
    if args.all:
        run_all_experiments(X_train, y_train, X_test, y_test)
    else:
        # Default: train best model
        train_best_classical_model(X_train, y_train, X_test, y_test)

