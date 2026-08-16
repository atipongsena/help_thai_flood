#!/usr/bin/env python
"""
Main Pipeline Runner for Thai Flood Relief NLP Pipeline

This script runs the complete pipeline:
1. Create sample data or scrape from URLs
2. Prepare and split dataset
3. Train classical models (BoW, TF-IDF, Word2Vec + SVM/NB)
4. Train BERT model
5. Evaluate and compare all models
6. Start API server
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import (
    DATA_DIR, MODELS_DIR, DB_PATH,
    TRAIN_CSV_PATH, TEST_CSV_PATH, LABELED_CSV_PATH
)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def step_1_prepare_data(use_sample: bool = True):
    """Step 1: Prepare data"""
    print_header("Step 1: Preparing Data")
    
    from training.prepare_dataset import (
        create_sample_data, export_db_to_csv, split_train_test
    )
    
    if use_sample:
        print("Creating sample data for testing...")
        create_sample_data()
    elif os.path.exists(DB_PATH):
        print("Exporting data from database...")
        export_db_to_csv()
    
    # Split into train/test
    print("Splitting into Train/Test...")
    split_train_test(use_auto_label=True)
    
    print("[OK] Data preparation complete!")


def step_2_train_classical(run_all: bool = False):
    """Step 2: Train classical models"""
    print_header("Step 2: Training Classical Models")
    
    from training.train_classical import (
        load_train_test_data, run_all_experiments, train_best_classical_model
    )
    
    # Check if data exists
    if not os.path.exists(TRAIN_CSV_PATH):
        print("Error: Training data not found. Run step 1 first.")
        return False
    
    X_train, y_train, X_test, y_test = load_train_test_data()
    
    if run_all:
        print("Running all 24 pipeline experiments...")
        run_all_experiments(X_train, y_train, X_test, y_test)
    else:
        print("Training best classical model (TF-IDF + SVM)...")
        train_best_classical_model(X_train, y_train, X_test, y_test)
    
    print("[OK] Classical model training complete!")
    return True


def step_3_train_bert():
    """Step 3: Train BERT model"""
    print_header("Step 3: Training BERT Model")
    
    try:
        import torch
        from training.train_bert import load_data_for_bert, train_bert_model
        
        # Check if data exists
        if not os.path.exists(TRAIN_CSV_PATH):
            print("Error: Training data not found. Run step 1 first.")
            return False
        
        train_df, test_df, label_names = load_data_for_bert()
        
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        train_bert_model(train_df, test_df, label_names=label_names)
        
        print("[OK] BERT training complete!")
        return True
        
    except ImportError:
        print("Warning: torch/transformers not installed. Skipping BERT training.")
        print("Install with: pip install torch transformers")
        return False


def step_4_evaluate():
    """Step 4: Evaluate all models"""
    print_header("Step 4: Evaluating Models")
    
    import pandas as pd
    
    # Check for classical results
    classical_results_path = os.path.join(MODELS_DIR, 'classical_results.csv')
    if os.path.exists(classical_results_path):
        print("\n--- Classical Model Results ---")
        df = pd.read_csv(classical_results_path)
        print(df.sort_values('f1', ascending=False).head(10).to_string())
    
    # Check for BERT results
    bert_results_path = os.path.join(MODELS_DIR, 'bert_flood_model', 'training_results.json')
    if os.path.exists(bert_results_path):
        import json
        print("\n--- BERT Model Results ---")
        with open(bert_results_path) as f:
            bert_results = json.load(f)
        for k, v in bert_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    print("\n[OK] Evaluation complete!")


def step_5_start_api():
    """Step 5: Start API server"""
    print_header("Step 5: Starting API Server")
    
    print("Starting FastAPI server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="Thai Flood Relief NLP Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all              # Run complete pipeline
  python run_pipeline.py --prepare          # Only prepare data
  python run_pipeline.py --train-classical  # Train classical models
  python run_pipeline.py --train-bert       # Train BERT model
  python run_pipeline.py --api              # Start API server
  python run_pipeline.py --all-experiments  # Run all 24 pipeline combinations
        """
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Run complete pipeline (prepare → train → evaluate)")
    parser.add_argument("--prepare", action="store_true",
                       help="Step 1: Prepare and split data")
    parser.add_argument("--train-classical", action="store_true",
                       help="Step 2: Train classical models")
    parser.add_argument("--train-bert", action="store_true",
                       help="Step 3: Train BERT model")
    parser.add_argument("--evaluate", action="store_true",
                       help="Step 4: Evaluate models")
    parser.add_argument("--api", action="store_true",
                       help="Step 5: Start API server")
    parser.add_argument("--all-experiments", action="store_true",
                       help="Run all 24 pipeline-representation combinations")
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use scraped data instead of sample data")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" Thai Flood Relief NLP Pipeline")
    print(" ระบบวิเคราะห์ข้อความขอความช่วยเหลือน้ำท่วม")
    print("=" * 60)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # If no arguments, show help
    if not any([args.all, args.prepare, args.train_classical, 
                args.train_bert, args.evaluate, args.api]):
        parser.print_help()
        print("\n\nQuick start with sample data:")
        print("  python run_pipeline.py --all")
        return
    
    # Run selected steps
    if args.all or args.prepare:
        step_1_prepare_data(use_sample=not args.use_real_data)
    
    if args.all or args.train_classical:
        step_2_train_classical(run_all=args.all_experiments)
    
    if args.all or args.train_bert:
        step_3_train_bert()
    
    if args.all or args.evaluate:
        step_4_evaluate()
    
    if args.api:
        step_5_start_api()
    
    print("\n" + "=" * 60)
    print(" Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

