"""
BERT Fine-tuning for Thai Flood Relief NLP Pipeline

Fine-tunes WangchanBERTa or other Thai BERT models for
binary classification of urgent vs non-urgent flood relief posts.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not installed. BERT training unavailable.")

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report
)

from utils.config import (
    TRAIN_CSV_PATH, TEST_CSV_PATH, BERT_MODEL_DIR,
    DEFAULT_BERT_MODEL, TrainingConfig, MODELS_DIR
)
from utils.risk_tags import parse_multi_label_field


# =============================================================================
# Data Loading and Preparation
# =============================================================================
def load_data_for_bert(
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH,
    label_column: str = "label",
    multi_label: bool = False,
    label_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load and prepare data for BERT training
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = train_df[['text', label_column]].dropna(subset=['text'])
    test_df = test_df[['text', label_column]].dropna(subset=['text'])
    
    label_names: List[str] = []
    
    if multi_label:
        train_labels = train_df[label_column].apply(parse_multi_label_field)
        test_labels = test_df[label_column].apply(parse_multi_label_field)
        if label_list:
            label_names = label_list
        else:
            label_set = set()
            for labels in pd.concat([train_labels, test_labels]):
                label_set.update(labels)
            label_names = sorted(label_set)
        if not label_names:
            raise ValueError("No labels found for multi-label training")
        
        def to_multi_hot(labels: List[str]) -> List[float]:
            return [1.0 if name in labels else 0.0 for name in label_names]
        
        train_df = pd.DataFrame({
            'text': train_df['text'],
            'labels': train_labels.apply(to_multi_hot)
        })
        test_df = pd.DataFrame({
            'text': test_df['text'],
            'labels': test_labels.apply(to_multi_hot)
        })
    else:
        train_df = train_df[['text', label_column]].dropna()
        test_df = test_df[['text', label_column]].dropna()
        if pd.api.types.is_numeric_dtype(train_df[label_column]):
            train_labels = train_df[label_column].astype(int)
            test_labels = test_df[label_column].astype(int)
            label_names = sorted(train_labels.unique().tolist())
        else:
            train_labels = train_df[label_column].astype(str)
            test_labels = test_df[label_column].astype(str)
            label_names = sorted(train_labels.unique().tolist())
            label_map = {name: idx for idx, name in enumerate(label_names)}
            train_labels = train_labels.map(label_map)
            test_labels = test_labels.map(label_map)
        train_df = pd.DataFrame({'text': train_df['text'], 'labels': train_labels})
        test_df = pd.DataFrame({'text': test_df['text'], 'labels': test_labels})
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    if multi_label:
        label_counter = {name: 0 for name in label_names}
        for label_vector in train_df['labels']:
            for idx, value in enumerate(label_vector):
                if value:
                    label_counter[label_names[idx]] += 1
        print(f"Label activation (train): {label_counter}")
    else:
        print(f"Label distribution (train): {pd.Series(train_df['labels']).value_counts().to_dict()}")
    
    return train_df, test_df, label_names


def create_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_length: int = 256
) -> Tuple[Dataset, Dataset]:
    """
    Create HuggingFace datasets for training
    """
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )
        tokens['labels'] = examples['labels']
        return tokens
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    keep_cols = {'input_ids', 'attention_mask', 'labels'}
    drop_train = [col for col in train_dataset.column_names if col not in keep_cols]
    drop_test = [col for col in test_dataset.column_names if col not in keep_cols]
    if drop_train:
        train_dataset = train_dataset.remove_columns(drop_train)
    if drop_test:
        test_dataset = test_dataset.remove_columns(drop_test)
    
    train_dataset.set_format(type='torch')
    test_dataset.set_format(type='torch')
    
    return train_dataset, test_dataset


# =============================================================================
# Metrics Computation
# =============================================================================
def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    logits, labels = eval_pred
    labels = np.array(labels)
    logits = np.array(logits)
    
    if labels.ndim > 1 and labels.shape[-1] > 1:
        probs = 1 / (1 + np.exp(-logits))
        predictions = (probs >= 0.5).astype(int)
        accuracy = (predictions == labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
    elif len(np.unique(labels)) > 2:
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
    else:
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# =============================================================================
# Training Function
# =============================================================================
def train_bert_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_BERT_MODEL,
    output_dir: str = BERT_MODEL_DIR,
    config: TrainingConfig = None,
    multi_label: bool = False,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fine-tune BERT model on flood relief classification
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        test_df: Test DataFrame
        model_name: HuggingFace model name
        output_dir: Directory to save model
        config: Training configuration
    
    Returns:
        Dict with training results and metrics
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and torch are required for BERT training")
    
    if config is None:
        config = TrainingConfig()
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")
    print(f"{'='*60}")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device_name} ({device})")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    num_labels = len(label_names) if multi_label else len(set(train_df['labels'].tolist()))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    if multi_label:
        model.config.problem_type = "multi_label_classification"
    model.to(device)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, test_dataset = create_datasets(
        train_df, test_df, tokenizer, config.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints'),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_strategy=config.eval_strategy,
        logging_steps=config.logging_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=2,
        report_to='none',  # Disable wandb/tensorboard
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=config.dataloader_num_workers,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    eval_result = trainer.evaluate()
    
    print(f"\n--- Final Results ---")
    print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"Precision: {eval_result['eval_precision']:.4f}")
    print(f"Recall: {eval_result['eval_recall']:.4f}")
    print(f"F1 Score: {eval_result['eval_f1']:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel saved to: {output_dir}")
    
    # Save results
    results = {
        'model_name': model_name,
        'accuracy': eval_result['eval_accuracy'],
        'precision': eval_result['eval_precision'],
        'recall': eval_result['eval_recall'],
        'f1': eval_result['eval_f1'],
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'label_names': label_names,
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


# =============================================================================
# Prediction Function
# =============================================================================
class BERTPredictor:
    """
    Class for making predictions with trained BERT model
    """
    
    def __init__(self, model_dir: str = BERT_MODEL_DIR):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        texts: list,
        batch_size: int = 16,
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on a list of texts
        
        Returns:
            Dict with 'labels' and optionally 'probabilities'
        """
        all_preds = []
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        
        result = {'labels': all_preds}
        if return_probs:
            result['probabilities'] = all_probs
        
        return result
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on a single text
        """
        result = self.predict([text])
        return {
            'label': result['labels'][0],
            'probability': result['probabilities'][0] if 'probabilities' in result else None
        }


# =============================================================================
# Detailed Evaluation
# =============================================================================
def evaluate_bert_detailed(
    model_dir: str = BERT_MODEL_DIR,
    test_path: str = TEST_CSV_PATH,
    label_column: str = "label",
    multi_label: bool = False,
):
    """
    Run detailed evaluation with classification report
    """
    # Load test data
    if multi_label:
        print("Detailed evaluation for multi-label tasks is not implemented yet.")
        return
    
    test_df = pd.read_csv(test_path)
    if label_column not in test_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {test_path}")
    texts = test_df['text'].astype(str).tolist()
    labels = pd.to_numeric(test_df[label_column], errors='ignore')
    if not np.issubdtype(labels.dtype, np.number):
        label_names = sorted(pd.Series(labels).astype(str).unique().tolist())
        label_map = {name: idx for idx, name in enumerate(label_names)}
        labels = labels.map(label_map).values
    else:
        labels = labels.astype(int).values
        label_names = None
    
    results_path = os.path.join(model_dir, 'training_results.json')
    if os.path.exists(results_path):
        with open(results_path, encoding='utf-8') as f:
            training_meta = json.load(f)
        saved_names = training_meta.get('label_names')
        if saved_names:
            label_names = saved_names
    
    # Load model and predict
    predictor = BERTPredictor(model_dir)
    results = predictor.predict(texts)
    predictions = results['labels']
    
    # Print detailed report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    unique_labels = sorted(np.unique(labels))
    if label_names and len(label_names) == len(unique_labels):
        target_names = [str(name) for name in label_names]
    else:
        target_names = [f"class_{idx}" for idx in unique_labels]
    
    print(classification_report(
        labels, predictions,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    ))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    print("\nConfusion Matrix:")
    header = "Predicted " + " ".join(f"{name:^8}" for name in target_names)
    print(header)
    for idx, row in enumerate(cm):
        row_str = " ".join(f"{val:^8}" for val in row)
        print(f"{target_names[idx]:<10} {row_str}")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BERT model")
    parser.add_argument("--model", type=str, default=DEFAULT_BERT_MODEL,
                       help="HuggingFace model name")
    parser.add_argument("--output", type=str, default=BERT_MODEL_DIR,
                       help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--evaluate", action="store_true",
                       help="Only evaluate existing model")
    parser.add_argument("--train-path", type=str, default=TRAIN_CSV_PATH)
    parser.add_argument("--test-path", type=str, default=TEST_CSV_PATH)
    parser.add_argument("--label-column", type=str, default="label",
                       help="Column to use as label (e.g., label, priority_label, risk_flags)")
    parser.add_argument("--multi-label", action="store_true",
                       help="Enable multi-label classification (labels separated by | or JSON)")
    parser.add_argument("--label-list", nargs="*", help="Explicit label order for multi-label tasks")
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers and torch are required.")
        print("Install with: pip install transformers torch")
        sys.exit(1)
    
    if args.evaluate:
        evaluate_bert_detailed(
            args.output,
            args.test_path,
            label_column=args.label_column,
            multi_label=args.multi_label,
        )
    else:
        # Check if data exists
        if not os.path.exists(args.train_path) or not os.path.exists(args.test_path):
            print("Training/Test data not found!")
            print("Please run prepare_dataset.py first.")
            sys.exit(1)
        
        # Load data
        train_df, test_df, label_names = load_data_for_bert(
            args.train_path,
            args.test_path,
            label_column=args.label_column,
            multi_label=args.multi_label,
            label_list=args.label_list,
        )
        
        # Create config
        config = TrainingConfig(
            bert_model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        
        # Train
        results = train_bert_model(
            train_df, test_df,
            model_name=args.model,
            output_dir=args.output,
            config=config,
            multi_label=args.multi_label,
            label_names=label_names,
        )
        
        if args.multi_label:
            print("\nSkipping detailed classification report for multi-label task (not yet supported in CLI).")
        else:
            print("\n" + "="*60)
            print("Detailed Evaluation")
            print("="*60)
            evaluate_bert_detailed(
                args.output,
                args.test_path,
                label_column=args.label_column,
                multi_label=args.multi_label,
            )

