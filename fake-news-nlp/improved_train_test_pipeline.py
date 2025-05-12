import os
import torch
import argparse
import pandas as pd
from transformers import (
    BertForSequenceClassification, 
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    BertTokenizer, 
    DistilBertTokenizer,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import local modules
from models.trainer import NewsDataset, train_model, evaluate_model
from utils.dataset_loader import load_fnc1_dataset, load_liar_dataset
from utils.preprocessing import clean_text, extract_features
from improved_config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate fake news detection models')
    parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'], 
                        default=Config.DEFAULT_MODEL, help='Model architecture to use')
    parser.add_argument('--dataset', type=str, choices=['fnc1', 'liar', 'both'], 
                        default='both', help='Dataset to train/evaluate on')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, 
                        help='Learning rate')
    parser.add_argument('--sample_frac', type=float, default=Config.SAMPLE_FRAC, 
                        help='Fraction of data to use (0-1)')
    parser.add_argument('--cross_val', action='store_true', 
                        help='Perform cross-validation')
    parser.add_argument('--save_model', action='store_true', 
                        help='Save the trained model')
    
    return parser.parse_args()

def get_model_and_tokenizer(model_type, num_labels):
    """Initialize model and tokenizer based on type"""
    model_config = Config.MODELS[model_type]
    model_name = model_config["name"]
    
    if model_config["type"] == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
    elif model_config["type"] == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            dropout=0.2
        )
    elif model_config["type"] == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
    
    return tokenizer, model

def train_fnc1(args, device):
    """Train and evaluate on FNC-1 dataset"""
    print("\n==== Training on FNC-1 dataset ====")
    
    # Load data
    fnc_train = load_fnc1_dataset(split="train", sample_frac=args.sample_frac)
    fnc_val = load_fnc1_dataset(split="val", sample_frac=args.sample_frac)
    
    # Clean text with advanced preprocessing if enabled
    if Config.USE_ADVANCED_PREPROCESSING:
        fnc_train["headline"] = fnc_train["headline"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        fnc_train["body"] = fnc_train["body"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        fnc_val["headline"] = fnc_val["headline"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        fnc_val["body"] = fnc_val["body"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
    
    # Initialize model
    tokenizer, model = get_model_and_tokenizer(args.model, Config.FNC1_LABELS)
    model.to(device)
    
    # Prepare datasets
    train_dataset = NewsDataset(
        fnc_train["headline"].tolist(),
        fnc_train["body"].tolist(),
        fnc_train["label"].tolist(),
        tokenizer,
        Config.FNC1_MAX_LEN
    )
    
    val_dataset = NewsDataset(
        fnc_val["headline"].tolist(),
        fnc_val["body"].tolist(),
        fnc_val["label"].tolist(),
        tokenizer,
        Config.FNC1_MAX_LEN
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = None
    if Config.USE_SCHEDULER:
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * Config.WARMUP_STEPS)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    
    # Create save directory if needed
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, f"fnc1_{args.model}_best.pt")
    
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        epochs=args.epochs,
        patience=Config.EARLY_STOPPING_PATIENCE,
        scheduler=scheduler,
        model_save_path=model_save_path
    )
    
    return trained_model, tokenizer

def train_liar(args, device):
    """Train and evaluate on LIAR dataset"""
    print("\n==== Training on LIAR dataset ====")
    
    # Load data
    liar_train = load_liar_dataset(split="train")
    liar_val = load_liar_dataset(split="valid")
    liar_test = load_liar_dataset(split="test")
    
    # Sample data if needed
    if args.sample_frac < 1.0:
        liar_train = liar_train.sample(frac=args.sample_frac, random_state=42)
    
    # Clean text with advanced preprocessing if enabled
    if Config.USE_ADVANCED_PREPROCESSING:
        liar_train["statement"] = liar_train["statement"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        liar_val["statement"] = liar_val["statement"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        liar_test["statement"] = liar_test["statement"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
    
    # Initialize model
    tokenizer, model = get_model_and_tokenizer(args.model, Config.LIAR_LABELS)
    model.to(device)
    
    # Prepare datasets
    train_dataset = NewsDataset(
        liar_train["statement"].tolist(),
        labels=liar_train["label"].tolist(),
        tokenizer=tokenizer,
        max_len=Config.LIAR_MAX_LEN
    )
    
    val_dataset = NewsDataset(
        liar_val["statement"].tolist(),
        labels=liar_val["label"].tolist(),
        tokenizer=tokenizer,
        max_len=Config.LIAR_MAX_LEN
    )
    
    test_dataset = NewsDataset(
        liar_test["statement"].tolist(),
        labels=liar_test["label"].tolist(),
        tokenizer=tokenizer,
        max_len=Config.LIAR_MAX_LEN
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = None
    if Config.USE_SCHEDULER:
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * Config.WARMUP_STEPS)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    
    # Create save directory if needed
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, f"liar_{args.model}_best.pt")
    
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        epochs=args.epochs,
        patience=Config.EARLY_STOPPING_PATIENCE,
        scheduler=scheduler,
        model_save_path=model_save_path
    )
    
    # Evaluate on test set
    print("\n==== LIAR Test Evaluation ====")
    test_metrics = evaluate_model(trained_model, test_loader, device, "LIAR Test")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_metrics, "liar", args.model)
    
    return trained_model, tokenizer

def transfer_learning(fnc_model, args, device):
    """Transfer knowledge from FNC-1 to LIAR dataset with multi-class support"""
    print("\n==== Transfer Learning from FNC-1 to LIAR ====")
    
    # Load LIAR data
    liar_train = load_liar_dataset(split="train")
    liar_val = load_liar_dataset(split="valid")
    liar_test = load_liar_dataset(split="test")
    
    # Sample data if needed
    if args.sample_frac < 1.0:
        liar_train = liar_train.sample(frac=args.sample_frac, random_state=42)
    
    # Clean text with advanced preprocessing if enabled
    if Config.USE_ADVANCED_PREPROCESSING:
        liar_train["statement"] = liar_train["statement"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        liar_val["statement"] = liar_val["statement"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
        liar_test["statement"] = liar_test["statement"].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=True))
    
    # Initialize a new model for LIAR
    tokenizer, liar_model = get_model_and_tokenizer(args.model, Config.LIAR_LABELS)
    
    # Transfer weights from FNC-1 model (excluding classification layer)
    pretrained_dict = fnc_model.state_dict()
    model_dict = liar_model.state_dict()
    
    # Filter out the classifier weights since dimensions won't match (4 classes vs 6 classes)
    # This is the key change for multi-class support
    transferred_dict = {}
    for k, v in pretrained_dict.items():
        # Only transfer non-classifier weights
        if "classifier" not in k and k in model_dict:
            transferred_dict[k] = v
    
    # Update the model with transferred weights
    model_dict.update(transferred_dict)
    liar_model.load_state_dict(model_dict)
    
    # Print classifier layer information for debugging - with model type handling
    try:
        # For models with different classifier structures
        if hasattr(fnc_model.classifier, 'out_proj'):
            print(f"FNC-1 output dimensions (out_proj): {fnc_model.classifier.out_proj.weight.shape}")
            print(f"LIAR output dimensions (out_proj): {liar_model.classifier.out_proj.weight.shape}")
        else:
            # For models with direct linear classifier
            print(f"FNC-1 output dimensions: {fnc_model.classifier.weight.shape}")
            print(f"LIAR output dimensions: {liar_model.classifier.weight.shape}")
    except AttributeError:
        # Fallback for any other model structure
        print("Model architectures differ - classifier structures don't match exactly")
        print(f"FNC-1 model type: {type(fnc_model.classifier)}")
        print(f"LIAR model type: {type(liar_model.classifier)}")
    
    print(f"Transferred {len(transferred_dict)}/{len(pretrained_dict)} layers")
    
    liar_model.to(device)
    
    # Prepare datasets
    train_dataset = NewsDataset(
        liar_train["statement"].tolist(),
        labels=liar_train["label"].tolist(),
        tokenizer=tokenizer,
        max_len=Config.LIAR_MAX_LEN
    )
    
    val_dataset = NewsDataset(
        liar_val["statement"].tolist(),
        labels=liar_val["label"].tolist(),
        tokenizer=tokenizer,
        max_len=Config.LIAR_MAX_LEN
    )
    
    test_dataset = NewsDataset(
        liar_test["statement"].tolist(),
        labels=liar_test["label"].tolist(),
        tokenizer=tokenizer,
        max_len=Config.LIAR_MAX_LEN
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer with different learning rates for different layers
    # Higher learning rate for the classifier (needs to learn from scratch)
    param_optimizer = list(liar_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    # Different learning rates for different parts of the model
    optimizer_grouped_parameters = [
        # Base model parameters with lower learning rate
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) 
                    and 'classifier' not in n],
         'weight_decay': Config.WEIGHT_DECAY, 'lr': args.lr / 10.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) 
                    and 'classifier' not in n],
         'weight_decay': 0.0, 'lr': args.lr / 10.0},
        # Classifier parameters with higher learning rate (needs to learn from scratch)
        {'params': [p for n, p in param_optimizer if 'classifier' in n],
         'weight_decay': Config.WEIGHT_DECAY, 'lr': args.lr * 3.0}  # Higher learning rate for classifier
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # Learning rate scheduler
    scheduler = None
    if Config.USE_SCHEDULER:
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * Config.WARMUP_STEPS)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    
    # Create save directory if needed
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    model_save_path = os.path.join(Config.MODEL_SAVE_PATH, f"transfer_{args.model}_best.pt")
    
    # Fine-tune the model
    trained_model = train_model(
        liar_model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        epochs=args.epochs,
        patience=Config.EARLY_STOPPING_PATIENCE,
        scheduler=scheduler,
        model_save_path=model_save_path
    )
    
    # Evaluate on test set
    print("\n==== Transfer Learning Test Evaluation ====")
    test_metrics = evaluate_model(trained_model, test_loader, device, "Transfer Test")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_metrics, "transfer", args.model)
    
    return trained_model

def plot_confusion_matrix(metrics, dataset_name, model_name):
    """Plot and save confusion matrix from metrics"""
    try:
        # Create figures directory if it doesn't exist
        os.makedirs("figures", exist_ok=True)
        
        # Extract metrics for plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(metrics), annot=True, fmt='.4f', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name.upper()} with {model_name.upper()}')
        plt.tight_layout()
        plt.savefig(f'fake-news-nlp/figures/{dataset_name}_{model_name}_confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error in plotting confusion matrix: {e}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train on selected dataset(s)
    if args.dataset in ['fnc1', 'both']:
        fnc_model, tokenizer = train_fnc1(args, device)
        
        if args.dataset == 'both':
            # Transfer learning from FNC-1 to LIAR
            transfer_model = transfer_learning(fnc_model, args, device)
    
    if args.dataset == 'liar':
        liar_model, tokenizer = train_liar(args, device)
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()