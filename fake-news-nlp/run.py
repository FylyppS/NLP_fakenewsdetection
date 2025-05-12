#!/usr/bin/env python
"""
Fake News Detection - Runner Script
This script provides a simple interface to run various experiments.
"""
import os
import argparse
from improved_train_test_pipeline import main as run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Fake News Detection Experiments')
    parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'], 
                        default='bert', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, choices=['fnc1', 'liar', 'both'], 
                        default='both', help='Dataset to train/evaluate on')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5, 
                        help='Learning rate')
    parser.add_argument('--sample_frac', type=float, default=1.0, 
                        help='Fraction of data to use (0-1)')
    parser.add_argument('--save_model', action='store_true', 
                        help='Save the trained model')
    parser.add_argument('--cross_val', action='store_true', 
                        help='Perform cross-validation')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Running experiment with model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Sample fraction: {args.sample_frac}")
    print("=" * 50)
    
    # Create directories for outputs
    #os.makedirs("fake-news-nlp/saved_models", exist_ok=True)
    #os.makedirs("fake-news-nlp/figures", exist_ok=True)
    
    # Run the pipeline
    run_pipeline()
    
    print("\nExperiment completed! Check 'saved_models/' for model files and 'figures/' for visualizations.")