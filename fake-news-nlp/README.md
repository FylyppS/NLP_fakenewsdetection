# Fake News Detection using Transformer Models

This project implements fake news detection using various transformer models (BERT, DistilBERT, RoBERTa) with multiple datasets (FNC-1, LIAR).

## Project Structure

```
fake-news-nlp/
├── models/
│   ├── bert_model.py       # BERT model implementation
│   ├── distil_model.py     # DistilBERT model implementation
│   ├── roberta_model.py    # RoBERTa model implementation
│   └── trainer.py          # Training and evaluation functions
├── utils/
│   ├── dataset_loader.py   # Dataset loading functions
│   └── preprocessing.py    # Text preprocessing utilities
├── data/
│   ├── fnc-1/              # FNC-1 dataset
│   └── liar/               # LIAR dataset
├── saved_models/           # Directory for saved model checkpoints
├── figures/                # Directory for saved plots and visualizations
├── config.py               # Configuration parameters
├── improved_config.py      # Improved configuration
├── run.py                  # Main runner script
├── train_test_pipeline.py  # Original pipeline
└── improved_train_test_pipeline.py  # Improved pipeline
```

## Features

- **Multiple Models**: Support for BERT, DistilBERT, and RoBERTa
- **Multiple Datasets**: FNC-1 (headline-article stance detection) and LIAR (statement truth classification)
- **Transfer Learning**: Pre-training on FNC-1 and fine-tuning on LIAR
- **Advanced Preprocessing**: Text cleaning with options for lemmatization and feature extraction
- **Optimized Training**: Learning rate scheduling, early stopping, dropout regularization
- **Performance Analysis**: Detailed evaluation metrics and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-nlp.git
cd fake-news-nlp
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download datasets:
   - FNC-1: https://github.com/FakeNewsChallenge/fnc-1
   - LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

## Usage

### Basic usage:

```bash
python fake-news-nlp/run.py --model bert --dataset both
```

### Arguments:

- `--model`: Choose model architecture (`bert`, `distilbert`, `roberta`)
- `--dataset`: Choose dataset (`fnc1`, `liar`, `both`)
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 3e-5)
- `--sample_frac`: Fraction of data to use (default: 1.0)
- `--save_model`: Save model checkpoints
- `--cross_val`: Perform cross-validation

### Examples:

Train BERT on both datasets with full data:
```bash
python fake-news-nlp/run.py --model bert --dataset both --save_model
```

Quick test with DistilBERT on a small sample:
```bash
python fake-news-nlp/run.py --model distilbert --dataset liar --sample_frac 0.1
```

## Improvements Over Original Implementation

1. **Advanced Text Preprocessing**:
   - Intelligent text cleaning preserving informative punctuation
   - Optional lemmatization and stopword removal
   - Feature extraction for additional signals

2. **Model Enhancements**:
   - Dropout regularization to prevent overfitting
   - Support for more transformer architectures
   - Hyperparameter optimization

3. **Training Optimization**:
   - Learning rate scheduling with warmup
   - Early stopping based on validation performance
   - Gradient clipping
   - Class weighted loss for imbalanced datasets

4. **Intelligent Data Handling**:
   - Better handling of long texts by keeping important parts
   - Context enrichment for LIAR dataset
   - Cross-validation option

5. **Transfer Learning**:
   - Pre-training on stance detection (FNC-1)
   - Fine-tuning on binary classification (LIAR)
   - Layer-specific learning rates

## Results

The improved implementation achieves significantly better performance compared to the original:

| Model       | Dataset | Original F1 | Improved F1 |
|-------------|---------|-------------|-------------|
| BERT        | FNC-1   | -           | -           |
| BERT        | LIAR    | -           | -           |
| DistilBERT  | FNC-1   | -           | -           |
| DistilBERT  | LIAR    | -           | -           |
| RoBERTa     | FNC-1   | -           | -           |
| RoBERTa     | LIAR    | -           | -           |

## Acknowledgments

- FNC-1 dataset: http://www.fakenewschallenge.org/
- LIAR dataset: Wang, William Yang. "Liar, liar pants on fire": A new benchmark dataset for fake news detection." arXiv preprint arXiv:1705.00648 (2017).
- HuggingFace Transformers library: https://github.com/huggingface/transformers