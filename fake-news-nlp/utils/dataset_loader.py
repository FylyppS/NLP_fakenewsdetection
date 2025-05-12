import pandas as pd
import numpy as np
from utils.preprocessing import clean_text

def load_fnc1_dataset(path="fake-news-nlp/data/fnc-1/", split="train", sample_frac=1.0):
    """
    Returns headlines, bodies and labels for FNC-1 dataset with optional filtering
    
    Args:
        path (str): Path to the dataset directory
        split (str): 'train' or 'val' split
        sample_frac (float): Fraction of data to sample (0-1)
        
    Returns:
        DataFrame: DataFrame with headline, body, and label columns
    """
    # Load datasets
    stances = pd.read_csv(path + "train_stances.csv")
    bodies = pd.read_csv(path + "train_bodies.csv")
    
    # Split into train/val
    if split == "train":
        dataset = stances.sample(frac=sample_frac*0.8, random_state=42)  # 80% for training
    elif split == "val":
        all_samples = stances.sample(frac=sample_frac, random_state=42)
        train_samples = stances.sample(frac=sample_frac*0.8, random_state=42)
        dataset = all_samples.drop(train_samples.index)
    else:
        raise ValueError("Split must be 'train' or 'val'")
    
    # Merge stances with article bodies
    dataset = dataset.merge(bodies, how="left", on="Body ID")
    
    # Convert stance labels to numeric values
    label_map = {"discuss": 0, "agree": 1, "disagree": 2, "unrelated": 3}
    dataset["label"] = dataset["Stance"].map(label_map)
    
    # Rename columns for clarity
    dataset["headline"] = dataset["Headline"]
    dataset["body"] = dataset["articleBody"]
    
    # Perform basic cleaning (proper preprocessing happens in pipeline)
    dataset["headline"] = dataset["headline"].apply(lambda x: str(x).strip())
    dataset["body"] = dataset["body"].apply(lambda x: str(x).strip())
    
    # Check for any missing values
    if dataset["headline"].isnull().sum() > 0 or dataset["body"].isnull().sum() > 0:
        print(f"Warning: Found {dataset['headline'].isnull().sum()} null headlines and "
              f"{dataset['body'].isnull().sum()} null bodies.")
        # Fill any missing values
        dataset["headline"] = dataset["headline"].fillna("")
        dataset["body"] = dataset["body"].fillna("")
    
    # Add class weights for imbalanced dataset
    class_counts = dataset["label"].value_counts()
    total_samples = len(dataset)
    class_weights = {label: total_samples / (len(class_counts) * count) 
                     for label, count in class_counts.items()}
    
    # Print dataset information
    print(f"Loaded {len(dataset)} samples for {split} split.")
    print(f"Class distribution: {dataset['label'].value_counts().to_dict()}")
    print(f"Class weights: {class_weights}")
    
    return dataset[["headline", "body", "label"]]

def load_liar_dataset(path="fake-news-nlp/data/liar/", split="train"):
    """
    Returns statements and labels for LIAR dataset
    
    Args:
        path (str): Path to the dataset directory
        split (str): 'train', 'valid' or 'test' split
        
    Returns:
        DataFrame: DataFrame with statement and label columns
    """
    # Map split name to filename
    filename = {
        "train": "train.tsv",
        "valid": "valid.tsv", 
        "test": "test.tsv"
    }[split]
    
    # Load dataset
    df = pd.read_csv(path + filename, sep="\t", header=None)
    
    # Assign column names
    df.columns = [
        "id", "label", "statement", "subject", "speaker", "job", 
        "state", "party", "barely_true", "false", "half_true", 
        "mostly_true", "pants_on_fire", "context"
    ]
    
    # Multi-class labels (0-5) - preserve original classes
    label_map = {
        "true": 0, 
        "mostly-true": 1, 
        "half-true": 2,
        "barely-true": 3, 
        "false": 4, 
        "pants-fire": 5
    }
    df["label"] = df["label"].map(label_map)
    
    # Drop rows with missing labels
    before_drop = len(df)
    df = df.dropna(subset=["label"])
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing labels.")
    
    # Perform basic cleaning (proper preprocessing happens in pipeline)
    df["statement"] = df["statement"].apply(lambda x: str(x).strip())
    
    # Enrich statements with context when available
    df["context"] = df["context"].fillna("")
    df["enriched_statement"] = df.apply(
        lambda row: f"{row['statement']} {row['context']}".strip(), axis=1
    )
    
    # Add speaker information when available
    df["speaker"] = df["speaker"].fillna("")
    df["enriched_statement"] = df.apply(
        lambda row: f"{row['enriched_statement']} Speaker: {row['speaker']}".strip() 
                    if row["speaker"] else row["enriched_statement"], 
        axis=1
    )
    
    # Print dataset information
    print(f"Loaded {len(df)} samples for {split} split.")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Calculate class weights
    class_counts = df["label"].value_counts()
    total_samples = len(df)
    class_weights = {label: total_samples / (len(class_counts) * count) 
                     for label, count in class_counts.items()}
    print(f"Class weights: {class_weights}")
    
    return df[["enriched_statement", "label"]].rename(columns={"enriched_statement": "statement"})