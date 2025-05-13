import torch
import transformers
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

transformers.logging.set_verbosity_error()

class NewsDataset(Dataset):
    def __init__(self, texts_a, texts_b=None, labels=None, tokenizer=None, max_len=128):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        # Different processing for FNC-1 (text pairs) vs LIAR (single text)
        if self.texts_b is not None:
            # Truncate longer text intelligently - take beginning and end
            text_a = self.texts_a[idx]
            text_b = self.texts_b[idx]
            
            # For article bodies, extract the first and last parts for better signal
            if len(text_b.split()) > 100:  # If the body is long
                words = text_b.split()
                first_part = " ".join(words[:100])
                last_part = " ".join(words[-100:])
                text_b = first_part + " [...] " + last_part
            
            encoded = self.tokenizer(
                text_a, 
                text_b,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
        else:
            encoded = self.tokenizer(
                self.texts_a[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

        item = {key: val.squeeze(0) for key, val in encoded.items()}
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

def train_model(model, train_loader, val_loader, optimizer, device, 
                epochs=5, patience=3, scheduler=None, model_save_path="best_model.pt"):
    """Training loop with validation and early stopping"""
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
                
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Validation
        avg_loss = total_loss / len(train_loader)
        print(f"\nTrain Loss: {avg_loss:.4f}")
        val_metrics = evaluate_model(model, val_loader, device, "Validation")
        
        # Early stopping based on F1 score
        val_f1 = val_metrics['weighted avg']['f1-score']
        print(f"Validation F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(model_save_path))
    return model

class WeightedCrossEntropyLoss(nn.Module):
    """
    Implementacja ważonej funkcji straty dla niezbalansowanych zbiorów danych
    """
    def __init__(self, class_weights: Dict[int, float], device: torch.device):
        super(WeightedCrossEntropyLoss, self).__init__()
        # Przekształć słownik wag na tensor
        self.weights = torch.FloatTensor([class_weights[i] for i in sorted(class_weights.keys())])
        self.weights = self.weights.to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')
        
    def forward(self, logits, targets):
        return self.ce_loss(logits, targets)

def train_model_with_weighted_loss(
    model, train_loader, val_loader, optimizer, device, 
    class_weights, epochs=5, patience=3, scheduler=None, model_save_path="best_model.pt"
):
    """Training loop z ważoną funkcją straty i walidacją"""
    # Inicjalizacja ważonej funkcji straty
    criterion = WeightedCrossEntropyLoss(class_weights, device)
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training
        from tqdm import tqdm
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            # Pobieramy etykiety, aby użyć ich oddzielnie z naszą własną funkcją straty
            labels = batch.pop("labels")
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Obliczenie straty przy użyciu niestandardowej funkcji straty
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
                
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Validation
        avg_loss = total_loss / len(train_loader)
        print(f"\nTrain Loss: {avg_loss:.4f}")
        
        # Tutaj używamy standardowej funkcji evaluate_model
        from models.trainer import evaluate_model
        val_metrics = evaluate_model(model, val_loader, device, "Validation")
        
        # Early stopping based on F1 score
        val_f1 = val_metrics['weighted avg']['f1-score']
        print(f"Validation F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(model_save_path))
    return model

def evaluate_model(model, dataloader, device, prefix="Evaluation"):
    """Evaluation with classification report"""
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=prefix):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    
    report = classification_report(true_labels, predictions, digits=4, output_dict=True)
    print(f"\n{prefix} Report:")
    print(classification_report(true_labels, predictions, digits=4))
    
    return report