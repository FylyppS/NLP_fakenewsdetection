import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
import numpy as np
from typing import Dict, Any

import config
from application.models.prediction import PredictionOutput

class ModelService:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure model is loaded only once"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize model and tokenizer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get model config from settings
        model_config = config.MODELS[config.DEFAULT_MODEL]
        model_path = model_config["path"]
        model_type = model_config["type"]
        model_name = model_config["name"]
        
        # Load tokenizer based on model type
        if model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, text: str) -> PredictionOutput:
        """
        Make a prediction for the given text
        """
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=config.MAX_LENGTH
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits and apply softmax to get probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Convert to numpy for easier handling
        probs_np = probabilities.cpu().numpy()[0]
        predicted_class = np.argmax(probs_np)
        confidence = float(probs_np[predicted_class])
        
        # Create dictionary of class probabilities
        class_probs = {
            config.LABEL_MAPPING[i]: float(probs_np[i])
            for i in range(len(probs_np))
        }
        
        # Create prediction output
        prediction = PredictionOutput(
            predicted_class=int(predicted_class),
            predicted_label=config.LABEL_MAPPING[int(predicted_class)],
            confidence=confidence,
            probabilities=class_probs
        )
        
        return prediction