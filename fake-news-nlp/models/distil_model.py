from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    DistilBertConfig
)

def get_tokenizer_and_model(model_name="distilbert-base-uncased", num_labels=2, **kwargs):
    """
    Initialize DistilBERT tokenizer and model with optional configuration
    
    Args:
        model_name (str): Name of the pretrained model
        num_labels (int): Number of output labels
        **kwargs: Additional configuration parameters
    
    Returns:
        tuple: (tokenizer, model)
    """
    # Default configuration parameters with better defaults for fake news detection
    config_params = {
        "dropout": 0.2,  # More dropout to prevent overfitting
        "num_labels": num_labels
    }
    
    # Override with any user-provided parameters
    config_params.update(kwargs)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Create model with custom configuration
    config = DistilBertConfig.from_pretrained(model_name, **config_params)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)
    
    return tokenizer, model