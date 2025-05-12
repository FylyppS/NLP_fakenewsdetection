from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    BertConfig
)

def get_tokenizer_and_model(model_name="bert-base-uncased", num_labels=2, **kwargs):
    """
    Initialize BERT tokenizer and model with optional configuration
    
    Args:
        model_name (str): Name of the pretrained model
        num_labels (int): Number of output labels
        **kwargs: Additional configuration parameters
    
    Returns:
        tuple: (tokenizer, model)
    """
    # Default configuration parameters with better defaults for fake news detection
    config_params = {
        "hidden_dropout_prob": 0.2,  # More dropout to prevent overfitting
        "attention_probs_dropout_prob": 0.2,  # More attention dropout
        "num_labels": num_labels
    }
    
    # Override with any user-provided parameters
    config_params.update(kwargs)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Create model with custom configuration
    config = BertConfig.from_pretrained(model_name, **config_params)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    
    return tokenizer, model