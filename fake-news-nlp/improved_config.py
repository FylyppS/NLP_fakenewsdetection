class Config:
    # Data
    FNC1_PATH = "data/fnc-1/"
    LIAR_PATH = "data/liar/"
    
    # Model options
    MODELS = {
        "bert": {
            "name": "bert-base-uncased",
            "type": "bert"
        },
        "distilbert": {
            "name": "distilbert-base-uncased",
            "type": "distilbert"
        },
        "roberta": {
            "name": "roberta-base",
            "type": "roberta"
        }
    }
    
    # Default model selection
    DEFAULT_MODEL = "bert"
    
    # Sequence lengths
    FNC1_MAX_LEN = 512  # For article-headline pairs
    LIAR_MAX_LEN = 256  # Increased length for statements with context
    
    # Training
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 0.1  # Proportion of total steps
    EARLY_STOPPING_PATIENCE = 3
    
    # Data usage
    SAMPLE_FRAC = 1.0  # Use all available data
    
    # Labels
    FNC1_LABELS = 4  # discuss, agree, disagree, unrelated
    LIAR_LABELS = 6  # true, mostly-true, half-true, barely-true, false, pants-fire
    
    # Training options
    USE_ADVANCED_PREPROCESSING = True
    USE_SCHEDULER = True
    USE_EARLY_STOPPING = True
    
    # Model saving
    MODEL_SAVE_PATH = "fake-news-nlp/saved_models/"
    
    # Cross-validation
    NUM_FOLDS = 5
    DO_CROSS_VALIDATION = False