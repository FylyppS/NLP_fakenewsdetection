import re
import string
import nltk
# Download NLTK resources (uncomment first time)
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False, lemmatize=False):
    """
    Advanced text cleaning with options for lemmatization and stopword removal
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with [URL]
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Replace emails with [EMAIL]
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Replace numbers with [NUM] but keep years intact
    text = re.sub(r'\b(?<!\d)\d{1,3}(?:\.\d+)?(?!\d|\.\d)\b', '[NUM]', text)
    
    # Keep sentence structure by preserving some punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\":]', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords or lemmatize:
        tokens = nltk.word_tokenize(text)
        
        if remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in stop_words]
            
        if lemmatize:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
        text = ' '.join(tokens)
    
    return text

def extract_features(text):
    """
    Extract additional features from text for fake news detection
    """
    features = {}
    
    # Count of special characters
    features['special_char_count'] = len(re.findall(r'[!?#$%^&*()]', text))
    
    # Presence of all caps words
    features['has_all_caps'] = 1 if re.search(r'\b[A-Z]{2,}\b', text) else 0
    
    # Count of exclamation/question marks (potential sensationalism)
    features['exclaim_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Count of quotation marks (potential misrepresentation)
    features['quote_count'] = text.count('"') + text.count("'")
    
    # Text length
    features['text_length'] = len(text.split())
    
    return features