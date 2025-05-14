import re
from typing import List, Dict, Any
import newspaper
from newspaper import Article
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class TextService:
    """Service for text processing and analysis"""
    
    @staticmethod
    def extract_text_from_url(url: str) -> str:
        """Extract main text content from a URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            raise Exception(f"Failed to extract text from URL: {str(e)}")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract most common words after removing stopwords"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Get most common words
        word_counts = Counter(filtered_tokens)
        return [word for word, _ in word_counts.most_common(top_n)]
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, float]:
        """Analyze sentiment of text using VADER"""
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        return sentiment_scores
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict[str, int]:
        """Get basic text statistics"""
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0
            }
            
        # Character count
        char_count = len(text)
        
        # Word count
        words = word_tokenize(text)
        word_count = len(words)
        
        # Sentence count (rough estimate based on punctuation)
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        if sentence_count < 1:
            sentence_count = 1
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count
        }