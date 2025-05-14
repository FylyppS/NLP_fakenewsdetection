from pydantic import BaseModel
from typing import Dict, Optional, List

class Message(BaseModel):
    text: str
    url: Optional[str] = None

class PredictionInput(BaseModel):
    message: Message

class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]

class TextAnalysisResult(BaseModel):
    original_text: str
    cleaned_text: str
    keywords: List[str]
    sentiment: Dict[str, float]
    statistics: Dict[str, int]
    prediction: PredictionOutput