from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

from application.models.prediction import Message, PredictionInput, PredictionOutput, TextAnalysisResult
from application.services.model_service import ModelService
from application.services.text_service import TextService

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(Path(__file__).resolve().parent.parent, "templates"))

@router.post("/classify", response_model=PredictionOutput)
async def classify_text(input_data: PredictionInput):
    """
    Classify text as real or fake news
    """
    try:
        # Get text from input (either direct text or from URL)
        if input_data.message.url:
            # Extract text from URL using TextService
            text = TextService.extract_text_from_url(input_data.message.url)
        else:
            text = input_data.message.text
            
        # Get model service singleton
        model_service = ModelService.get_instance()
        
        # Make prediction
        prediction = model_service.predict(text)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/analyze", response_model=TextAnalysisResult)
async def analyze_text(input_data: PredictionInput):
    """
    Complete text analysis including classification, keywords, sentiment, etc.
    """
    try:
        # Get text from input (either direct text or from URL)
        original_text = input_data.message.text
        if input_data.message.url:
            # Extract text from URL using TextService
            original_text = TextService.extract_text_from_url(input_data.message.url)
        
        # Clean and analyze text
        cleaned_text = TextService.clean_text(original_text)
        keywords = TextService.extract_keywords(cleaned_text)
        sentiment = TextService.analyze_sentiment(cleaned_text)
        statistics = TextService.get_text_statistics(original_text)
        
        # Get model prediction
        model_service = ModelService.get_instance()
        prediction = model_service.predict(cleaned_text)
        
        # Create complete analysis result
        analysis = TextAnalysisResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            keywords=keywords,
            sentiment=sentiment,
            statistics=statistics,
            prediction=prediction
        )
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@router.get("/results/{prediction_id}", response_class=HTMLResponse)
async def show_results(request: Request, prediction_id: str):
    """
    Show detailed prediction results
    """
    # This would typically fetch results from a database
    # For simplicity, we're just rendering a template
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request, 
            "prediction_id": prediction_id
        }
    )