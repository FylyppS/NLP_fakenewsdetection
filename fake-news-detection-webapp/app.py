from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your model and tokenizer
model_path = "fake-news-nlp/models/bert_model.py"  # Update with your model path
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/classify/")
async def classify_message(message: Message):
    inputs = tokenizer(message.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return {"predicted_class": predicted_class}

from application.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application.main:app", host="0.0.0.0", port=8000, reload=True)