### Step 1: Create a Flask or FastAPI Application

Here’s an example using FastAPI, which is known for its performance and ease of use:

#### FastAPI Example

1. **Install FastAPI and Uvicorn**:
   ```bash
   pip install fastapi uvicorn
   ```

2. **Create a new file, e.g., `app.py`**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your model and tokenizer
model_path = "path/to/your/model.pt"  # Update with your model path
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

3. **Run the FastAPI application**:
   ```bash
   uvicorn app:app --reload
   ```

4. **Test the API**:
   You can use tools like Postman or curl to send a POST request to `http://localhost:8000/classify/` with a JSON body:
   ```json
   {
       "text": "Your message or article text here."
   }
   ```

### Step 2: Model Saving Method Suitability

Your current model saving method, which uses `torch.save()` to save the model's state dictionary, is suitable for use in a web application. However, consider the following:

1. **Loading the Model**: Ensure that the model is loaded only once when the application starts, as shown in the FastAPI example. This avoids reloading the model for every request, which can be inefficient.

2. **Model Versioning**: If you plan to update your model frequently, consider implementing a versioning system for your saved models. This allows you to switch between different model versions without downtime.

3. **Error Handling**: Implement error handling in your API to manage cases where the model fails to load or if the input data is invalid.

4. **Concurrency**: FastAPI is asynchronous and can handle multiple requests simultaneously. Ensure that your model inference code is thread-safe if you are using shared resources.

5. **Performance**: Depending on the size of your model and the hardware you are using, you may want to consider using GPU acceleration for inference if available.

### Conclusion

The provided FastAPI example allows users to input text for classification using your NLP model. Your current model saving method is suitable for a web application, provided you follow best practices for loading and managing the model.