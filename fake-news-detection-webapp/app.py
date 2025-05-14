from fastapi import FastAPI
from pydantic import BaseModel

# Note: We don't need to initialize the model here since we're using ModelService
# The actual model loading happens in application/services/model_service.py
# and is initialized at startup in application/main.py

app = FastAPI()

class Message(BaseModel):
    text: str
    url: str = None

# We don't need this endpoint because it's already defined in application/routes/api.py
# and is included in the main app via app.include_router(api.router, prefix=config.API_V1_PREFIX)

# Import the main application
from application.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application.main:app", host="0.0.0.0", port=8000, reload=True)