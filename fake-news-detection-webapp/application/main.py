from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

import config
from application.routes import api
from application.services.model_service import ModelService

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(Path(__file__).resolve().parent, "static")), name="static")

# Initialize templates
templates = Jinja2Templates(directory=os.path.join(Path(__file__).resolve().parent, "templates"))

# Include API routes
app.include_router(api.router, prefix=config.API_V1_PREFIX)

# Pre-load model during app startup
@app.on_event("startup")
async def startup_event():
    # Initialize the model service (this loads the model)
    ModelService.get_instance()
    print("Model loaded successfully!")

# Web routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}