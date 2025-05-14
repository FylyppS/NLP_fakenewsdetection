import uvicorn
from application.main import app

if __name__ == "__main__":
    uvicorn.run("application.main:app", host="0.0.0.0", port=8000, reload=True)