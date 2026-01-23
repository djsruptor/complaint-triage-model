from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import os

from src.config import API_NAME, API_VERSION, DEFAULT_THRESHOLD
from src.model import ComplaintPriorityModel


app = FastAPI(title=API_NAME, version=API_VERSION)

model = ComplaintPriorityModel()

class ScoreRequest(BaseModel):
    complaint_text: str = Field(min_length=1, description="Raw complaint narrative text")

class ScoreResponse(BaseModel):
    high_priority_score: float
    predicted_label: int
    threshold: float

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    score = model.predict_proba(req.complaint_text)
    label = int(score >= DEFAULT_THRESHOLD)

    return ScoreResponse(
        high_priority_score=score,
        predicted_label=label,
        threshold=DEFAULT_THRESHOLD
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)