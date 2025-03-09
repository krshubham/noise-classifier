from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
import os
from typing import Dict, Any
import shutil

from app import get_all_model_predictions, create_confidence_chart, create_voting_chart

app = FastAPI(title="Engine Sound Classifier")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Engine Sound Classifier API is running"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Get predictions from all models
        results, best_model, best_prediction, highest_confidence = get_all_model_predictions(tmp_path)
        
        # Create visualization charts
        confidence_chart = create_confidence_chart(results, best_model)
        voting_chart = create_voting_chart(results)
        
        # Prepare response
        response = {
            "predictions": results,
            "best_model": best_model,
            "best_prediction": best_prediction,
            "confidence": highest_confidence,
            "confidence_chart": None,
            "voting_chart": None
        }

        # Read and encode charts if they exist
        if confidence_chart and os.path.exists(confidence_chart):
            response["confidence_chart"] = confidence_chart
        if voting_chart and os.path.exists(voting_chart):
            response["voting_chart"] = voting_chart

        return response

    finally:
        # Clean up temporary files
        os.unlink(tmp_path)
        for chart in [confidence_chart, voting_chart]:
            if chart and os.path.exists(chart):
                os.unlink(chart)

@app.get("/charts/{chart_path:path}")
async def get_chart(chart_path: str):
    """Serve chart images"""
    if os.path.exists(chart_path):
        return FileResponse(chart_path)
    return {"error": "Chart not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)