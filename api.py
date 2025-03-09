from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import base64
from typing import Dict, Any
import shutil
import json
import numpy as np

from app import get_all_model_predictions, create_confidence_chart, create_voting_chart

def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = FastAPI(title="Engine Sound Classifier")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html file"""
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Process uploaded audio file and return predictions"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Get predictions from all models
        results, best_model, best_prediction, highest_confidence = get_all_model_predictions(tmp_path)
        
        # Convert numpy types to Python native types
        results = convert_numpy_types(results)
        highest_confidence = float(highest_confidence) if isinstance(highest_confidence, np.floating) else highest_confidence
        
        # Create visualization charts
        confidence_chart_path = create_confidence_chart(results, best_model)
        voting_chart_path = create_voting_chart(results)
        
        # Read and encode charts if they exist
        confidence_chart_data = None
        voting_chart_data = None
        
        if confidence_chart_path and os.path.exists(confidence_chart_path):
            with open(confidence_chart_path, "rb") as img_file:
                confidence_chart_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        if voting_chart_path and os.path.exists(voting_chart_path):
            with open(voting_chart_path, "rb") as img_file:
                voting_chart_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare response
        response = {
            "predictions": results,
            "best_model": best_model,
            "best_prediction": best_prediction,
            "confidence": highest_confidence,
            "confidence_chart": confidence_chart_data,
            "voting_chart": voting_chart_data
        }

        return response

    finally:
        # Clean up temporary files
        os.unlink(tmp_path)
        for chart_path in [confidence_chart_path, voting_chart_path]:
            if chart_path and os.path.exists(chart_path):
                os.unlink(chart_path)

@app.get("/api/models")
async def get_models():
    """Return a list of available models"""
    models_dir = 'models'
    model_names = {
        'lr_model.joblib': 'Logistic Regression',
        'nn_model.joblib': 'Neural Network',
        'rf_model.joblib': 'Random Forest',
        'svm_model.joblib': 'Support Vector Machine',
        'xgb_model.joblib': 'XGBoost'
    }
    
    available_models = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
        available_models = [model_names[file] for file in model_files if file in model_names]
    
    return {"models": available_models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)