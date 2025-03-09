from http.client import HTTPException
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil
from typing import Dict, Any

from app import get_all_model_predictions, create_confidence_chart, create_voting_chart

app = FastAPI()

@app.post("/api/predict")
async def predict_audio(request: Request, file: UploadFile = File(...)):
    # Create a temporary directory that will be cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file temporarily
        tmp_path = os.path.join(temp_dir, 'audio.wav')
        with open(tmp_path, 'wb') as tmp:
            shutil.copyfileobj(file.file, tmp)

        try:
            # Get predictions from all models
            results, best_model, best_prediction, highest_confidence = get_all_model_predictions(tmp_path)
            
            # Create visualization charts in temp directory
            confidence_chart = create_confidence_chart(results, best_model, output_dir=temp_dir)
            voting_chart = create_voting_chart(results, output_dir=temp_dir)
            
            # Read chart data if they exist
            chart_data = {}
            for chart_path in [confidence_chart, voting_chart]:
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as f:
                        chart_data[os.path.basename(chart_path)] = f.read()
            
            # Prepare response
            response = {
                "predictions": results,
                "best_model": best_model,
                "best_prediction": best_prediction,
                "confidence": highest_confidence,
                "charts": chart_data
            }

            return JSONResponse(content=response)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
