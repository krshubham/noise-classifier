FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY api.py.requirements.txt .
RUN pip install --no-cache-dir -r api.py.requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy the application code
COPY . .

# Expose the port
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
