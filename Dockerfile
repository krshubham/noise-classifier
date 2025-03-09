FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Create necessary directories
RUN mkdir -p static/icons

# Copy the application code and static files
COPY . .

# Create static directory if it doesn't exist
RUN mkdir -p static

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose the port
EXPOSE ${PORT}

# Command to run the FastAPI server
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 4"]
