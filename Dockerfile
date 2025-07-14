# Use official Python base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8000

# Command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.enableCORS=false"]
