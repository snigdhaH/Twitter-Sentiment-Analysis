# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data (for stopwords)
RUN python -m nltk.downloader stopwords

# Set the port Render will use
ENV PORT=10000

# Expose that port
EXPOSE $PORT

# Start the server with Gunicorn
CMD ["gunicorn", "--workers=1", "--bind=0.0.0.0:$PORT", "app:app"]
