FROM python:3.10.5

# Set environment variable for non-interactive installs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for Render (Render sets the PORT env automatically)
EXPOSE $PORT

# Start your app
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:$PORT", "app:app"]
