# Base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy everything
COPY app/ app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r app/requirements.txt

# Expose FastAPI on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]