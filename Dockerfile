FROM python:3.11-slim

WORKDIR /app

# Copy only requirements
COPY requirements.txt .

# Install dependencies inside container
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app code (exclude bvenv)
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]