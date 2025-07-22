# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY deployment/ ./  # assumes you're copying from the deployment/ folder in repo
COPY deployment/model ./model  # include saved model folder


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
