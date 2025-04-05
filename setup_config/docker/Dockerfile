FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY setup_config/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for data and output if they don't exist
RUN mkdir -p mimic_iv_data
RUN mkdir -p output
RUN mkdir -p models
RUN mkdir -p logs

# Make sure the directories are writable
RUN chmod -R 777 mimic_iv_data
RUN chmod -R 777 output
RUN chmod -R 777 models
RUN chmod -R 777 logs

# Expose port for Streamlit
EXPOSE 8501
# Expose port for API (if used)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the Streamlit application (can be overridden in docker-compose.yml)
CMD ["streamlit", "run", "src/visualization/app.py"]
