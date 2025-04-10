FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY config/requirements.txt .

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.cargo/bin:$PATH"

# Install Python dependencies
RUN uv pip install --no-cache-dir -r requirements.txt

# Install Jupyter (already in requirements.txt)

# Copy project files
COPY . .

# Expose port for Jupyter
EXPOSE 8888

# Command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
