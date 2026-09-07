# Use the official NVIDIA PyTorch runtime with CUDA acceleration pre-configured
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files to disk and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for data processing and scipy image filters
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install target python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Specify the default command to execute when launching the container
CMD ["python", "train_pinn.py"]
