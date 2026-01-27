FROM python:3.11-slim

WORKDIR /app

# Needed for opencv/easyocr runtime on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt /app/requirements.docker.txt

# Force CPU-only PyTorch (prevents downloading nvidia_* CUDA wheels)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install the rest of your dependencies
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . /app

EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
