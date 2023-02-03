# This assumes the container is running on a system with a CUDA GPU
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install -r requirements.txt