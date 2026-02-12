# GPU-ready image for training/inference
FROM pytorch/pytorch:2.2.0-cuda12.2-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml ./
RUN pip install poetry
