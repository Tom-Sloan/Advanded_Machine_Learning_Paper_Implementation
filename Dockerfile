# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV CONDA_ENV_NAME=general_purpose
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment from environment.yml
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

# Activate conda environment by default
SHELL ["conda", "run", "-n", "general_purpose", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy essential files first
COPY descriptions.md ./
COPY environment.yml ./
COPY run_paper.py ./

# Copy source code and scripts
COPY src/ ./src/
COPY data/download_dataset.sh ./data/

# Make scripts executable
RUN chmod +x run_paper.py data/download_dataset.sh

# Create necessary directories
RUN mkdir -p /app/trained_models /app/paper_management/PDFs

# Set the default command to run run_paper.py
CMD ["conda", "run", "--no-capture-output", "-n", "general_purpose", "python", "run_paper.py"] 