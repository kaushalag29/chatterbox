# Base image with Anaconda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Set working directory
WORKDIR /app

# Install system dependencies
# git is needed for some pip installs from git repos.
# ffmpeg is a common dependency for audio processing libraries like librosa.
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && apt-get clean

# Copy the application files into the container
COPY . .

# Create a conda environment for the application
RUN conda create -n chatterbox python=3.11 -y

# Activate the conda environment for subsequent commands
SHELL ["conda", "run", "-n", "chatterbox", "/bin/bash", "-c"]

# Install the project and its dependencies from pyproject.toml
# The '.' refers to the current directory (/app) where pyproject.toml is located.
RUN pip install -e .

# Pre-download the Chatterbox model during build time
# This ensures the model is cached in the Docker image and doesn't need to be downloaded at runtime
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"

# Expose the port the server will run on
EXPOSE 8000

# NOTE: The server.py file has a hardcoded path to a file in the OpenVoice submodule:
# audio_prompt_path = "./../OpenVoice/target000.wav"
# This will cause a runtime error because the OpenVoice directory is not part of this container.
# To fix this, you will need to make this file available to the container,
# for example by using a multi-stage build to copy it, or by mounting a volume.

# Command to run the application server with conda environment activated
CMD ["conda", "run", "-n", "chatterbox", "python", "server.py"] 