# Use an official NVIDIA CUDA runtime image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    default-libmysqlclient-dev\
    cmake \
    git \
    python3.11 \
    python3-pip \
    python3-dev

# Set the working directory in the container
WORKDIR /code

# Copy the code directory contents into the container at /code
#COPY ./code /code

COPY ./requirements.txt ./

# Set the CMAKE_ARGS environment variable
ENV CMAKE_ARGS="-DGGML_CUDA=on"

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install llama-cpp-python

# Make port 8888 available to the world outside this container
EXPOSE 8888