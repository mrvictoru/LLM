# Use an official Python runtime as a parent image
FROM python:slim

RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing build-essential pkg-config default-libmysqlclient-dev

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