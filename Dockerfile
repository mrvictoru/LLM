# Use an official NVIDIA CUDA runtime image
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing \
    build-essential pkg-config libglib2.0-0 \
    default-libmysqlclient-dev\
    cmake git \
    python3.11 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev
    
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Set the working directory in the container
WORKDIR /code

# Copy the code directory contents into the container at /code
#COPY ./code /code

COPY ./requirements.txt ./

# Set the CMAKE_ARGS environment variable
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLABS=1

RUN pip install -r requirements.txt

# Attempt to locate libcuda.so.1 and create a symbolic link in /usr/lib if found
RUN find /usr/local/cuda-12.3 /usr/local/cuda -name 'libcuda.so.1' -exec ln -s {} /usr/lib/libcuda.so.1 \; || \
    echo "libcuda.so.1 not found"

# Set environment variables to ensure the linker finds libcuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Make port 8888 available to the world outside this container
EXPOSE 8888