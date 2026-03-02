# Use unsloth's official CUDA 12.1 + PyTorch 2.4.0 image
# RTX 2080 Ti has compute capability 7.5, supported by CUDA 12.x
FROM nvidia/cuda:13.0.2-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /code

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev

COPY ./LLM_distill_setup/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available for JupyterLab
EXPOSE 8888
