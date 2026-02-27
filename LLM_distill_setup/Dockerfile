# Use unsloth's official CUDA 12.1 + PyTorch 2.4.0 image
# RTX 2080 Ti has compute capability 7.5, supported by CUDA 12.x
FROM unsloth/unsloth-env-cu121-torch240:latest

# Set the working directory in the container
WORKDIR /code

COPY ./LLM_distill_setup/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available for JupyterLab
EXPOSE 8888
