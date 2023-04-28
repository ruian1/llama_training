# First stage: base image with Python installed
FROM python:3.10.10 AS python-build

# Second stage: image with CUDA and Python
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04 as cuda-build

ENV DEBIAN_FRONTEND=noninteractive

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y emacs &&\ 
    apt-get install -y --no-install-recommends apt-utils  && \
    rm -rf /var/lib/apt/lists/*

# Copy Python binary, pip, and other necessary files from the python-build stage
COPY --from=python-build /usr/local /usr/local

# Remove existing python symlink and create a new one for python3.10
RUN rm /usr/local/bin/python && ln -s /usr/local/bin/python3.10 /usr/local/bin/python

# Remove existing pip3 symlink and create a new one for pip
RUN rm /usr/local/bin/pip3 && ln -s /usr/local/bin/pip /usr/local/bin/pip3

# Update PATH to include Python and pip
ENV PATH="/usr/local/bin:${PATH}"

# Copy Python site-packages from the python-build stage
COPY --from=python-build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Set PYTHONPATH to include the site-packages
ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages:${PYTHONPATH}"

# Install any additional dependencies
# RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install git+https://github.com/huggingface/peft
RUN pip install git+https://github.com/huggingface/transformers

# Clone the llama_training repository
RUN git clone https://github.com/ruian1/llamma_training.git && \
    cd llamma_training && \
    pip install -r requirements.txt

# Set the working directory
WORKDIR /projects

# Define the command to run the training script
# CMD ["python", "train.py"]