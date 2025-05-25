# Use the official Ubuntu 22.04 (Jammy) as the base image
FROM ubuntu:24.04

# Set environment variables to avoid user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=llama
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN getent group $USER_GID 2>&1 >/dev/null || groupadd --gid $USER_GID $USERNAME 
# Check if "ubuntu" user exists, delete it if it does, then create the desired user
RUN if getent passwd ubuntu > /dev/null 2>&1; then \
        userdel -r ubuntu && \
        echo "Deleted existing ubuntu user"; \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME -G video && \
    echo "Created new user $USERNAME"

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    sudo \
    git-all \
    curl \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
#    python3-pip \
    python-is-python3 \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random
 # PIP_NO_CACHE_DIR=off \
  #PIP_DISABLE_PIP_VERSION_CHECK=on \
  #PIP_DEFAULT_TIMEOUT=100 \

# Set environment variables for NVM
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 22.14.0
ENV NVM_VERSION 0.40.2


# Install NVM
RUN mkdir $NVM_DIR && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v${NVM_VERSION}/install.sh | bash

# Load NVM and install Node.js
RUN bash -c "source $NVM_DIR/nvm.sh && \
    nvm install $NODE_VERSION && \
    nvm alias default $NODE_VERSION && \
    nvm use default"

# Add Node.js and npm to PATH
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Verify installation
RUN node -v
RUN npm -v

RUN npm install -g nodejs

USER $USERNAME
ARG WORKSPACE
ARG VENV=/home/$USERNAME/.venv

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN echo 'eval "$(uv generate-shell-completion bash)"' >> /home/$USERNAME/.bashrc
# Copy requirements files
#COPY requirements-core.txt ${WORKSPACE}/

# Create and activate virtual environment, install Python dependencies
RUN python3 -m venv ${VENV} --system-site-packages --symlinks \
    && . ${VENV}/bin/activate 

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python3", "app.py"]