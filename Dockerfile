FROM python:3.11.3-slim

# Stup ENV variables
ENV GIT_PYTHON_REFRESH=quiet

# Install all OS dependencies for fully functional requirements.txt install
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the python base image may be rebuilt too seldom sometimes (less than once a month)
    # required for psutil python package to install
    python3-dev \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /root/model-training/
COPY requirements.txt .

# Install requirements
RUN mkdir ml_models && python -m pip install --upgrade pip && pip install -r requirements.txt

# Import files
COPY . .

# Train ML model
RUN dvc repro
