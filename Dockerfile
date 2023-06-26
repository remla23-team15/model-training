# Build stage to install dependencies
FROM python:3.11.3-slim as build

# Setup ENV variables
ENV GIT_PYTHON_REFRESH=quiet

# Install all OS dependencies for fully functional requirements.txt install
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the python base image may be rebuilt too seldom sometimes (less than once a month)
    # required for psutil python package to install
    python3-dev \
    git \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /root/model-training/
COPY requirements.txt .

# Install requirements
RUN python -m pip install --upgrade pip && pip install --user -r requirements.txt

# Build app image
FROM python:3.11.3-slim as app

# Setup ENV variables
ENV GIT_PYTHON_REFRESH=quiet

# Set work directory
WORKDIR /root/model-training/

# Import files
COPY --from=build /root/.local /root/.local
COPY . .

# Setup Python path
ENV PATH=/root/.local/bin:$PATH

# Train ML model
RUN dvc pull && dvc repro

# Upload the ML models to the remote repository
RUN python upload_ml_models.py
