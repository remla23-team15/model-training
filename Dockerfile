FROM python:3.11.3-slim

# Stup ENV variables
ENV GIT_PYTHON_REFRESH=quiet

ARG gcloud_project_id
ENV GCLOUD_PROJECT_ID gcloud_project_id

ARG gcloud_private_key_id
ENV GCLOUD_PRIVATE_KEY_ID gcloud_private_key_id

ARG gcloud_private_key
ENV GCLOUD_PRIVATE_KEY gcloud_private_key

ARG gcloud_client_email
ENV GCLOUD_CLIENT_EMAIL gcloud_client_email

ARG gcloud_client_id
ENV GCLOUD_CLIENT_ID gcloud_client_id

ARG gcloud_cert_url
ENV GCLOUD_CERT_URL gcloud_cert_url

# Install all OS dependencies for fully functional requirements.txt install
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the python base image may be rebuilt too seldom sometimes (less than once a month)
    # required for psutil python package to install
    python3-dev \
    gettext-base \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /root/model-training/
COPY requirements.txt .

# Install requirements
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Import files
COPY . .

# Setup GDrive credentials
RUN envsubst < gdrive-auth-template.json > gdrive-auth.json

# Train ML model
RUN dvc pull && dvc repro

# Upload the ML models to the remote repository
RUN python upload_ml_models.py
