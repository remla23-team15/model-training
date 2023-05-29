FROM python:3.11.3-slim

# Set work directory
WORKDIR /root/model-training/
COPY requirements.txt .

# Install requirements
RUN mkdir ml_models 
RUN mkdir output
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Improt files
COPY . .

# Train ML model
RUN dvc repro