FROM python:3.10.9

# Set work directory
WORKDIR /root/model-training/

# Install requirements
RUN python3 -m venv /opt/venv

# Install dependencies:
COPY requirements.txt .
RUN . /opt/venv/bin/activate

RUN python -m pip install --upgrade pip
RUN pip install --no-binary :all: psutil
RUN pip install -r requirements.txt

# Improt files
COPY . .

# Train ML model
RUN dvc repro
