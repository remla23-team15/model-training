# model-training
Contains the ML training pipeline.

## How To Run It

#### Clone

Clone this repo to your local machine using 
```
git clone https://github.com/remla23-team15/model-training.git
```

#### Create Virtual Environment (venv)
Move to  the application folder and run in your terminal:
```
# Create virtualenv, make sure to use python >= 3.8
$ virtualenv -p python3 venv
# Activate venv
$ source venv/bin/activate
```
Alternatively:
* Open the project with PyCharm (either Pro or CE)  or your favorite Python IDE
* Select python (>= 3.8) as project interpreter

#### Add Google Drive Authentication
Using the file `gdrive-auth-template.json`, create a new file called `gdrive-auth.json` and replace the necessary credentials with your remote Google Cloud application linked to a Google Drive storage.

#### Install Requirements
Move to  the application folder and run in your terminal:
```
# Install python packages
pip install -r requirements.txt

# Get the dataset from the DVC remote storage
dvc pull
```

#### Run
Move to  the application folder and run in your terminal:
```
dvc repro
```

#### Test
Move to  the application folder and run in your terminal:
```
# Install packages needed for tests
pip install -r requirements-ci.txt

# Run tests
pytest

# Run tests with coverage and generate reports (will be added to the reports folder)
coverage run -m pytest --junitxml=reports/junit/junit.xml

# To print the branch coverage detials in the terminal
coverage report
# To generate a full XML report
coverage xml
# To generate a full HTML report
coverage html
# To generate a full JSON report
coverage json
```

#### Lint
To assess the code quality using PyLint with dslinter plugin, execute the following commands:
``` 
# Install CI requirements 
pip install -r requirements-ci.txt

# Execute pylint on the src files
pylint pylint src
```

#### Upload The ML Models
Move to  the application folder and run in your terminal:
```
python upload_ml_models.py
```

The models will be uploaded in the remote repository: https://liv.nl.tab.digital/s/TyPqR5HCjExqNQq

## Docker
To build a Docker image of the application, you can open the terminal (in the application folder) and run:
```shell script
docker build -t ghcr.io/remla23-team15/model-training:VERSION .
```

**VERSION indicates the version that you want to apply to the Docker image, for example 1.0.0, latest or so.**

## Contributors

REMLA 2023 - Group 15
