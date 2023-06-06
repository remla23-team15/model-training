import logging
import pickle
import random

import joblib
from sklearn.naive_bayes import GaussianNB

import utils


def train_model(seed, X_train, y_train):
    
    random.seed(seed)

    model = GaussianNB()
    model.fit(X_train, y_train)

    return model


def main():
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    log = logging.getLogger(__name__)
    log.info("-------- CLASSIFIER TRAINING ----------")
    
    params = utils.parse_params()
    SEED: int = params['base']['seed']
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    MODEL_C2_PATH: str = params['train']['model_c2']
    
    # Load the data
    X_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_train.pckl").read_bytes())
    y_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_train.pckl").read_bytes())

    # Model fitting
    log.info("Fit model into a Gaussian classifier...")
    model = train_model(SEED, X_train, y_train)

    # Export model for future usage
    log.info("Exporting NB Classifier to later use in prediction...")
    joblib.dump(model, utils.SCRIPTS_PATH / MODEL_C2_PATH)


if __name__ == "__main__":
    main()
