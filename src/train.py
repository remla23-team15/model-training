""" Description """

import logging
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import pickle
import utils
import random


def main():
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    log = logging.getLogger(__name__)

    params = utils.parse_params()
    SEED: int = params['base']['seed']
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    MODEL_C2_PATH: str = params['train']['model_c2']

    random.seed(SEED)
    
    # Load the data
    X_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_train.pckl").read_bytes())
    y_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_train.pckl").read_bytes())

    # Model fitting
    log.info("Fit model into a Gaussian classifier...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Export model for future usage
    log.info("Exporting NB Classifier to later use in prediction...")
    joblib.dump(model, utils.SCRIPTS_PATH / MODEL_C2_PATH)


if __name__=="__main__":
    main()