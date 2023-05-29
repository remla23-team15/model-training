""" Description """

import joblib
import random
import utils
import yaml
import logging
import pickle
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score)

def main():
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    log = logging.getLogger(__name__)
    log.info("-------- CLASSIFIER PREDICTIONS ----------")
    
    params = utils.parse_params()
    SEED: int = params['base']['seed']
    MODEL_C2_PATH: str = params['train']['model_c2']
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    METRICS_PATH: str = params['eval']['metrics_file']

    random.seed(SEED)

    # Import model
    model = joblib.load(utils.SCRIPTS_PATH / MODEL_C2_PATH)

    # Import data 
    X_test = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_test.pckl").read_bytes())
    y_test = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_test.pckl").read_bytes())

    # Evaluate performance
    log.info("Model performance evaluation...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {}
    metrics["acc"] = float(accuracy_score(y_test, y_pred))
    metrics["precision"] = float(precision_score(y_test, y_pred))
    metrics["recall"] = float(recall_score(y_test, y_pred))
    metrics["f1"] = float(f1_score(y_test, y_pred))

    # Save metrics
    with open(utils.SCRIPTS_PATH / METRICS_PATH, 'w') as file:
        yaml.dump(metrics, file, default_flow_style=False)


if __name__=="__main__":
    main()