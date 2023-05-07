import logging

from scripts.data_classification import classifier_training
from scripts.data_predictions import data_predictions
from scripts.data_preprocessing import data_preprocess

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info("-------- DATA PRE-PROCESSING ----------")
    X, y = data_preprocess()

    log.info("-------- CLASSIFIER TRAINING ----------")
    classifier_training(X, y)

    log.info("-------- CLASSIFIER PREDICTIONS ----------")
    data_predictions()
