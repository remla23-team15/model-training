import logging

from scripts.data_classification import classifier_training
from scripts.data_preprocessing import data_preprocess

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")


if __name__ == "__main__":
    print("-------- DATA PRE-PROCESSING ----------")
    X, y = data_preprocess()

    print("\n-------- CLASSIFIER TRAINING ----------")
    classifier_training(X, y)

    print("\n-------- CLASSIFIER PREDICTIONS ----------")
