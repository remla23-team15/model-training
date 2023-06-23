""" data_preprocess.py """
import logging
import pickle
import random

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import utils


def preprocess_dataset(dataset):
    """ Preprocess: remove stopwords """
    corpus = utils.remove_stopwords(dataset)
    return corpus


def transform_dataset(dataset, corpus, MAX_FEATURES):
    """ Transform dataset ... """
    cv = CountVectorizer(max_features=MAX_FEATURES)

    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    return X, y, cv


def split_dataset(X, y, TEST_SIZE, SEED):
    """ Split dataset in train and test sets. """
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        random_state=SEED)
    return X_train, X_test, y_train, y_test


def main():
    """ Main """

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    log = logging.getLogger(__name__)
    log.info("-------- DATA PRE-PROCESSING ----------")

    params = utils.parse_params()
    SEED: int = params['base']['seed']
    DATASET_A1_PATH: str = params['data_preprocess']['dataset_train']
    MAX_FEATURES: int = params['data_preprocess']['max_features']
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    TEST_SIZE: float = params['data_preprocess']['test_size']
    MODEL_C1_PATH: str = params['data_preprocess']['model_c1']

    random.seed(SEED)

    # Open the dataset
    log.info("Opening the dataset...")
    dataset = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A1_PATH, delimiter="\t", quoting=3)

    log.info(dataset.shape)
    log.info(dataset.head())

    # Data pre-processing
    log.info("Pre-process the dataset...")
    corpus = preprocess_dataset(dataset)

    # Transform data
    log.info("Transforming the dataset...")
    print(f'MAX_FEATURES: {MAX_FEATURES}')
    X, y, cv = transform_dataset(dataset, corpus, MAX_FEATURES)

    # Split dataset
    log.info("Dividing dataset into training and test set...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE, SEED)

    # Save pickle object
    log.info("Saving BoW dictionary to later use in prediction...")
    pickle.dump(cv, open(utils.SCRIPTS_PATH / MODEL_C1_PATH, "wb"))

    # Save data into pickle objects
    pickle.dump(X, open(utils.SCRIPTS_PATH / DESTINATION_DIR / "X.pckl", "wb"))
    pickle.dump(X_train, open(utils.SCRIPTS_PATH / DESTINATION_DIR / "X_train.pckl", "wb"))
    pickle.dump(X_test, open(utils.SCRIPTS_PATH / DESTINATION_DIR / "X_test.pckl", "wb"))

    pickle.dump(y, open(utils.SCRIPTS_PATH / DESTINATION_DIR / "y.pckl", "wb"))
    pickle.dump(y_train, open(utils.SCRIPTS_PATH / DESTINATION_DIR / "y_train.pckl", "wb"))
    pickle.dump(y_test, open(utils.SCRIPTS_PATH / DESTINATION_DIR / "y_test.pckl", "wb"))


if __name__ == "__main__":
    main()
