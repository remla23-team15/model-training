import logging

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from scripts import SCRIPTS_PATH
from scripts.data_preprocessing import data_preprocess

log = logging.getLogger(__name__)


def classifier_training(X, y):
    """
    Train the classifier.

    :param X: The restaurant reviews.
    :param y: The labelled sentiments.
    """
    # Split dataset
    log.info("Dividing dataset into training and test set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Model fitting
    log.info("Fit model into a Gaussian classifier...")
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Export model for future usage
    log.info("Exporting NB Classifier to later use in prediction...")
    joblib.dump(classifier, f"{SCRIPTS_PATH}/ml_models/c2_Classifier_Sentiment_Model")

    # Evaluate performance
    log.info("Model performance evaluation...")
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    log.info(cm)

    log.info(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    X_data, y_data = data_preprocess()
    classifier_training(X_data, y_data)
