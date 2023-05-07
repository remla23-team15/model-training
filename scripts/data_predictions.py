import logging
import pickle
import os

import joblib
import pandas as pd

from scripts import SCRIPTS_PATH
from scripts.utils import remove_stopwords

log = logging.getLogger(__name__)


def data_predictions():
    """
    Predict results using the trained classifier.
    """
    # Open the dataset
    log.info("Opening the dataset...")

    dataset = pd.read_csv(f"{SCRIPTS_PATH}/data/a2_RestaurantReviews_FreshDump.tsv", delimiter="\t", quoting=3)

    log.info(dataset.tail())

    # Clean data
    log.info("Data cleaning...")

    corpus = remove_stopwords(dataset)

    # Load pickle
    log.info("Loading BoW dictionary...")

    cv = pickle.load(open(f"{SCRIPTS_PATH}/ml_models/c1_BoW_Sentiment_Model.pkl", "rb"))

    X_fresh = cv.transform(corpus).toarray()
    log.info(X_fresh.shape)

    # Perform predictions
    log.info("Predictions (via sentiment classifier)...")

    classifier = joblib.load(f"{SCRIPTS_PATH}/ml_models/c2_Classifier_Sentiment_Model")
    y_pred = classifier.predict(X_fresh)
    log.info(y_pred)

    dataset['predicted_label'] = y_pred.tolist()
    log.info(dataset[dataset['predicted_label'] == 1])

    # Create output folder if it does not exist yet
    os.makedirs(f"{SCRIPTS_PATH}/output", exist_ok=True)

    # Store results
    dataset.to_csv(
        f"{SCRIPTS_PATH}/output/c3_Predicted_Sentiments_Fresh_Dump.tsv",
        sep="\t",
        encoding="UTF-8",
        index=False
    )

    # Try with single inputs
    log.info("Predicting single inputs...")

    prediction_map = {
        0: "negative",
        1: "positive",
    }

    reviews = [
        "We are so glad we found this place.",
        "I'm not sure I will ever come back.",
        "Loved it...friendly servers, great food, wonderful and imaginative menu.!",
    ]

    for review in reviews:
        processed_input = cv.transform([review]).toarray()[0]
        prediction = classifier.predict([processed_input])[0]

        log.info(f"{review} ---> {prediction_map[prediction]}")


if __name__ == "__main__":
    data_predictions()
