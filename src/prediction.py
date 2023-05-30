import logging
import os
import pickle
import random

import joblib
import pandas as pd

import utils


def main():

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    log = logging.getLogger(__name__)

    params = utils.parse_params()
    SEED: int = params['base']['seed']
    DATASET_A2_PATH: str = params['eval']['dataset_eval']
    MODEL_C1_PATH: str = params['data_preprocess']['model_c1']
    MODEL_C2_PATH: str = params['train']['model_c2']
    MODEL_C3_PATH: str = params['eval']['model_c3']

    random.seed(SEED)

    # Open the dataset
    log.info("Opening the dataset...")
    dataset = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A2_PATH, delimiter="\t", quoting=3)
    log.info(dataset.tail())

    # Clean data
    log.info("Data cleaning...")

    corpus = utils.remove_stopwords(dataset)

    # Load pickle
    log.info("Loading BoW dictionary...")
    cv = pickle.load(open(utils.SCRIPTS_PATH / MODEL_C1_PATH, "rb"))

    X_fresh = cv.transform(corpus).toarray()
    log.info(X_fresh.shape)

    # Load model
    model = joblib.load(utils.SCRIPTS_PATH / MODEL_C2_PATH)

    # perform prediction
    y_pred = model.predict(X_fresh)
    log.info(y_pred)

    dataset['predicted_label'] = y_pred.tolist()
    log.info(dataset[dataset['predicted_label'] == 1])

    # Create output folder if it does not exist yet
    os.makedirs(utils.SCRIPTS_PATH / "outputs", exist_ok=True)

    # Store results
    dataset.to_csv(utils.SCRIPTS_PATH / MODEL_C3_PATH, sep="\t", encoding="UTF-8", index=False)

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
        prediction = model.predict([processed_input])[0]

        log.info(f"{review} ---> {prediction_map[prediction]}")


if __name__ == "__main__":
    main()
