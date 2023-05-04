import logging
import pickle
import re

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from scripts import SCRIPTS_PATH
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

log = logging.getLogger(__name__)


def data_preprocess():
    """
    Preprocess the restaurant reviews sentiment
    """
    # Open the dataset
    log.info("Opening the dataset...")

    dataset = pd.read_csv(f"{SCRIPTS_PATH}/data/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3)

    log.info(dataset.shape)
    log.info(dataset.head())

    # Data pre-processing
    log.info("Pre-process the dataset...")
    nltk.download("stopwords")
    ps = PorterStemmer()

    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")

    corpus = []

    for i in range(0, 900):
        review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = " ".join(review)
        corpus.append(review)

    # Transform data
    log.info("Transforming the dataset...")
    cv = CountVectorizer(max_features=1420)

    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Save pickle object
    log.info("Saving BoW dictionary to later use in prediction...")
    pickle.dump(
        cv,
        open(
            f"{SCRIPTS_PATH}/ml_models/c1_BoW_Sentiment_Model.pkl",
            "wb")
    )

    return X, y


if __name__ == "__main__":
    X_data, y_data = data_preprocess()
    log.info(X_data)
    log.info(y_data)
