import re

import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def remove_stopwords(dataset):
    """
    Remove stopwords from the dataset.

    :param dataset: The input dataset.
    :return: The cleaned corpus.
    """
    nltk.download("stopwords")
    ps = PorterStemmer()

    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")

    corpus = []

    for i in range(0, dataset.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = " ".join(review)
        corpus.append(review)

    return corpus
