import re
from argparse import ArgumentParser
from pathlib import Path

import nltk
import yaml
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

SCRIPTS_PATH = Path(__file__).resolve().parent.parent


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


def parse_params():
    """
    Get parameters from params.yaml file
    """
    args_parser = ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)

    return params
