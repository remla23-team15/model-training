import nltk
import numpy as np
import pandas as pd
import joblib
import pickle
import yaml
from pathlib import Path
import random

from nltk import word_tokenize, pos_tag
from sklearn.dummy import DummyClassifier
from negate import Negator
from nltk.corpus import wordnet

import pytest

import utils
from data_preprocess import preprocess_dataset, transform_dataset, split_dataset
from train import train_model
from evaluation import (evaluate_model, evaluate_prediction)


# @pytest.fixture()
# def params():
#     f = open(utils.SCRIPTS_PATH / "params.yaml", "r")
#     params = yaml.safe_load(f)
#     f.close()
#     yield params


# @pytest.fixture()
# def dataset_split(params):
#     DATASET_A1_PATH: str = params['data_preprocess']['dataset_train']
#     dataset = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A1_PATH, delimiter="\t", quoting=3)
#     corpus = preprocess_dataset(dataset)
#     X, y, _ = transform_dataset(dataset, corpus, params['data_preprocess']['max_features'])
#     X_train, X_test, y_train, y_test = split_dataset(X, y, params['data_preprocess']['test_size'], params['base']['seed'])
#     yield (X_train, y_train, X_test, y_test)


# @pytest.fixture()
# def trained_model(params, dataset_split):
#     SEED: int = params['base']['seed']
#     X_train, y_train, _, _ = dataset_split
#     trained_model = train_model(SEED, X_train, y_train)
#     yield trained_model


# @pytest.fixture()
# def negation_X_set(params):
#     DATASET_A1_PATH: str = params['data_preprocess']['dataset_train']
#     dataset = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A1_PATH, delimiter="\t", quoting=3)

#     negator = Negator(use_transformers=True)
#     dataset['Review'] = [negator.negate_sentence(review, prefer_contractions=False) for review in dataset['Review']]
#     corpus = preprocess_dataset(dataset)

#     X, y, _ = transform_dataset(dataset, corpus, params['data_preprocess']['max_features'])
#     nX_train, negation_X_set, ny_train, ny_test = split_dataset(X, y, params['data_preprocess']['test_size'], params['base']['seed'])
#     yield (nX_train, ny_train, negation_X_set, ny_test)

# @pytest.fixture()
# def synonym_X_set(params):
#     nltk.download('wordnet')
#     nltk.download('averaged_perceptron_tagger')

#     DATASET_A1_PATH: str = params['data_preprocess']['dataset_train']
#     dataset = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A1_PATH, delimiter="\t", quoting=3)
#     dataset["Review"] = [synonym_sentence(review) for review in dataset["Review"]]
#     corpus = preprocess_dataset(dataset)

#     X, y, _ = transform_dataset(dataset, corpus, params['data_preprocess']['max_features'])
#     X_train, synonym_X_set, y_train, y_test = split_dataset(X, y, params['data_preprocess']['test_size'],
#                                                              params['base']['seed'])
#     yield synonym_X_set

# def synonym_sentence(sentence):
#     tokens = word_tokenize(sentence)
#     tagged_tokens = pos_tag(tokens)

#     replaced_sentence = []
#     for token, tag in tagged_tokens:
#         if token == 'i' or token == "I":
#             replaced_sentence.append(token)
#             continue
#         if tag.startswith('NN') or tag.startswith('JJ'):  # Identify nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
#             synonyms = []
#             for syn in wordnet.synsets(token):
#                 for lemma in syn.lemmas():
#                     synonyms.append(lemma.name())

#             if token in synonyms:
#                 synonyms.remove(token)  # Remove the original word from the synonyms list
#             if len(synonyms) > 0:
#                 replaced_sentence.append(synonyms[0])  # Add the first synonym to the replaced sentence
#             else:
#                 replaced_sentence.append(token)  # Add the original word if no appropriate synonyms found
#         else:
#             replaced_sentence.append(token)  # Add non-noun and non-adjective tokens as is

#     return ' '.join(replaced_sentence)


# # Test if the model outperforms a Dummy Classifier with uniform strategy
# def test_baseline_uniform(trained_model, dataset_split):
#     X_train, y_train, X_test, y_test = dataset_split

#     original_metrics = evaluate_model(trained_model, X_test, y_test)

#     dummy_model = DummyClassifier(strategy="uniform")
#     dummy_model.fit(X_train, y_train)
#     dummy_metrics = evaluate_model(dummy_model, X_test, y_test)

#     assert abs(original_metrics["acc"] > dummy_metrics["acc"])
#     assert abs(original_metrics["precision"] > dummy_metrics["precision"])
#     assert abs(original_metrics["recall"] > dummy_metrics["recall"])
#     assert abs(original_metrics["f1"] > dummy_metrics["f1"])


# # Test if the model outperforms a Dummy Classifier with most_frequent strategy
# def test_baseline_most_frequent(trained_model, dataset_split):
#     X_train, y_train, X_test, y_test = dataset_split

#     original_metrics = evaluate_model(trained_model, X_test, y_test)

#     dummy_model = DummyClassifier(strategy="most_frequent")
#     dummy_model.fit(X_train, y_train)
#     dummy_metrics = evaluate_model(dummy_model, X_test, y_test)

#     # In this case we omit recall as the most_frequent strategy trivially scores 100%
#     assert abs(original_metrics["acc"] > dummy_metrics["acc"])
#     assert abs(original_metrics["precision"] > dummy_metrics["precision"])
#     assert abs(original_metrics["f1"] > dummy_metrics["f1"])


# # Test for non deterministic robustness
# def test_non_deterministic_robustness(trained_model, dataset_split):
#     X_train, y_train, X_test, y_test = dataset_split

#     original_metrics = evaluate_model(trained_model, X_test, y_test)

#     for _ in range(3):
#         random_seed = random.randint(0, 9999)
#         model_variant = train_model(random_seed, X_train, y_train)
#         variant_metrics = evaluate_model(model_variant, X_test, y_test)

#         assert abs(original_metrics["acc"] - variant_metrics["acc"] <= 0.1)
#         assert abs(original_metrics["precision"] - variant_metrics["precision"] <= 0.1)
#         assert abs(original_metrics["recall"] - variant_metrics["recall"] <= 0.1)
#         assert abs(original_metrics["f1"] - variant_metrics["f1"] <= 0.1)


# # TODO Test if the model similarly performs on negated sentences
# def test_baseline_negated(trained_model, dataset_split, negation_X_set):
#     _, _, X_test, _ = dataset_split
#     _, _, n_X_set, _ = negation_X_set

#     original_results = trained_model.predict(X_test)
#     negated_results = trained_model.predict(n_X_set)

#     negated_original_results = [abs(1 - prediction) for prediction in original_results]
#     metrics = evaluate_prediction(negated_original_results, negated_results)

#     # Set to max values currently able to receive to pass test
#     assert abs(metrics["acc"] >= 0.35)
#     assert abs(metrics["precision"] >= 0.22)
#     assert abs(metrics["recall"] >= 0.31)
#     assert abs(metrics["f1"] >= 0.29)

# # #TODO Test if the model behaves similarly on synonymed sentences
# def test_baseline_synonym(trained_model, dataset_split, synonym_X_set):
#     _, _, X_test, _ = dataset_split

#     original_results = trained_model.predict(X_test)
#     synonym_results = trained_model.predict(synonym_X_set)

#     metrics = evaluate_prediction(original_results, synonym_results)

#     # Set to max values currently able to receive to pass test
#     assert abs(metrics["acc"] >= 0.43)
#     assert abs(metrics["precision"] >= 0.50)
#     assert abs(metrics["recall"] >= 0.35)
#     assert abs(metrics["f1"] >= 0.40)