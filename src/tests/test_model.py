import nltk
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
from train import train_model
from evaluation import (evaluate_model, evaluate_prediction)


@pytest.fixture()
def params():
    ROOT = Path(__file__).resolve().parent.parent.parent
    f = open(ROOT / "params.yaml", "r")
    params = yaml.safe_load(f)
    f.close()
    yield params


@pytest.fixture()
def SEED(params):
    SEED: int = params['base']['seed']
    yield SEED


@pytest.fixture()
def X_train(params):
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    X_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_train.pckl").read_bytes())
    yield X_train


@pytest.fixture()
def y_train(params):
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    y_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_train.pckl").read_bytes())
    yield y_train


@pytest.fixture()
def X_test(params):
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    X_test = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_test.pckl").read_bytes())
    yield X_test


@pytest.fixture()
def y_test(params):
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    y_test = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_test.pckl").read_bytes())
    yield y_test


@pytest.fixture()
def dataset_negation_mut():
    negator = Negator(use_transformers=True)
    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    negation_X_set = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_test.pckl").read_bytes())
    negation_X_set = map(negator.negate_sentence, negation_X_set)
    yield negation_X_set

@pytest.fixture()
def dataset_synonym_mut():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    DESTINATION_DIR: str = params['data_preprocess']['destination_directory']
    synonym_X_set = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_test.pckl").read_bytes())
    synonym_X_set = map(synonym_sentence, synonym_X_set)
    yield synonym_X_set

def synonym_sentence(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)

    replaced_sentence = []
    for token, tag in tagged_tokens:
        if token == 'i' or token == "I":
            replaced_sentence.append(token)
            continue
        if tag.startswith('NN') or tag.startswith('JJ'):  # Identify nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
            synonyms = []
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())

            if token in synonyms:
                synonyms.remove(token)  # Remove the original word from the synonyms list
            if len(synonyms) > 0:
                replaced_sentence.append(synonyms[0])  # Add the first synonym to the replaced sentence
            else:
                replaced_sentence.append(token)  # Add the original word if no appropriate synonyms found
        else:
            replaced_sentence.append(token)  # Add non-noun and non-adjective tokens as is

    return ' '.join(replaced_sentence)


# Test if the model outperforms a Dummy Classifier with uniform strategy
def test_baseline_uniform(SEED, X_train, y_train, X_test, y_test):
    trained_model = train_model(SEED, X_train, y_train)

    original_metrics = evaluate_model(trained_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="uniform")
    dummy_model.fit(X_train, y_train)
    dummy_metrics = evaluate_model(dummy_model, X_test, y_test)

    assert abs(original_metrics["acc"] > dummy_metrics["acc"])
    assert abs(original_metrics["precision"] > dummy_metrics["precision"])
    assert abs(original_metrics["recall"] > dummy_metrics["recall"])
    assert abs(original_metrics["f1"] > dummy_metrics["f1"])


# Test if the model outperforms a Dummy Classifier with most_frequent strategy
def test_baseline_most_frequent(SEED, X_train, y_train, X_test, y_test):
    trained_model = train_model(SEED, X_train, y_train)

    original_metrics = evaluate_model(trained_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_metrics = evaluate_model(dummy_model, X_test, y_test)

    # In this case we omit recall as the most_frequent strategy trivially scores 100%
    assert abs(original_metrics["acc"] > dummy_metrics["acc"])
    assert abs(original_metrics["precision"] > dummy_metrics["precision"])
    assert abs(original_metrics["f1"] > dummy_metrics["f1"])


# Test for non deterministic robustness
def test_non_deterministic_robustness(SEED, X_train, y_train, X_test, y_test):
    trained_model = train_model(SEED, X_train, y_train)

    original_metrics = evaluate_model(trained_model, X_test, y_test)

    for _ in range(3):
        random_seed = random.randint(0, 9999)
        model_variant = train_model(random_seed, X_train, y_train)
        variant_metrics = evaluate_model(model_variant, X_test, y_test)

        assert abs(original_metrics["acc"] - variant_metrics["acc"] <= 0.1)
        assert abs(original_metrics["precision"] - variant_metrics["precision"] <= 0.1)
        assert abs(original_metrics["recall"] - variant_metrics["recall"] <= 0.1)
        assert abs(original_metrics["f1"] - variant_metrics["f1"] <= 0.1)


# Test if the model similarly performs on negated sentences
# Performance should drop no less than 20% (Mainly caused by incorrect negation of semantic)
def test_baseline_negated(SEED, X_train, y_train, X_test, negation_X_set, y_test):
    trained_model = train_model(SEED, X_train, y_train)

    original_results = trained_model.predict(X_test)
    negated_results = train_model.predict(negation_X_set)

    negated_original_results = [not prediction for prediction in original_results]
    metrics = evaluate_prediction(negated_original_results, negated_results)

    assert abs(metrics["acc"] >= 0.80)
    assert abs(metrics["precision"] >= 0.80)
    assert abs(metrics["recall"] >= 0.80)
    assert abs(metrics["f1"] >= 0.80)

# Test if the model behaves similarly on synonymed sentences
def test_baseline_synonym(SEED, X_train, y_train, X_test, synonym_X_set, y_test):
    trained_model = train_model(SEED, X_train, y_train)

    original_results = trained_model.predict(X_test)
    synonym_results = train_model.predict(synonym_X_set)

    assert original_results == synonym_results