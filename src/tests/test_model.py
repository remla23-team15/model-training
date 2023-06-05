import pandas as pd
import joblib
import pickle
import random
from sklearn.dummy import DummyClassifier

import pytest

import utils
from train import train_model
from evaluation import evaluate_model


@pytest.fixture()
def trained_model():
    MODEL_C2_PATH: str = "ml_models/c2_Classifier_Sentiment_Model"
    trained_model = joblib.load(utils.SCRIPTS_PATH / MODEL_C2_PATH)
    yield trained_model

@pytest.fixture()
def X_train():
    DESTINATION_DIR: str = "data/processed"
    X_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_train.pckl").read_bytes())
    yield X_train

@pytest.fixture()
def y_train():
    DESTINATION_DIR: str = "data/processed"
    y_train = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_train.pckl").read_bytes())
    yield y_train

@pytest.fixture()
def X_test():
    DESTINATION_DIR: str = "data/processed"
    X_test = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "X_test.pckl").read_bytes())
    yield X_test

@pytest.fixture()
def y_test():
    DESTINATION_DIR: str = "data/processed"
    y_test = pickle.loads((utils.SCRIPTS_PATH / DESTINATION_DIR / "y_test.pckl").read_bytes())
    yield y_test


# Test if the model outperforms a Dummy Classifier with uniform strategy
def test_baseline_uniform(trained_model, X_train, y_train, X_test, y_test):
    original_metrics = evaluate_model(trained_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="uniform")
    dummy_model.fit(X_train, y_train)
    dummy_metrics = evaluate_model(dummy_model, X_test, y_test)

    assert abs(original_metrics["acc"] > dummy_metrics["acc"])
    assert abs(original_metrics["precision"] > dummy_metrics["precision"])
    assert abs(original_metrics["recall"] > dummy_metrics["recall"])
    assert abs(original_metrics["f1"] > dummy_metrics["f1"])


# Test if the model outperforms a Dummy Classifier with most_frequent strategy
def test_baseline_most_frequent(trained_model, X_train, y_train, X_test, y_test):
    original_metrics = evaluate_model(trained_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_metrics = evaluate_model(dummy_model, X_test, y_test)

    # In this case we omit recall as the most_frequent strategy trivially scores 100%
    assert abs(original_metrics["acc"] > dummy_metrics["acc"])
    assert abs(original_metrics["precision"] > dummy_metrics["precision"])
    assert abs(original_metrics["f1"] > dummy_metrics["f1"])


# Test for non deterministic robustness
def test_non_deterministic_robustness(trained_model, X_train, y_train, X_test, y_test):
    original_metrics = evaluate_model(trained_model, X_test, y_test)

    for _ in range(3):
        seed = random.randint(0, 9999)
        model_variant = train_model(seed, X_train, y_train)
        variant_metrics = evaluate_model(model_variant, X_test, y_test)

        assert abs(original_metrics["acc"] - variant_metrics["acc"] <= 0.1)
        assert abs(original_metrics["precision"] - variant_metrics["precision"] <= 0.1)
        assert abs(original_metrics["recall"] - variant_metrics["recall"] <= 0.1)
        assert abs(original_metrics["f1"] - variant_metrics["f1"] <= 0.1)
