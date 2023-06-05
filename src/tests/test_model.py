import pandas as pd
import joblib
import pickle
import yaml
from pathlib import Path
import random
from sklearn.dummy import DummyClassifier

import pytest

import utils
from train import train_model
from evaluation import evaluate_model


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
