import pandas as pd
import yaml
from pathlib import Path
import utils

import pytest


@pytest.fixture()
def params():
    f = open(utils.SCRIPTS_PATH / "params.yaml", "r")
    params = yaml.safe_load(f)
    f.close()
    yield params


@pytest.fixture()
def dataset_a1_df(params):
    DATASET_A1_PATH: str = params['data_preprocess']['dataset_train']
    dataset_a1_df = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A1_PATH, delimiter="\t", quoting=3)
    yield dataset_a1_df


@pytest.fixture()
def dataset_a2_df(params):
    DATASET_A2_PATH: str = params['eval']['dataset_eval']
    dataset_a2_df = pd.read_csv(utils.SCRIPTS_PATH / DATASET_A2_PATH, delimiter="\t", quoting=3)
    yield dataset_a2_df


# Test for duplicates in the train and test datasets
def test_no_duplicate_reviews(dataset_a1_df, dataset_a2_df):
    assert len(dataset_a1_df['Review']) == dataset_a1_df.shape[0]
    assert len(dataset_a2_df['Review']) == dataset_a2_df.shape[0]


# Test for empty reviews
def test_no_empty_reviews(dataset_a1_df, dataset_a2_df):
    assert all ([len(review) > 0 for review in dataset_a1_df['Review']])
    assert all ([len(review) > 0 for review in dataset_a2_df['Review']])
