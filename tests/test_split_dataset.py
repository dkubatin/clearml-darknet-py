import pytest

from clearml_darknet import split_dataset
from clearml_darknet.errors import DatasetError


def test_split_dataset(dataset_path):
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)
    assert len(train) == len(valid)


def test_split_dataset_hierarchy(dataset_hierarchy_path):
    train, valid = split_dataset(dataset_path=dataset_hierarchy_path, ratio=0.5)
    assert len(train) == len(valid)


def test_split_dataset_shuffle_true(dataset_path):
    train_no_shuffle, valid_no_shuffle = split_dataset(dataset_path=dataset_path, ratio=0.5)
    train_shuffle, valid_shuffle = split_dataset(dataset_path=dataset_path, ratio=0.5, shuffle=True)
    assert len(train_shuffle) == len(valid_shuffle)
    assert train_no_shuffle != train_shuffle
    assert valid_no_shuffle != valid_shuffle


def test_split_dataset_failed_dataset_path(dataset_empty_path):
    with pytest.raises(DatasetError):
        _, _ = split_dataset(dataset_path=dataset_empty_path, ratio=0.7)


def test_split_dataset_failed_ration_big(dataset_path):
    with pytest.raises(ValueError):
        _, _ = split_dataset(dataset_path=dataset_path, ratio=1.5)


def test_split_dataset_failed_ration_small(dataset_path):
    with pytest.raises(ValueError):
        _, _ = split_dataset(dataset_path=dataset_path, ratio=-1.5)

