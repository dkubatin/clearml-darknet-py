import os

import pytest

from clearml_darknet import Darknet
from clearml_darknet.utils import split_dataset


def test_darknet_init_success(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'classes.txt')
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)

    darknet_exec = darknet.strpath
    config_path = './examples/data/yolov3-tiny.cfg'
    Darknet(
        task=task,
        darknet_exec=darknet_exec,
        config_path=config_path,
        classes_path=classes,
        train=train,
        valid=valid,
    )


def test_darknet_init_failed_darknet_exec_missing(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'classes.txt')
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)

    darknet_exec = os.path.join(darknet.strpath, 'darknet-test')
    config_path = './examples/data/yolov3-tiny.cfg'

    with pytest.raises(FileNotFoundError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
        )


def test_darknet_init_failed_config_path_missing(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'classes.txt')
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)

    darknet_exec = darknet.strpath
    config_path = os.path.join(darknet.dirname, 'darknet91.cfg')

    with pytest.raises(FileNotFoundError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
        )


def test_darknet_init_failed_weight_path_missing(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'classes.txt')
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)

    darknet_exec = darknet.strpath
    config_path = './examples/data/yolov3-tiny.cfg'
    weight_path = os.path.join(darknet.strpath, 'yolov3-tiny.weights')

    with pytest.raises(FileNotFoundError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
            weight_path=weight_path
        )


def test_darknet_init_failed_classes_path_missing(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'test-classes.txt')
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)

    darknet_exec = darknet.strpath
    config_path = './examples/data/yolov3-tiny.cfg'

    with pytest.raises(FileNotFoundError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
        )


def test_darknet_init_failed_classes_empty(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'test-classes-missing.txt')
    train, valid = split_dataset(dataset_path=dataset_path, ratio=0.5)

    darknet_exec = darknet.strpath
    config_path = './examples/data/yolov3-tiny.cfg'

    with pytest.raises(ValueError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
        )


def test_darknet_init_failed_train_empty(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'classes.txt')
    train, valid = [], ['0.jpg']

    darknet_exec = darknet.strpath
    config_path = './examples/data/yolov3-tiny.cfg'

    with pytest.raises(ValueError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
        )


def test_darknet_init_failed_valid_empty(dataset_path, task, darknet):
    classes = os.path.join(dataset_path, 'classes.txt')
    train, valid = ['0.jpg'], []

    darknet_exec = darknet.strpath
    config_path = './examples/data/yolov3-tiny.cfg'

    with pytest.raises(ValueError):
        Darknet(
            task=task,
            darknet_exec=darknet_exec,
            config_path=config_path,
            classes_path=classes,
            train=train,
            valid=valid,
        )


def test_darknet_init_failed_reading_config():
    # ToDo: add test
    pass


def test_darknet_init_failed_reading_parameter_config():
    # ToDo: add test
    pass
