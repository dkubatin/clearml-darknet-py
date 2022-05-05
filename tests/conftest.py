import shutil

import pytest
from unittest import mock

from clearml import Task


@pytest.fixture
def dataset_path(tmpdir):
    tmp_path = tmpdir.mkdir("dataset")
    for i in range(0, 100):
        img_path = tmp_path / f"{i}.jpg"
        img_path.write('...')
        txt_path = tmp_path / f"{i}.txt"
        txt_path.write('..')
    classes_txt_path = tmp_path / "classes.txt"
    classes_txt_path.write('class1\nclass2\nclass3')
    classes_txt_path = tmp_path / "test-classes-missing.txt"
    classes_txt_path.write('')
    yield tmp_path
    shutil.rmtree(str(tmp_path))


@pytest.fixture
def dataset_empty_path(tmpdir):
    tmp_path = tmpdir.mkdir("dataset-empty")
    yield tmp_path
    shutil.rmtree(str(tmp_path))


@pytest.fixture(scope="class")
def task():
    task = mock.Mock(Task)
    task.name = 'test'
    task.id = 'test_id'
    return task


@pytest.fixture()
def darknet(tmpdir):
    tmp_path = tmpdir.mkdir("darknet")
    darknet = tmp_path / "darknet"
    darknet.write_binary(b'...')
    yield darknet
    shutil.rmtree(str(tmp_path))
