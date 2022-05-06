"""
Classifier training example.
"""
import os
import random

from clearml import Dataset, Task
from clearml_darknet import Darknet
from clearml_darknet.utils import split_dataset


dataset_path = Dataset.get(dataset_name='dataset-seatbelt', dataset_project='Tests/darknet').get_local_copy()
classes = os.path.join(dataset_path, 'classes.txt')

params = {'dataset_ratio': 0.5, 'dataset_shuffle': False}

task = Task.init(project_name='Tests/darknet', task_name='train-darknet-classificator', output_uri=None)
task.connect(params)
task.execute_remotely(queue_name='default', clone=False, exit_process=True)

print(params)

random.seed(0)
train, valid = split_dataset(
    dataset_path=dataset_path,
    ratio=params['dataset_ratio'],
    shuffle=params['dataset_shuffle']
)
print(f'Number of images for training={len(train)} for validation={len(valid)}')

darknet = Darknet(
    task=task,
    darknet_exec='/opt/darknet/darknet',
    config_path='data/darknet19.cfg',
    classes_path=classes,
    train=train,
    valid=valid,
)
darknet.train_classifier()
