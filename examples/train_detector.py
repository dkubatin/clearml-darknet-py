import os
import random

from clearml import Dataset, Task
from clearml_darknet import Darknet, split_dataset


dataset_path = Dataset.get(dataset_name='dataset-cars', dataset_project='Tests/darknet').get_local_copy()
classes = os.path.join(dataset_path, 'classes.txt')

random.seed(0)
train, valid = split_dataset(dataset_path=dataset_path, ratio=0.7, shuffle=True)

task = Task.init(project_name='Tests/darknet', task_name='train-darknet-detector', output_uri=None)
task.execute_remotely(queue_name='default', clone=False, exit_process=True)

darknet = Darknet(
    task=task,
    darknet_exec='/opt/darknet/darknet',
    config_path='data/yolov3-tiny.cfg',
    classes_path=classes,
    train=train,
    valid=valid,
)
darknet.train_detector()
