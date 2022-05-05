"""
Detector optimize example.
"""
from clearml import Task

from clearml.automation import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna


task = Task.init(
    project_name='Tests/darknet',
    task_name='train-darknet-detector hyperparameters optimization',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=True,
    output_uri=True,
)

optimizer = HyperParameterOptimizer(
    base_task_id='94b931f683ee4c61a6895112e347d56a',

    hyper_parameters=[
        UniformParameterRange('General/batch', min_value=16, max_value=128, step_size=16),
        UniformParameterRange('General/learning_rate', min_value=0.001, max_value=0.1, step_size=0.001),
        DiscreteParameterRange('General/dataset_ratio', values=[0.5, 0.6, 0.7])
    ],
    objective_metric_title='mAP@0.50',
    objective_metric_series='series',
    objective_metric_sign='max',

    optimizer_class=OptimizerOptuna,

    execution_queue='default',
    max_number_of_concurrent_tasks=4,
    optimization_time_limit=None,
    max_iteration_per_job=None,
    total_max_jobs=None,
)

task.execute_remotely(queue_name='default', clone=False, exit_process=True)

optimizer.start()
optimizer.wait()

optimizer.stop()
