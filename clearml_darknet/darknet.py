"""
Darknet learning package via ClearML
"""
import os
import re
import typing
import asyncio
import tempfile

from clearml import Task

from .errors import ConfigParseError


class Darknet:
    """
    The `Darknet` class provides an interface for training a detector or classifier
    various configurations.

    Methods
    -------
    set_subprocess_buffer_size()
        Changes the subprocess buffer size.
    train_detector()
        Starts the learning process of the detector.
    train_classifier(top: int)
        Starts the classifier training process.
    """

    def __init__(
            self,
            task: Task,
            darknet_exec: str,
            config_path: str,
            classes_path: str,
            train: list,
            valid: list,
            weight_path: str = None,
            is_save_weights: bool = True,
            save_weights_from_n_iterations: int = 1,
            save_weights_every_n_iterations: int = None,
            save_path_weights: str = './weights'
    ):
        """
        :param task: An instance of the clearml.Task class.
        :param darknet_exec: Path to compiled darknet.
        :param config_path: Path to network configuration.
        :param classes_path: Path to classes.txt file.
        :param train: Path to the list of images for training.
        :param valid: Path to the list of images for validation.
        :param weight_path: Path to the file with weights (additional training).
        :param is_save_weights: Save training weights.
        :param save_weights_from_n_iterations: Save weights from N iterations
        :param save_weights_every_n_iterations: Save weights every N iterations.
        :param save_path_weights: Path to the save weights.
        """
        self.__task = task

        self.__darknet_exec = darknet_exec
        self.__config_path = config_path
        self.__classes_path = classes_path
        self.__weight_path = weight_path
        self.__is_save_weights = is_save_weights
        self.__save_weights_from_n_iterations = save_weights_from_n_iterations
        self.__save_weights_every_n_iterations = save_weights_every_n_iterations
        self.__save_path_weights = save_path_weights

        self.__subprocess_buffer_size = 8224 * 8224

        # validate params
        if not os.path.isfile(darknet_exec):
            raise FileNotFoundError(f'Darknet file not found on path: {darknet_exec}')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f'Config file not found on path: {config_path}')
        if weight_path and not os.path.isfile(weight_path):
            raise FileNotFoundError(f'Weight file not found on path: {weight_path}')
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(f'Class file not found on path: {classes_path}')
        if os.path.getsize(classes_path) == 0:
            raise ValueError('Classes file is empty')
        if len(train) == 0:
            raise ValueError('List of training samples is empty.')
        if len(valid) == 0:
            raise ValueError('List of validating samples is empty.')
        if not is_save_weights and save_weights_from_n_iterations > 1 or save_weights_every_n_iterations is not None:
            raise ValueError('The parameter to save the weights must be True '
                             'if the rest of the parameters related to the weights are used.')
        if isinstance(save_weights_from_n_iterations, int) and 0 >= save_weights_from_n_iterations:
            raise ValueError('The parameter to save weights from N iterations must be greater than 0')
        if isinstance(save_weights_every_n_iterations, int) and 0 >= save_weights_every_n_iterations:
            raise ValueError('The parameter to save weights every N iterations must be greater than 0')

        self.__task_hyperparameters = self.__task.get_parameters_as_dict()
        self.__net_hyperparameters = {}

        with open(self.__classes_path, 'r') as f:
            self.classes_num = len(f.readlines())

        self.temp_dir_path = tempfile.mkdtemp()

        self.train_txt_path = os.path.join(self.temp_dir_path, 'train.txt')
        self.valid_txt_path = os.path.join(self.temp_dir_path, 'valid.txt')

        self._parse_hyperparameters_config()
        self._gen_txt_file(self.train_txt_path, train)
        self._gen_txt_file(self.valid_txt_path, valid)

    @staticmethod
    def _gen_txt_file(filename: str, sample: list) -> None:
        """Writes paths to images to a text file.

        :param filename: File name.
        :param sample: List of image paths.
        """
        with open(filename, 'w') as f:
            f.write('\n'.join(sample) + '\n')

    @staticmethod
    def _parse_checkpoint(line: str) -> typing.Optional[str]:
        """Parses a string to find the path to the weights.

        :param line: Text string.
        :return: path | None.
        """
        checkpoint_re = re.findall(r'Saving weights to (.*)', line)
        if checkpoint_re:
            return checkpoint_re[0]

    @staticmethod
    def _parse_checkpoint_iteration(line: str) -> typing.Optional[int]:
        """Parses the iteration number in the weights storage path string.

        :param line: Text string.
        :return: int | None
        """
        checkpoint_iteration_re = re.findall(r'[a-zA-Z\d_-]+_+(\d+)\.weight', line)
        if checkpoint_iteration_re:
            return int(checkpoint_iteration_re[0])

    @staticmethod
    def _parse_iteration(line: str) -> typing.Optional[int]:
        """Parses a string to find the iteration number.

        :param line: Text string.
        :return: iteration | None.
        """
        iteration_re = re.findall(r"^([\d ]+)[,|:] [\d.|\-nan]+[,|:] [\[\d.|\-nan]+|], ([\d.|\-nan]+) avg", line)
        if iteration_re:
            return int(iteration_re[0][0])

    @staticmethod
    def _parse_learning_rate(line: str) -> typing.Optional[float]:
        """Parses a string to find the value learning rate (lr).

        :param line: Text string.
        :return: lr | None
        """
        lr_re = re.findall(r"([\d.]+) rate", line)
        if lr_re:
            return float(lr_re[0])

    @staticmethod
    def _parse_avg_loss(line: str) -> typing.Optional[float]:
        """Parses a string to find the value of average loss.

        :param line: Text string.
        :return: avg_loss | None.
        """
        loss_re = re.findall(r"[\d.|\-nan]+, ([\d.|\-nan]+) avg", line)
        if loss_re:
            return float(loss_re[0])

    @staticmethod
    def _parse_map(line: str) -> typing.Optional[float]:
        """Parses a string to find the value mean average precision (map).

        :param line: Text string.
        :return: map | None
        """
        map_re = re.findall(r"mean average precision \(mAP@0\.50\) = [0.-9]+, or ([0.-9]+) %", line)
        if map_re:
            return float(map_re[0])

    @staticmethod
    def _parse_precision(line: str) -> typing.Optional[float]:
        """Parses a string to find the value of precision.

        :param line: Text string.
        :return: precision | None
        """
        precision_re = re.findall(r"precision = ([\d]*\.[\d]*|-nan)", line)
        if precision_re:
            return float(precision_re[0])

    @staticmethod
    def _parse_recall(line: str) -> typing.Optional[float]:
        """Parses a string to find the value of recall.

        :param line: Text string.
        :return: recall | None
        """
        recall_re = re.findall(r"recall = ([\d]*\.[\d]*|-nan)", line)
        if recall_re:
            return float(recall_re[0])

    @staticmethod
    def _parse_f1_score(line: str) -> typing.Optional[float]:
        """Parses a string to find the value of f1-score.

        :param line: Text string.
        :return: f1-score | None
        """
        f1_score_re = re.findall(r"F1-score = ([\d]*\.[\d]*|-nan)", line)
        if f1_score_re:
            return float(f1_score_re[0])

    @staticmethod
    def _parse_accuracy(line: str) -> typing.Optional[float]:
        """Parses a string to find the value of accuracy.

        :param line: Text string.
        :return: accuracy | None
        """
        accuracy_re = re.findall(r"accuracy top1 = ([\d.]+)", line)
        if accuracy_re:
            return float(accuracy_re[0]) * 100

    @staticmethod
    def _parse_hyperparameter(line: str) -> typing.Optional[typing.Any]:
        """Parses a string to find the parameter name and value.

        :param line: Text string.
        :return: (name, value) | None
        """
        result_re = re.findall(r"^([a-z_]+)[= ]+([a-z|\d.,]+)$", line)
        if result_re:
            return result_re[0][0], result_re[0][1]

    @staticmethod
    def _parse_section(line: str) -> typing.Optional[str]:
        """Parses a string to find the config section name.

        :param line: Text string.
        :return: name | None
        """
        result_re = re.findall(r"([\[a-z]+])", line)
        if result_re:
            return result_re[0]

    def _parse_hyperparameters_config(self) -> None:
        """Parses hyperparameters from the neural network configuration file."""
        NET_SECTION = '[net]'

        try:
            with open(self.__config_path, 'r') as f:
                self.__config_content = f.readlines()
        except Exception:
            raise ConfigParseError('Error reading network config file')

        for line in self.__config_content:
            line = line.strip()
            if not line:
                continue

            section = self._parse_section(line)
            if section and section != NET_SECTION:
                break
            if section and section == NET_SECTION or line[0] == '#':
                continue

            result = self._parse_hyperparameter(line)
            if not result:
                raise ConfigParseError(f"Error reading parameter '{line}' of network config")

            param_name, param_value = result
            self.__net_hyperparameters[param_name] = param_value

    def _gen_cfg_file(self) -> str:
        """Generates a new network config.

        :return: Path to the generated network config file.
        """
        cfg_name = os.path.basename(self.__config_path)
        cfg_file_path = os.path.join(self.temp_dir_path, f'{cfg_name}')

        content = []
        for raw_line in self.__config_content:
            result = raw_line.strip().split('=')
            if result and len(result[0]) != 0:
                param_name = result[0].strip()
                if self.__task_hyperparameters['General'].get(param_name):
                    raw_line = f"{param_name}={self.__task_hyperparameters['General'][param_name]}\n"
            content.append(raw_line)

        with open(cfg_file_path, 'w') as f:
            f.writelines(content)

        return cfg_file_path

    def _gen_obj_file(self, type_: str, top: int = None) -> str:
        """Generates the obj.data file.

        :param type_: Network type `detector`/`classifier`.
        :param top: top parameter for `classifier`.

        :return: Path to the obj.data file.
        """
        obj_data_file_path = os.path.join(self.temp_dir_path, 'obj.data')
        os.makedirs(self.__save_path_weights, exist_ok=True)

        content = [
            f'classes={self.classes_num}\n'
            f'train={self.train_txt_path}\n',
            f'valid={self.valid_txt_path}\n',
            f'{"names" if type_ == "detector" else "labels"}={self.__classes_path}\n',
            f'backup={self.__save_path_weights}\n',
            'eval=coco' if type_ == "detector" else f'top={top}'
        ]
        with open(obj_data_file_path, 'w') as f:
            f.writelines(content)

        return obj_data_file_path

    def _process_output(self, line: str) -> typing.Any:
        """Parses strings to find result values.

        :param line: Text string.
        :return: iteration, learning_rate, avg_loss, map, precision, recall, f1_score, accuracy
        """
        iteration = self._parse_iteration(line)
        learning_rate = self._parse_learning_rate(line)
        avg_loss = self._parse_avg_loss(line)
        mean_avg_precision = self._parse_map(line)
        precision = self._parse_precision(line)
        recall = self._parse_recall(line)
        f1_score = self._parse_f1_score(line)
        accuracy = self._parse_accuracy(line)

        return iteration, learning_rate, avg_loss, mean_avg_precision, precision, recall, f1_score, accuracy

    def _is_save_iteration_weights(self, checkpoint_iteration: int, iteration_difference: int) -> bool:
        """Checks whether iterative weights can be stored or not.

        :param checkpoint_iteration: Checkpoint iteration number.
        :param iteration_difference: Difference between the checkpoint iteration number and
        the last iteration of saving weights.
        :return: bool.
        """
        if ((checkpoint_iteration >= self.__save_weights_from_n_iterations) and
                not self.__save_weights_every_n_iterations):
            return True
        elif (
                (checkpoint_iteration >= self.__save_weights_from_n_iterations) and
                (iteration_difference >= self.__save_weights_every_n_iterations)
        ):
            return True
        return False

    def _train(self, obj_data: str, type_: str, **kwargs) -> None:
        """Running network training.

        :param obj_data: Path to the obj.data file.
        :param type_: Network type `detector`/`classifier`.
        """
        if self.__task_hyperparameters:
            self.__config_path = self._gen_cfg_file()
        self.__task.connect(self.__net_hyperparameters)

        command: list = [self.__darknet_exec, type_, "train", obj_data, self.__config_path, '-dont_show']
        command.append(self.__weight_path) if self.__weight_path else ...
        command.append('-map') if kwargs.get('calc_map') else ...
        command.append('-topk') if kwargs.get('calc_acc') else ...

        asyncio.run(self._run_process(command))

    def train_detector(self) -> None:
        """Starting the Detector Learning."""
        type_ = 'detector'
        obj_data = self._gen_obj_file(type_=type_)
        self._train(obj_data, type_=type_, calc_map=True)

    def train_classifier(self, top: int = 1) -> None:
        """Start classifier training.

        :param top: verification accuracy (top1, top3, top5, topN).
        """
        type_ = 'classifier'
        obj_data = self._gen_obj_file(type_=type_, top=top)
        self._train(obj_data, type_=type_, calc_acc=True)

    def set_subprocess_buffer_size(self, buffer_size: int) -> None:
        """Changes the subprocess buffer size.

        :param buffer_size: int. DEFAULT: 8224 * 8224.
        """
        if 0 >= buffer_size:
            raise ValueError('The parameter to buffer_size must be greater than 0')
        self.__subprocess_buffer_size = buffer_size

    async def _run_process(self, command: list) -> None:
        """Starts the learning process of the neural network.

        :param command: Executable command.
        """
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.__subprocess_buffer_size
        )
        await asyncio.gather(
            self._stream_process(process.stdout, is_stdout=True),
            self._stream_process(process.stderr)
        )
        await process.wait()

    def _send_stream_data_to_task(self, last_iteration: int, **kwargs) -> None:
        """Sends stream data to a task.

        :param last_iteration: Number of the last training iteration.
        """
        if kwargs['lr']:
            self.__task.logger.report_scalar(
                title='lr', series='series', value=kwargs['lr'], iteration=last_iteration
            )
        if kwargs['avg_loss']:
            self.__task.logger.report_scalar(
                title='avg_loss', series='series', value=kwargs['avg_loss'], iteration=last_iteration
            )
        if kwargs['mean_avg_precision']:
            self.__task.logger.report_scalar(
                title='mAP@0.50', series='series', value=kwargs['mean_avg_precision'], iteration=last_iteration
            )
        if kwargs['precision']:
            self.__task.logger.report_scalar(
                title='precision', series='series', value=kwargs['precision'], iteration=last_iteration
            )
        if kwargs['recall']:
            self.__task.logger.report_scalar(
                title='recall', series='series', value=kwargs['recall'], iteration=last_iteration
            )
        if kwargs['f1_score']:
            self.__task.logger.report_scalar(
                title='f1_score', series='series', value=kwargs['f1_score'], iteration=last_iteration
            )
        if kwargs['accuracy']:
            self.__task.logger.report_scalar(
                title='accuracy', series='series', value=kwargs['accuracy'], iteration=last_iteration
            )

    async def _stream_process(self, stream: asyncio.StreamReader, is_stdout: bool = False) -> None:
        """Reads streaming data from the training process and sends it to the task.

        Also, here is the process of upload weights.

        :param stream: asyncio.StreamReader.
        :param is_stdout: Is a stdout stream.
        """
        last_iteration = 0
        last_save_weights_iteration = 0
        while True:
            line = await stream.readline()
            line_decode = line.decode().strip()

            if line:
                print(line_decode)

                if is_stdout:
                    iteration, lr, avg_loss, mean_avg_precision, precision, recall, f1_score, accuracy = \
                        self._process_output(line_decode)

                    if iteration is not None:
                        last_iteration = iteration

                    self._send_stream_data_to_task(
                        last_iteration,
                        iteration=iteration,
                        lr=lr,
                        avg_loss=avg_loss,
                        mean_avg_precision=mean_avg_precision,
                        precision=precision,
                        recall=recall,
                        f1_score=f1_score,
                        accuracy=accuracy
                    )
                else:
                    if self.__is_save_weights:
                        checkpoint = self._parse_checkpoint(line_decode)
                        if checkpoint:
                            checkpoint_iteration = self._parse_checkpoint_iteration(checkpoint)
                            is_save_iteration_weights = False
                            if checkpoint_iteration:
                                iteration_difference = checkpoint_iteration - last_save_weights_iteration
                                is_save_iteration_weights = self._is_save_iteration_weights(
                                    checkpoint_iteration, iteration_difference
                                )
                            if not checkpoint_iteration or is_save_iteration_weights:
                                self.__task.upload_artifact(
                                    name=os.path.basename(checkpoint), artifact_object=checkpoint
                                )
                            if is_save_iteration_weights:
                                last_save_weights_iteration = checkpoint_iteration
            else:
                break
