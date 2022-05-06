"""
Module additional functions
"""
import os
import random
import typing

from clearml_darknet.errors import DatasetError


def extension_filter(extension: str, filename: str) -> bool:
    """Image and text file format detection filter.

    :param filename: File name.
    :param extension: File type.

    :return: bool.
    """
    format_file = {'image': ['.jpg', '.png', '.bmp', '.jpeg', '.gif'],
                   'video': ['.mp4'],
                   'text': ['.txt', '.xml']}

    return os.path.splitext(filename)[-1] in format_file[extension]


def split_dataset(dataset_path: str, ratio: float, shuffle: bool = False) -> typing.Any:
    """Splits the dataset into two samples based on the value of ration.

    :param dataset_path: Path to data folder.
    :param ratio: Portion (Ratio).
    :param shuffle: Shuffle images.

    :return: First sample list, second sample list.
    """
    if 0.0 > ratio or ratio >= 1.0:
        raise ValueError('The ration coefficient must be within the limits 0.0 < ratio < 1.0')

    list_dataset_images = []
    for dirs, paths, files in os.walk(dataset_path):
        for filename in files:
            if extension_filter('image', filename):
                list_dataset_images.append(os.path.join(dirs, filename))

    if not list_dataset_images:
        raise DatasetError(f'No images found in dataset={dataset_path}')

    if shuffle:
        random.shuffle(list_dataset_images)

    first_part = int(len(list_dataset_images) * ratio)
    first_sample_files, second_sample_files = list_dataset_images[:first_part], list_dataset_images[first_part:]

    return first_sample_files, second_sample_files
