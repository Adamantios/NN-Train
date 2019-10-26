from os import makedirs
from os.path import join, isfile, dirname, exists
from typing import Union

import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Sequential, Model

from networks.cifar10.complicated_ensemble.complicated_ensemble import cifar10_complicated_ensemble
from networks.cifar10.different_architectures.model1 import cifar10_model1
from networks.cifar10.different_architectures.model2 import cifar10_model2
from networks.cifar10.different_architectures.model3 import cifar10_model3
from networks.cifar10.pyramid_ensemble.pyramid_ensemble import cifar10_pyramid_ensemble
from networks.cifar10.students.student_strong import cifar10_student_strong
from networks.cifar10.students.student_weak import cifar10_student_weak
from networks.cifar100.cifar100_complicated_ensemble import cifar100_complicated_ensemble
from networks.cifar100.cifar100_model1 import cifar100_model1
from networks.cifar100.cifar100_model2 import cifar100_model2
from networks.cifar100.cifar100_model3 import cifar100_model3


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = dirname(filepath)

    # Create directory if it does not exist
    if not exists(directory):
        makedirs(directory)


def plot_results(history: dict, save_folder: str = None) -> None:
    """
    Plots a given history's accuracy and loss results.

    :param history: the history to plot.
    :param save_folder: the folder to save the plots. If None, then the plots will not be saved.
    """
    acc_plot_filepath = join(save_folder, 'acc_plot.png')
    loss_plot_filepath = join(save_folder, 'loss_plot.png')
    acc_loss_plot_filepath = join(save_folder, 'acc_loss_plot.png')

    # Accuracy.
    fig = plt.figure(figsize=(12, 10))
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.title('Train/Test accuracy', fontsize='x-large')
    plt.xlabel('epoch', fontsize='large')
    plt.ylabel('accuracy', fontsize='large')
    plt.legend(['train', 'test'], loc='upper left', fontsize='large')
    plt.show()
    if save_folder is not None:
        fig.savefig(acc_plot_filepath)

    # Loss.
    fig = plt.figure(figsize=(12, 10))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Train/Test loss', fontsize='x-large')
    plt.xlabel('epoch', fontsize='large')
    plt.ylabel('loss', fontsize='large')
    plt.legend(['train', 'test'], loc='upper left', fontsize='large')
    plt.show()
    if save_folder is not None:
        fig.savefig(loss_plot_filepath)

    # Accuracy + Loss.
    fig = plt.figure(figsize=(12, 10))
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.plot(history['loss'], linestyle='dashed')
    plt.plot(history['val_loss'], linestyle='dashed')
    plt.title('Train/Test accuracy and loss', fontsize='x-large')
    plt.xlabel('epoch', fontsize='large')
    plt.ylabel('metric', fontsize='large')
    plt.legend(['train acc', 'test acc', 'train loss', 'test loss'],
               loc='upper left', fontsize='large')
    plt.show()
    if save_folder is not None:
        fig.savefig(acc_loss_plot_filepath)


def create_model(model_name: str, start_point: str, x_train: ndarray, n_classes: int) -> Union[Sequential, Model]:
    """
    Creates the model and loads weights as a start point if they exist.

    :return: Keras Sequential or functional API model.
    """
    if model_name == 'cifar10_model1':
        model_generator = cifar10_model1
    elif model_name == 'cifar10_model2':
        model_generator = cifar10_model2
    elif model_name == 'cifar10_model3':
        model_generator = cifar10_model3
    elif model_name == 'cifar10_complicated_ensemble':
        model_generator = cifar10_complicated_ensemble
    elif model_name == 'cifar10_pyramid_ensemble':
        model_generator = cifar10_pyramid_ensemble
    elif model_name == 'cifar10_student_strong':
        model_generator = cifar10_student_strong
    elif model_name == 'cifar10_student_weak':
        model_generator = cifar10_student_weak
    elif model_name == 'cifar100_model1':
        model_generator = cifar100_model1
    elif model_name == 'cifar100_model2':
        model_generator = cifar100_model2
    elif model_name == 'cifar100_model3':
        model_generator = cifar100_model3
    elif model_name == 'cifar100_complicated_ensemble':
        model_generator = cifar100_complicated_ensemble
    else:
        raise ValueError('Unrecognised model!')

    if start_point != '':
        if isfile(start_point):
            return model_generator(input_shape=x_train.shape[1:], weights_path=start_point, n_classes=n_classes)
        else:
            raise FileNotFoundError('Checkpoint file \'{}\' not found.'.format(start_point))
    else:
        return model_generator(input_shape=x_train.shape[1:], n_classes=n_classes)
