from os import makedirs
from os.path import join, isfile, dirname, exists
from typing import Union

import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Sequential, Model

from networks.available_networks import networks


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
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.plot(history['categorical_accuracy'])
    ax.plot(history['val_categorical_accuracy'])
    ax.set_title('Train/Test accuracy', fontsize='x-large')
    ax.set_xlabel('epoch', fontsize='large')
    ax.set_ylabel('accuracy', fontsize='large')
    ax.legend(['train', 'test'], loc='upper left', fontsize='large')
    fig.show()
    if save_folder is not None:
        fig.savefig(acc_plot_filepath)

    # Loss.
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.plot(history['loss'])
    ax.plot(history['val_loss'])
    ax.set_title('Train/Test loss', fontsize='x-large')
    ax.set_xlabel('epoch', fontsize='large')
    ax.set_ylabel('loss', fontsize='large')
    ax.legend(['train', 'test'], loc='upper left', fontsize='large')
    fig.show()
    if save_folder is not None:
        fig.savefig(loss_plot_filepath)

    # Accuracy + Loss.
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.plot(history['categorical_accuracy'])
    ax.plot(history['val_categorical_accuracy'])
    ax.plot(history['loss'], linestyle='dashed')
    ax.plot(history['val_loss'], linestyle='dashed')
    ax.set_title('Train/Test accuracy and loss', fontsize='x-large')
    ax.set_xlabel('epoch', fontsize='large')
    ax.set_ylabel('metric', fontsize='large')
    ax.legend(['train acc', 'test acc', 'train loss', 'test loss'],
              loc='upper left', fontsize='large')
    fig.show()
    if save_folder is not None:
        fig.savefig(acc_loss_plot_filepath)


def create_model(model_name: str, start_point: str, x_train: ndarray, n_classes: int) -> Union[Sequential, Model]:
    """
    Creates the model and loads weights as a start point if they exist.

    :return: Keras Sequential or functional API model.
    """
    for name in networks.keys():
        if model_name == name:
            model_generator = networks.get(name)

            if start_point != '':
                if isfile(start_point):
                    return model_generator(input_shape=x_train.shape[1:], weights_path=start_point, n_classes=n_classes)
                else:
                    raise FileNotFoundError('Checkpoint file \'{}\' not found.'.format(start_point))
            else:
                return model_generator(input_shape=x_train.shape[1:], n_classes=n_classes)

    raise ValueError('Unrecognised model!')
