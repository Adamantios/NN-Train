from os import path, makedirs
from os.path import join

import matplotlib.pyplot as plt


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = path.dirname(filepath)

    # Create directory if it does not exist
    if not path.exists(directory):
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
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
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
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
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
