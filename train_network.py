import logging
import pickle
from itertools import zip_longest
from os.path import join
from typing import Union, Tuple

from numpy import ndarray, empty
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.optimizers import rmsprop, adam, adamax, adadelta, adagrad, sgd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving import save_model
from tensorflow.python.keras.utils import to_categorical
from tensorflow_datasets import load, as_numpy

from helpers.parser import create_training_parser
from helpers.utils import create_path, plot_results, create_model
from networks.available_networks import subnetworks


def load_data() -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray], int]:
    """
    Loads the dataset.

    :return: the data and the number of classes.
    """
    data, info = load(dataset, with_info=True)
    data = as_numpy(data)
    data_shape = info.features.shape['image']
    labels_shape = info.features.shape['label']
    train_examples_num = info.splits['train'].num_examples
    test_examples_num = info.splits['test'].num_examples
    classes = info.features['label'].num_classes

    train, test = data['train'], data['test']
    train_data = empty((train_examples_num,) + data_shape)
    train_labels = empty((train_examples_num,) + labels_shape)
    test_data = empty((test_examples_num,) + data_shape)
    test_labels = empty((test_examples_num,) + labels_shape)

    for i, (sample_train, sample_test) in enumerate(zip_longest(train, test)):
        train_data[i] = sample_train['image']
        train_labels[i] = sample_train['label']

        if i < test_examples_num:
            test_data[i] = sample_test['image']
            test_labels[i] = sample_test['label']

    return (train_data, train_labels), (test_data, test_labels), classes


def preprocess_data(train: ndarray, test: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Preprocess the given data.

    :param train: the train data.
    :param test: the test data.
    :return: the preprocessed data.
    """
    train, test = train / 255, test / 255

    return train, test


def initialize_optimizer() -> Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]:
    """
    Initializes an optimizer based on the user's choices.

    :return: the optimizer.
    """
    if optimizer_name == 'adam':
        opt = adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=decay)
    elif optimizer_name == 'rmsprop':
        opt = rmsprop(lr=learning_rate, rho=rho, decay=decay)
    elif optimizer_name == 'sgd':
        opt = sgd(lr=learning_rate, momentum=momentum, decay=decay)
    elif optimizer_name == 'adagrad':
        opt = adagrad(lr=learning_rate, decay=decay)
    elif optimizer_name == 'adadelta':
        opt = adadelta(lr=learning_rate, rho=rho, decay=decay)
    elif optimizer_name == 'adamax':
        opt = adamax(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=decay)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')

    if clip_norm is not None:
        opt.clip_norm = clip_norm
    if clip_value is not None:
        opt.clip_value = clip_value
    return opt


def init_callbacks() -> []:
    """
    Initializes callbacks for the training procedure.

    :return: the callbacks list.
    """
    callbacks = []
    if save_checkpoint:
        # Create path for the file.
        create_path(checkpoint_filepath)

        # Create checkpoint.
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_categorical_accuracy', verbose=verbosity,
                                     save_best_only=True, mode='max')
        callbacks.append(checkpoint)

    if lr_decay > 0:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=lr_patience,
                                                    verbose=verbosity, factor=lr_decay, min_lr=lr_min)
        callbacks.append(learning_rate_reduction)

    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=early_stopping_patience,
                                       min_delta=1e-04, verbose=verbosity)
        callbacks.append(early_stopping)

    return callbacks


def evaluate_results() -> None:
    """ Evaluates the network's final results. """
    print('Evaluating results...')
    scores = model.evaluate(x_test, y_test)

    results = ''
    for i in range(len(scores)):
        results += "{}: {:.4f}\n".format(model.metrics_names[i], scores[i])

    print(results)

    # Save model evaluation results.
    if save_evaluation:
        logging.basicConfig(filename=results_filepath, filemode='w', format='%(message)s', level=logging.INFO)
        logging.info(results)
        print('Evaluation results have been saved as {}.\n'.format(results_filepath))


def save_results() -> None:
    """ Saves the training results (final weights, network and history). """
    # Save weights.
    if save_weights:
        # Create path for the file.
        create_path(weights_filepath)
        # Save weights.
        model.save_weights(weights_filepath)
        print('Network\'s weights have been saved as {}.\n'.format(weights_filepath))

    # Save model.
    if save_network:
        # Create path for the file.
        create_path(model_filepath)
        # Save model.
        save_model(model, model_filepath)
        print('Network has been saved as {}.\n'.format(model_filepath))

    # Save history.
    if save_history:
        # Create path for the file.
        create_path(hist_filepath)
        # Save history.
        with open(hist_filepath, 'wb') as file:
            pickle.dump(history.history, file)
        print('Network\'s history has been saved as {}.\n'.format(hist_filepath))


def manipulate_labels(initial_n_classes: int) -> int:
    """
    Manipulates the labels, depending on the model being used.
    For example some submodels need to be evaluated, but they do not predict all the labels.
    Thus, we need to give them only the labels that they predict and all the others placed into a separate bin.

    :param initial_n_classes: the initial number of classes. Returned in case there was no manipulation.
    :return: the new number of classes.
    """
    for name in subnetworks.keys():
        if model_name == name:
            labels_manipulation = subnetworks.get(name)
            return labels_manipulation([y_train, y_test])

    return initial_n_classes


if __name__ == '__main__':
    # Get arguments.
    args = create_training_parser().parse_args()
    dataset = args.dataset
    model_name = args.network
    start_point = args.start_point
    save_weights = not args.omit_weights
    save_network = not args.omit_model
    save_checkpoint = not args.omit_checkpoint
    save_history = not args.omit_history
    save_plots = not args.omit_plots
    save_evaluation = not args.omit_evaluation
    out_folder = args.out_folder
    optimizer_name = args.optimizer
    augment_data = not args.no_augmentation
    learning_rate = args.learning_rate
    lr_patience = args.learning_rate_patience
    lr_decay = args.learning_rate_decay
    lr_min = args.learning_rate_min
    early_stopping_patience = args.early_stopping_patience
    clip_norm = args.clip_norm
    clip_value = args.clip_value
    beta1 = args.beta1
    beta2 = args.beta2
    rho = args.rho
    momentum = args.momentum
    decay = args.decay
    batch_size = args.batch_size
    evaluation_batch_size = args.evaluation_batch_size
    epochs = args.epochs
    verbosity = args.verbosity

    if clip_norm is not None and clip_value is not None:
        raise ValueError('You cannot set both clip norm and clip value.')

    weights_filepath = join(out_folder, 'network_weights.h5')
    model_filepath = join(out_folder, 'network.h5')
    hist_filepath = join(out_folder, 'train_history.pickle')
    checkpoint_filepath = join(out_folder, 'checkpoint.h5')
    results_filepath = join(out_folder, 'results.log')

    # Load dataset.
    (x_train, y_train), (x_test, y_test), n_classes = load_data()
    # Manipulate labels, based on the model being used.
    n_classes = manipulate_labels(n_classes)
    # Preprocess data.
    x_train, x_test = preprocess_data(x_train.copy(), x_test.copy())
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    # Create model.
    model = create_model(model_name, start_point, x_train, n_classes)
    # Initialize optimizer.
    optimizer = initialize_optimizer()
    # Compile model.
    model.compile(optimizer, 'categorical_crossentropy', ['categorical_accuracy'])
    # Initialize callbacks list.
    callbacks_list = init_callbacks()

    if augment_data:
        # Generate batches of tensor image data with real-time data augmentation.
        datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
        datagen.fit(x_train)
        # Fit network.
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                      steps_per_epoch=x_train.shape[0] // batch_size, validation_data=(x_test, y_test),
                                      callbacks=callbacks_list)
    else:
        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks_list)

    # Plot results.
    save_folder = out_folder if save_plots else None

    if history.history:
        plot_results(history.history, save_folder)
        # Evaluate results.
        evaluate_results()

    # Save results.
    save_results()
