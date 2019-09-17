from argparse import ArgumentParser

# ----------------------------------- DEFAULT ARGUMENTS ------------------------------------------

DATASET = 'cifar10'
DATASET_CHOICES = 'cifar10', 'cifar100'
NETWORK = 'cifar10_model1'
NETWORK_CHOICES = 'cifar10_model1', 'cifar100_model1'
START_POINT = ''
SAVE_WEIGHTS = True
SAVE_MODEL = True
SAVE_CHECKPOINT = True
SAVE_HISTORY = True
SAVE_PLOTS = True
SAVE_EVALUATION_RESULTS = True
OUT_FOLDER_NAME = 'out'
OPTIMIZER = 'rmsprop'
OPTIMIZER_CHOICES = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
AUGMENT = True
LEARNING_RATE = 1E-3
LR_PATIENCE = 8
LR_DECAY = 0.1
LR_MIN = 0.00000001
EARLY_STOPPING_PATIENCE = 15
BETA1 = .9
BETA2 = .999
RHO = .9
MOMENTUM = .0
DECAY = 1E-6
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
EPOCHS = 125
VERBOSITY = 1


# ------------------------------------------------------------------------------------------------


def create_training_parser() -> ArgumentParser:
    """
    Creates an argument parser for the network training script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Training a CNN network.',
                            epilog='Note: '
                                   'The hyperparameters will be ignored if the chosen optimizer does not use them.\n'
                                   'Little hack to just save a model from existing checkpoint: \n'
                                   '$python train_network.py -sp checkpoint.h5 -e 0')
    parser.add_argument('-db', '--dataset', type=str.lower, default=DATASET, required=False, choices=DATASET_CHOICES,
                        help='The dataset to be used (default %(default)s).')
    parser.add_argument('-n', '--network', type=str.lower, default=NETWORK, required=False, choices=NETWORK_CHOICES,
                        help='The network model to be used (default %(default)s).')
    parser.add_argument('-sp', '--start_point', type=str, required=False, default=START_POINT,
                        help='Filepath containing existing weights to initialize the model.')
    parser.add_argument('-ow', '--omit_weights', default=not SAVE_WEIGHTS, required=False, action='store_true',
                        help='Whether the weights should not be saved (default %(default)s).')
    parser.add_argument('-om', '--omit_model', default=not SAVE_MODEL, required=False, action='store_true',
                        help='Whether the model should not be saved (default %(default)s).')
    parser.add_argument('-oc', '--omit_checkpoint', default=not SAVE_CHECKPOINT, required=False, action='store_true',
                        help='Whether the best weights checkpoint should not be saved (default %(default)s).')
    parser.add_argument('-oh', '--omit_history', default=not SAVE_HISTORY, required=False, action='store_true',
                        help='Whether the training history should not be saved (default %(default)s).')
    parser.add_argument('-op', '--omit_plots', default=not SAVE_PLOTS, required=False, action='store_true',
                        help='Whether the training plots should not be saved (default %(default)s).')
    parser.add_argument('-oe', '--omit_evaluation', default=not SAVE_EVALUATION_RESULTS, required=False,
                        action='store_true',
                        help='Whether the evaluation results should not be saved (default %(default)s).')
    parser.add_argument('-out', '--out_folder', default=OUT_FOLDER_NAME, required=False, type=str,
                        help='Path to the folder where the outputs will be stored (default %(default)s).')
    parser.add_argument('-o', '--optimizer', type=str.lower, default=OPTIMIZER, required=False,
                        choices=OPTIMIZER_CHOICES,
                        help='The optimizer to be used. (default %(default)s).')
    parser.add_argument('-na', '--no_augmentation', default=not AUGMENT, required=False, action='store_true',
                        help='Whether the data should not be augmented. '
                             'Augmentation is suggested only for image data. (default %(default)s).')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, required=False,
                        help='The learning rate for the optimizer (default %(default)s).')
    parser.add_argument('-lrp', '--learning_rate_patience', type=int, default=LR_PATIENCE, required=False,
                        help='The number of epochs to wait before decaying the learning rate (default %(default)s).')
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=LR_DECAY, required=False,
                        help='The learning rate decay factor. '
                             'If 0 is given, then the learning rate will remain the same during the training process. '
                             '(default %(default)s).')
    parser.add_argument('-lrm', '--learning_rate_min', type=float, default=LR_MIN, required=False,
                        help='The minimum learning rate which can be reached (default %(default)s).')
    parser.add_argument('-esp', '--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, required=False,
                        help='The number of epochs to wait before early stopping'
                             'If 0 is given, early stopping will not be applied. (default %(default)s).')
    parser.add_argument('-cn', '--clip_norm', type=float, required=False,
                        help='The clip norm for the optimizer (default %(default)s).')
    parser.add_argument('-cv', '--clip_value', type=float, required=False,
                        help='The clip value for the optimizer (default %(default)s).')
    parser.add_argument('-b1', '--beta1', type=float, default=BETA1, required=False,
                        help='The beta 1 for the optimizer (default %(default)s).')
    parser.add_argument('-b2', '--beta2', type=float, default=BETA2, required=False,
                        help='The beta 2 for the optimizer (default %(default)s).')
    parser.add_argument('-rho', type=float, default=RHO, required=False,
                        help='The rho for the optimizer (default %(default)s).')
    parser.add_argument('-m', '--momentum', type=float, default=MOMENTUM, required=False,
                        help='The momentum for the optimizer (default %(default)s).')
    parser.add_argument('-d', '--decay', type=float, default=DECAY, required=False,
                        help='The decay for the optimizer (default %(default)s).')
    parser.add_argument('-bs', '--batch_size', type=int, default=TRAIN_BATCH_SIZE, required=False,
                        help='The batch size for the optimization (default %(default)s).')
    parser.add_argument('-ebs', '--evaluation_batch_size', type=int, default=EVAL_BATCH_SIZE, required=False,
                        help='The batch size for the evaluation (default %(default)s).')
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS, required=False,
                        help='The number of epochs to train the network (default %(default)s).')
    parser.add_argument('-v', '--verbosity', type=int, default=VERBOSITY, required=False,
                        help='The verbosity for the optimization procedure (default %(default)s).')

    return parser
