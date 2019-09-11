from argparse import ArgumentParser
from os import path, makedirs

# ----------------------------------- DEFAULT ARGUMENTS ------------------------------------------

START_POINT = ''
SAVE_WEIGHTS = True
SAVE_MODEL = True
SAVE_CHECKPOINT = True
SAVE_HIST = True
WEIGHTS_FILENAME = 'out/network_weights.h5'
MODEL_FILENAME = 'out/network.h5'
CHECKPOINT_FILENAME = 'out/checkpoint.h5'
HIST_FILENAME = 'out/train_history.pickle'
OPTIMIZER = 'rmsprop'
OPTIMIZER_CHOICES = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
LEARNING_RATE = 1E-3
CLIP_NORM = 1
CLIP_VALUE = .5
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
                                   'The hyperparameters will be ignored if the chosen optimizer does not use them.')
    parser.add_argument('-sp', '--start_point', type=str, required=False, default=START_POINT,
                        help='Filepath containing existing weights to initialize the model.')
    parser.add_argument('-ow', '--omit_weights', default=not SAVE_WEIGHTS, required=False, action='store_true',
                        help='Whether the weights should not be saved (default %(default)s).')
    parser.add_argument('-om', '--omit_model', default=not SAVE_MODEL, required=False, action='store_true',
                        help='Whether the model should not be saved (default %(default)s).')
    parser.add_argument('-oc', '--omit_checkpoint', default=not SAVE_CHECKPOINT, required=False, action='store_true',
                        help='Whether the best weights checkpoint should not be saved (default %(default)s).')
    parser.add_argument('-oh', '--omit_history', default=not SAVE_HIST, required=False, action='store_true',
                        help='Whether the training history should not be saved (default %(default)s).')
    parser.add_argument('-wf', '--weights_filepath', default=WEIGHTS_FILENAME, required=False, type=str,
                        help='Path to store the trained network\'s weights (default %(default)s). '
                             'Ignored if --omit_weights has been chosen')
    parser.add_argument('-mf', '--model_filepath', default=MODEL_FILENAME, required=False, type=str,
                        help='Path to store the trained network (default %(default)s). '
                             'Ignored if --omit_model has been chosen')
    parser.add_argument('-hf', '--history_filepath', default=HIST_FILENAME, required=False, type=str,
                        help='Path to store the trained network\'s history (default %(default)s). '
                             'Ignored if --omit_history has been chosen')
    parser.add_argument('-cf', '--checkpoint_filepath', default=CHECKPOINT_FILENAME, required=False, type=str,
                        help='Path to store the trained network\'s best checkpoint(default %(default)s). '
                             'Ignored if --omit_checkpoint has been chosen')
    parser.add_argument('-o', '--optimizer', type=str.lower, default=OPTIMIZER, required=False,
                        choices=OPTIMIZER_CHOICES,
                        help='The optimizer to be used. (default %(default)s).')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, required=False,
                        help='The learning rate for the optimizer (default %(default)s).')
    parser.add_argument('-cn', '--clip_norm', type=float, default=CLIP_NORM, required=False,
                        help='The clip norm for the optimizer (default %(default)s).')
    parser.add_argument('-cv', '--clip_value', type=float, default=CLIP_VALUE, required=False,
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
