from typing import Dict, Callable, Union, List

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model, Sequential

from networks.cifar10.complicated_ensemble.ensemble import cifar10_complicated_ensemble
from networks.cifar10.complicated_ensemble.submodel1 import cifar10_complicated_ensemble_submodel1, \
    cifar10_complicated_ensemble_submodel1_labels_manipulation
from networks.cifar10.complicated_ensemble.submodel2 import cifar10_complicated_ensemble_submodel2, \
    cifar10_complicated_ensemble_submodel2_labels_manipulation
from networks.cifar10.complicated_ensemble.submodel3 import cifar10_complicated_ensemble_submodel3, \
    cifar10_complicated_ensemble_submodel3_labels_manipulation
from networks.cifar10.complicated_ensemble.submodel4 import cifar10_complicated_ensemble_submodel4, \
    cifar10_complicated_ensemble_submodel4_labels_manipulation
from networks.cifar10.complicated_ensemble.submodel5 import cifar10_complicated_ensemble_submodel5, \
    cifar10_complicated_ensemble_submodel5_labels_manipulation
from networks.cifar10.different_architectures.model1 import cifar10_model1
from networks.cifar10.different_architectures.model2 import cifar10_model2
from networks.cifar10.different_architectures.model3 import cifar10_model3
from networks.cifar10.pyramid_ensemble.ensemble import cifar10_pyramid_ensemble
from networks.cifar10.pyramid_ensemble.submodel_strong import cifar10_pyramid_ensemble_submodel_strong
from networks.cifar10.pyramid_ensemble.submodel_weak1 import \
    cifar10_pyramid_ensemble_submodel_weak1_labels_manipulation, cifar10_pyramid_ensemble_submodel_weak1
from networks.cifar10.pyramid_ensemble.submodel_weak2 import \
    cifar10_pyramid_ensemble_submodel_weak2_labels_manipulation, cifar10_pyramid_ensemble_submodel_weak2
from networks.cifar10.students.strong import cifar10_student_strong
from networks.cifar10.students.weak import cifar10_student_weak
from networks.cifar100.cifar100_complicated_ensemble import cifar100_complicated_ensemble
from networks.cifar100.cifar100_model1 import cifar100_model1
from networks.cifar100.cifar100_model2 import cifar100_model2
from networks.cifar100.cifar100_model3 import cifar100_model3

NetworksType = Dict[str, Callable[[any, any, any, Union[None, str]], Union[Model, Sequential]]]
LabelsManipulatorType = Callable[[List[ndarray]], int]

_cifar10_networks: NetworksType = {
    'cifar10_model1': cifar10_model1,
    'cifar10_model2': cifar10_model2,
    'cifar10_model3': cifar10_model3,
    'cifar10_complicated_ensemble': cifar10_complicated_ensemble,
    'cifar10_complicated_ensemble_submodel1': cifar10_complicated_ensemble_submodel1,
    'cifar10_complicated_ensemble_submodel2': cifar10_complicated_ensemble_submodel2,
    'cifar10_complicated_ensemble_submodel3': cifar10_complicated_ensemble_submodel3,
    'cifar10_complicated_ensemble_submodel4': cifar10_complicated_ensemble_submodel4,
    'cifar10_complicated_ensemble_submodel5': cifar10_complicated_ensemble_submodel5,
    'cifar10_pyramid_ensemble': cifar10_pyramid_ensemble,
    'cifar10_pyramid_ensemble_submodel_strong': cifar10_pyramid_ensemble_submodel_strong,
    'cifar10_pyramid_ensemble_submodel_weak1': cifar10_pyramid_ensemble_submodel_weak1,
    'cifar10_pyramid_ensemble_submodel_weak2': cifar10_pyramid_ensemble_submodel_weak2,
    'cifar10_student_strong': cifar10_student_strong,
    'cifar10_student_weak': cifar10_student_weak
}

_cifar100_networks: NetworksType = {
    'cifar100_model1': cifar100_model1,
    'cifar100_model2': cifar100_model2,
    'cifar100_model3': cifar100_model3,
    'cifar100_complicated_ensemble': cifar100_complicated_ensemble
}

networks: NetworksType = dict(_cifar10_networks, **_cifar100_networks)


def _labels_manipulation(manipulator: Callable[[ndarray], int]) -> LabelsManipulatorType:
    def labels_manipulator(labels: List[ndarray]) -> int:
        n_classes = 0
        for labels_array in labels:
            n_classes = manipulator(labels_array)
        return n_classes

    return labels_manipulator


subnetworks: Dict[str, LabelsManipulatorType] = {
    'cifar10_complicated_ensemble_submodel1':
        _labels_manipulation(cifar10_complicated_ensemble_submodel1_labels_manipulation),
    'cifar10_complicated_ensemble_submodel2':
        _labels_manipulation(cifar10_complicated_ensemble_submodel2_labels_manipulation),
    'cifar10_complicated_ensemble_submodel3':
        _labels_manipulation(cifar10_complicated_ensemble_submodel3_labels_manipulation),
    'cifar10_complicated_ensemble_submodel4':
        _labels_manipulation(cifar10_complicated_ensemble_submodel4_labels_manipulation),
    'cifar10_complicated_ensemble_submodel5':
        _labels_manipulation(cifar10_complicated_ensemble_submodel5_labels_manipulation),
    'cifar10_pyramid_ensemble_submodel_weak1':
        _labels_manipulation(cifar10_pyramid_ensemble_submodel_weak1_labels_manipulation),
    'cifar10_pyramid_ensemble_submodel_weak2':
        _labels_manipulation(cifar10_pyramid_ensemble_submodel_weak2_labels_manipulation),
}
