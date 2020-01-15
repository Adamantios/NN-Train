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
from networks.cifar100.complicated_ensemble.ensemble import cifar100_complicated_ensemble
from networks.cifar100.complicated_ensemble.submodel1 import cifar100_complicated_ensemble_submodel1, \
    cifar100_complicated_ensemble_submodel1_labels_manipulation
from networks.cifar100.complicated_ensemble.submodel2 import cifar100_complicated_ensemble_submodel2, \
    cifar100_complicated_ensemble_submodel2_labels_manipulation
from networks.cifar100.complicated_ensemble.submodel3 import cifar100_complicated_ensemble_submodel3, \
    cifar100_complicated_ensemble_submodel3_labels_manipulation
from networks.cifar100.complicated_ensemble.submodel4 import cifar100_complicated_ensemble_submodel4, \
    cifar100_complicated_ensemble_submodel4_labels_manipulation
from networks.cifar100.complicated_ensemble.submodel5 import cifar100_complicated_ensemble_submodel5, \
    cifar100_complicated_ensemble_submodel5_labels_manipulation
from networks.cifar100.different_architectures.model1 import cifar100_model1
from networks.cifar100.different_architectures.model2 import cifar100_model2
from networks.cifar100.different_architectures.model3 import cifar100_model3
from networks.cifar100.pyramid_ensemble.ensemble import cifar100_pyramid_ensemble
from networks.cifar100.pyramid_ensemble.submodel_strong import cifar100_pyramid_ensemble_submodel_strong
from networks.cifar100.pyramid_ensemble.submodel_weak1 import cifar100_pyramid_ensemble_submodel_weak1, \
    cifar100_pyramid_ensemble_submodel_weak1_labels_manipulation
from networks.cifar100.pyramid_ensemble.submodel_weak2 import cifar100_pyramid_ensemble_submodel_weak2, \
    cifar100_pyramid_ensemble_submodel_weak2_labels_manipulation
from networks.cifar100.students.strong import cifar100_student_strong
from networks.cifar100.students.weak import cifar100_student_weak
from networks.svhn.complicated_ensemble.ensemble import svhn_complicated_ensemble
from networks.svhn.complicated_ensemble.submodel1 import svhn_complicated_ensemble_submodel1, \
    svhn_complicated_ensemble_submodel1_labels_manipulation
from networks.svhn.complicated_ensemble.submodel2 import svhn_complicated_ensemble_submodel2, \
    svhn_complicated_ensemble_submodel2_labels_manipulation
from networks.svhn.complicated_ensemble.submodel3 import svhn_complicated_ensemble_submodel3, \
    svhn_complicated_ensemble_submodel3_labels_manipulation
from networks.svhn.complicated_ensemble.submodel4 import svhn_complicated_ensemble_submodel4, \
    svhn_complicated_ensemble_submodel4_labels_manipulation
from networks.svhn.complicated_ensemble.submodel5 import svhn_complicated_ensemble_submodel5, \
    svhn_complicated_ensemble_submodel5_labels_manipulation
from networks.svhn.different_architectures.model1 import svhn_model1
from networks.svhn.different_architectures.model2 import svhn_model2
from networks.svhn.different_architectures.model3 import svhn_model3
from networks.svhn.pyramid_ensemble.ensemble import svhn_pyramid_ensemble
from networks.svhn.pyramid_ensemble.submodel_strong import svhn_pyramid_ensemble_submodel_strong
from networks.svhn.pyramid_ensemble.submodel_weak1 import svhn_pyramid_ensemble_submodel_weak1, \
    svhn_pyramid_ensemble_submodel_weak1_labels_manipulation
from networks.svhn.pyramid_ensemble.submodel_weak2 import svhn_pyramid_ensemble_submodel_weak2, \
    svhn_pyramid_ensemble_submodel_weak2_labels_manipulation
from networks.svhn.students.strong import svhn_student_strong
from networks.svhn.students.weak import svhn_student_weak

NetworksType = Dict[str, Callable[[any, any, any, Union[None, str]], Union[Model, Sequential]]]
LabelsManipulatorType = Callable[[List[ndarray]], int]
SubnetworksType = Dict[str, LabelsManipulatorType]


def _labels_manipulation(manipulator: Callable[[ndarray], int]) -> LabelsManipulatorType:
    def labels_manipulator(labels: List[ndarray]) -> int:
        n_classes = 0
        for labels_array in labels:
            n_classes = manipulator(labels_array)
        return n_classes

    return labels_manipulator


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
    'cifar100_complicated_ensemble': cifar100_complicated_ensemble,
    'cifar100_complicated_ensemble_submodel1': cifar100_complicated_ensemble_submodel1,
    'cifar100_complicated_ensemble_submodel2': cifar100_complicated_ensemble_submodel2,
    'cifar100_complicated_ensemble_submodel3': cifar100_complicated_ensemble_submodel3,
    'cifar100_complicated_ensemble_submodel4': cifar100_complicated_ensemble_submodel4,
    'cifar100_complicated_ensemble_submodel5': cifar100_complicated_ensemble_submodel5,
    'cifar100_pyramid_ensemble': cifar100_pyramid_ensemble,
    'cifar100_pyramid_ensemble_submodel_strong': cifar100_pyramid_ensemble_submodel_strong,
    'cifar100_pyramid_ensemble_submodel_weak1': cifar100_pyramid_ensemble_submodel_weak1,
    'cifar100_pyramid_ensemble_submodel_weak2': cifar100_pyramid_ensemble_submodel_weak2,
    'cifar100_student_strong': cifar100_student_strong,
    'cifar100_student_weak': cifar100_student_weak
}

_svhn_networks: NetworksType = {
    'svhn_model1': svhn_model1,
    'svhn_model2': svhn_model2,
    'svhn_model3': svhn_model3,
    'svhn_complicated_ensemble': svhn_complicated_ensemble,
    'svhn_complicated_ensemble_submodel1': svhn_complicated_ensemble_submodel1,
    'svhn_complicated_ensemble_submodel2': svhn_complicated_ensemble_submodel2,
    'svhn_complicated_ensemble_submodel3': svhn_complicated_ensemble_submodel3,
    'svhn_complicated_ensemble_submodel4': svhn_complicated_ensemble_submodel4,
    'svhn_complicated_ensemble_submodel5': svhn_complicated_ensemble_submodel5,
    'svhn_pyramid_ensemble': svhn_pyramid_ensemble,
    'svhn_pyramid_ensemble_submodel_strong': svhn_pyramid_ensemble_submodel_strong,
    'svhn_pyramid_ensemble_submodel_weak1': svhn_pyramid_ensemble_submodel_weak1,
    'svhn_pyramid_ensemble_submodel_weak2': svhn_pyramid_ensemble_submodel_weak2,
    'svhn_student_strong': svhn_student_strong,
    'svhn_student_weak': svhn_student_weak
}

_cifar10_subnetworks: SubnetworksType = {
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
        _labels_manipulation(cifar10_pyramid_ensemble_submodel_weak2_labels_manipulation)
}

_cifar100_subnetworks: SubnetworksType = {
    'cifar100_complicated_ensemble_submodel1':
        _labels_manipulation(cifar100_complicated_ensemble_submodel1_labels_manipulation),
    'cifar100_complicated_ensemble_submodel2':
        _labels_manipulation(cifar100_complicated_ensemble_submodel2_labels_manipulation),
    'cifar100_complicated_ensemble_submodel3':
        _labels_manipulation(cifar100_complicated_ensemble_submodel3_labels_manipulation),
    'cifar100_complicated_ensemble_submodel4':
        _labels_manipulation(cifar100_complicated_ensemble_submodel4_labels_manipulation),
    'cifar100_complicated_ensemble_submodel5':
        _labels_manipulation(cifar100_complicated_ensemble_submodel5_labels_manipulation),
    'cifar100_pyramid_ensemble_submodel_weak1':
        _labels_manipulation(cifar100_pyramid_ensemble_submodel_weak1_labels_manipulation),
    'cifar100_pyramid_ensemble_submodel_weak2':
        _labels_manipulation(cifar100_pyramid_ensemble_submodel_weak2_labels_manipulation)
}

_svhn_subnetworks: SubnetworksType = {
    'svhn_complicated_ensemble_submodel1':
        _labels_manipulation(svhn_complicated_ensemble_submodel1_labels_manipulation),
    'svhn_complicated_ensemble_submodel2':
        _labels_manipulation(svhn_complicated_ensemble_submodel2_labels_manipulation),
    'svhn_complicated_ensemble_submodel3':
        _labels_manipulation(svhn_complicated_ensemble_submodel3_labels_manipulation),
    'svhn_complicated_ensemble_submodel4':
        _labels_manipulation(svhn_complicated_ensemble_submodel4_labels_manipulation),
    'svhn_complicated_ensemble_submodel5':
        _labels_manipulation(svhn_complicated_ensemble_submodel5_labels_manipulation),
    'svhn_pyramid_ensemble_submodel_weak1':
        _labels_manipulation(svhn_pyramid_ensemble_submodel_weak1_labels_manipulation),
    'svhn_pyramid_ensemble_submodel_weak2':
        _labels_manipulation(svhn_pyramid_ensemble_submodel_weak2_labels_manipulation)
}

networks: NetworksType = dict(_cifar10_networks, **_cifar100_networks)
networks.update(_svhn_networks)

subnetworks: SubnetworksType = dict(_cifar10_subnetworks, **_cifar100_subnetworks)
subnetworks.update(_svhn_subnetworks)
