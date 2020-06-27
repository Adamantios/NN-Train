from typing import Dict, Callable, Union, List, Optional, Any

from numpy.core.multiarray import ndarray
from tensorflow.python.keras import Model, Sequential

from networks.caltech101.complicated_ensemble.ensemble import caltech_complicated_ensemble
from networks.caltech101.complicated_ensemble.submodel1 import caltech_complicated_ensemble_submodel1, \
    caltech_complicated_ensemble_submodel1_labels_manipulation
from networks.caltech101.complicated_ensemble.submodel2 import caltech_complicated_ensemble_submodel2, \
    caltech_complicated_ensemble_submodel2_labels_manipulation
from networks.caltech101.complicated_ensemble.submodel3 import caltech_complicated_ensemble_submodel3, \
    caltech_complicated_ensemble_submodel3_labels_manipulation
from networks.caltech101.complicated_ensemble.submodel4 import caltech_complicated_ensemble_submodel4, \
    caltech_complicated_ensemble_submodel4_labels_manipulation
from networks.caltech101.complicated_ensemble.submodel5 import caltech_complicated_ensemble_submodel5, \
    caltech_complicated_ensemble_submodel5_labels_manipulation
from networks.caltech101.different_architectures.model1 import caltech_model1
from networks.caltech101.different_architectures.model2 import caltech_model2
from networks.caltech101.different_architectures.model3 import caltech_model3
from networks.caltech101.pyramid_ensemble.ensemble import caltech_pyramid_ensemble
from networks.caltech101.pyramid_ensemble.submodel_strong import caltech_pyramid_ensemble_submodel_strong
from networks.caltech101.pyramid_ensemble.submodel_weak1 import caltech_pyramid_ensemble_submodel_weak1, \
    caltech_pyramid_ensemble_submodel_weak1_labels_manipulation
from networks.caltech101.pyramid_ensemble.submodel_weak2 import caltech_pyramid_ensemble_submodel_weak2, \
    caltech_pyramid_ensemble_submodel_weak2_labels_manipulation
from networks.caltech101.students.strong import caltech_student_strong
from networks.caltech101.students.weak import caltech_student_weak
from networks.cifar10.baseline_ensemble.ensemble import cifar10_baseline_ensemble
from networks.cifar10.baseline_ensemble_v2.ensemble import cifar10_baseline_ensemble_v2
from networks.cifar10.baseline_ensemble_v2.submodel1 import cifar10_baseline_ensemble_v2_submodel1
from networks.cifar10.baseline_ensemble_v2.submodel2 import cifar10_baseline_ensemble_v2_submodel2
from networks.cifar10.baseline_ensemble_v2.submodel3 import cifar10_baseline_ensemble_v2_submodel3
from networks.cifar10.baseline_ensemble_v2.submodel4 import cifar10_baseline_ensemble_v2_submodel4
from networks.cifar10.baseline_ensemble_v2.submodel5 import cifar10_baseline_ensemble_v2_submodel5
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
from networks.cifar10.complicated_ensemble_v2.ensemble import cifar10_complicated_ensemble_v2
from networks.cifar10.complicated_ensemble_v2.submodel1 import cifar10_complicated_ensemble_v2_submodel1, \
    cifar10_complicated_ensemble_v2_submodel1_labels_manipulation
from networks.cifar10.complicated_ensemble_v2.submodel2 import cifar10_complicated_ensemble_v2_submodel2, \
    cifar10_complicated_ensemble_v2_submodel2_labels_manipulation
from networks.cifar10.complicated_ensemble_v2.submodel3 import cifar10_complicated_ensemble_v2_submodel3, \
    cifar10_complicated_ensemble_v2_submodel3_labels_manipulation
from networks.cifar10.complicated_ensemble_v2.submodel4 import cifar10_complicated_ensemble_v2_submodel4, \
    cifar10_complicated_ensemble_v2_submodel4_labels_manipulation
from networks.cifar10.complicated_ensemble_v2.submodel5 import cifar10_complicated_ensemble_v2_submodel5, \
    cifar10_complicated_ensemble_v2_submodel5_labels_manipulation
from networks.cifar10.different_architectures.ensemble import cifar10_architectures_diverse_ensemble
from networks.cifar10.different_architectures.model1 import cifar10_model1
from networks.cifar10.different_architectures.model2 import cifar10_model2
from networks.cifar10.different_architectures.model3 import cifar10_model3
from networks.cifar10.pyramid_ensemble.ensemble import cifar10_pyramid_ensemble
from networks.cifar10.pyramid_ensemble.submodel_strong import cifar10_pyramid_ensemble_submodel_strong
from networks.cifar10.pyramid_ensemble.submodel_weak1 import \
    cifar10_pyramid_ensemble_submodel_weak1_labels_manipulation, cifar10_pyramid_ensemble_submodel_weak1
from networks.cifar10.pyramid_ensemble.submodel_weak2 import \
    cifar10_pyramid_ensemble_submodel_weak2_labels_manipulation, cifar10_pyramid_ensemble_submodel_weak2
from networks.cifar10.strong_ensemble.ensemble import cifar10_strong_ensemble
from networks.cifar10.students.strong import cifar10_student_strong
from networks.cifar10.students.weak import cifar10_student_weak
from networks.cifar10.weak_ensemble.ensemble import cifar10_weak_ensemble
from networks.cifar100.baseline_ensemble.ensemble import cifar100_baseline_ensemble
from networks.cifar100.baseline_ensemble_v2.ensemble import cifar100_baseline_ensemble_v2
from networks.cifar100.baseline_ensemble_v2.submodel1 import cifar100_baseline_ensemble_v2_submodel1
from networks.cifar100.baseline_ensemble_v2.submodel2 import cifar100_baseline_ensemble_v2_submodel2
from networks.cifar100.baseline_ensemble_v2.submodel3 import cifar100_baseline_ensemble_v2_submodel3
from networks.cifar100.baseline_ensemble_v2.submodel4 import cifar100_baseline_ensemble_v2_submodel4
from networks.cifar100.baseline_ensemble_v2.submodel5 import cifar100_baseline_ensemble_v2_submodel5
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
from networks.cifar100.complicated_ensemble_v2.ensemble import cifar100_complicated_ensemble_v2
from networks.cifar100.complicated_ensemble_v2.submodel1 import cifar100_complicated_ensemble_v2_submodel1, \
    cifar100_complicated_ensemble_v2_submodel1_labels_manipulation
from networks.cifar100.complicated_ensemble_v2.submodel2 import cifar100_complicated_ensemble_v2_submodel2, \
    cifar100_complicated_ensemble_v2_submodel2_labels_manipulation
from networks.cifar100.complicated_ensemble_v2.submodel3 import cifar100_complicated_ensemble_v2_submodel3, \
    cifar100_complicated_ensemble_v2_submodel3_labels_manipulation
from networks.cifar100.complicated_ensemble_v2.submodel4 import cifar100_complicated_ensemble_v2_submodel4, \
    cifar100_complicated_ensemble_v2_submodel4_labels_manipulation
from networks.cifar100.complicated_ensemble_v2.submodel5 import cifar100_complicated_ensemble_v2_submodel5, \
    cifar100_complicated_ensemble_v2_submodel5_labels_manipulation
from networks.cifar100.different_architectures.ensemble import cifar100_architectures_diverse_ensemble
from networks.cifar100.different_architectures.model1 import cifar100_model1
from networks.cifar100.different_architectures.model2 import cifar100_model2
from networks.cifar100.different_architectures.model3 import cifar100_model3
from networks.cifar100.pyramid_ensemble.ensemble import cifar100_pyramid_ensemble
from networks.cifar100.pyramid_ensemble.submodel_strong import cifar100_pyramid_ensemble_submodel_strong
from networks.cifar100.pyramid_ensemble.submodel_weak1 import cifar100_pyramid_ensemble_submodel_weak1, \
    cifar100_pyramid_ensemble_submodel_weak1_labels_manipulation
from networks.cifar100.pyramid_ensemble.submodel_weak2 import cifar100_pyramid_ensemble_submodel_weak2, \
    cifar100_pyramid_ensemble_submodel_weak2_labels_manipulation
from networks.cifar100.strong_ensemble.ensemble import cifar100_strong_ensemble
from networks.cifar100.students.strong import cifar100_student_strong
from networks.cifar100.students.weak import cifar100_student_weak
from networks.cifar100.weak_ensemble.ensemble import cifar100_weak_ensemble
from networks.omniglot.complicated_ensemble.ensemble import omniglot_complicated_ensemble
from networks.omniglot.complicated_ensemble.submodel1 import omniglot_complicated_ensemble_submodel1, \
    omniglot_complicated_ensemble_submodel1_labels_manipulation
from networks.omniglot.complicated_ensemble.submodel2 import omniglot_complicated_ensemble_submodel2, \
    omniglot_complicated_ensemble_submodel2_labels_manipulation
from networks.omniglot.complicated_ensemble.submodel3 import omniglot_complicated_ensemble_submodel3, \
    omniglot_complicated_ensemble_submodel3_labels_manipulation
from networks.omniglot.complicated_ensemble.submodel4 import omniglot_complicated_ensemble_submodel4, \
    omniglot_complicated_ensemble_submodel4_labels_manipulation
from networks.omniglot.complicated_ensemble.submodel5 import omniglot_complicated_ensemble_submodel5, \
    omniglot_complicated_ensemble_submodel5_labels_manipulation
from networks.omniglot.different_architectures.model1 import omniglot_model1
from networks.omniglot.different_architectures.model2 import omniglot_model2
from networks.omniglot.different_architectures.model3 import omniglot_model3
from networks.omniglot.pyramid_ensemble.ensemble import omniglot_pyramid_ensemble
from networks.omniglot.pyramid_ensemble.submodel_strong import omniglot_pyramid_ensemble_submodel_strong
from networks.omniglot.pyramid_ensemble.submodel_weak1 import omniglot_pyramid_ensemble_submodel_weak1, \
    omniglot_pyramid_ensemble_submodel_weak1_labels_manipulation
from networks.omniglot.pyramid_ensemble.submodel_weak2 import omniglot_pyramid_ensemble_submodel_weak2, \
    omniglot_pyramid_ensemble_submodel_weak2_labels_manipulation
from networks.omniglot.students.strong import omniglot_student_strong
from networks.omniglot.students.weak import omniglot_student_weak
from networks.svhn.baseline_ensemble.ensemble import svhn_baseline_ensemble
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
from networks.svhn.complicated_ensemble_v2.ensemble import svhn_complicated_ensemble_v2
from networks.svhn.complicated_ensemble_v2.submodel1 import svhn_complicated_ensemble_v2_submodel1, \
    svhn_complicated_ensemble_v2_submodel1_labels_manipulation
from networks.svhn.complicated_ensemble_v2.submodel2 import svhn_complicated_ensemble_v2_submodel2, \
    svhn_complicated_ensemble_v2_submodel2_labels_manipulation
from networks.svhn.complicated_ensemble_v2.submodel3 import svhn_complicated_ensemble_v2_submodel3, \
    svhn_complicated_ensemble_v2_submodel3_labels_manipulation
from networks.svhn.complicated_ensemble_v2.submodel4 import svhn_complicated_ensemble_v2_submodel4, \
    svhn_complicated_ensemble_v2_submodel4_labels_manipulation
from networks.svhn.complicated_ensemble_v2.submodel5 import svhn_complicated_ensemble_v2_submodel5, \
    svhn_complicated_ensemble_v2_submodel5_labels_manipulation
from networks.svhn.different_architectures.ensemble import svhn_architectures_diverse_ensemble
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

# TODO: change networks params to have the same order and correct the networks type.
NetworksType = Dict[str, Callable[[Any, Any, Any, Optional[str]], Union[Model, Sequential]]]
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
    'cifar10_student_weak': cifar10_student_weak,
    'cifar10_baseline_ensemble': cifar10_baseline_ensemble,
    'cifar10_architectures_diverse_ensemble': cifar10_architectures_diverse_ensemble,
    'cifar10_strong_ensemble': cifar10_strong_ensemble,
    'cifar10_weak_ensemble': cifar10_weak_ensemble,
    'cifar10_complicated_ensemble_v2': cifar10_complicated_ensemble_v2,
    'cifar10_complicated_ensemble_v2_submodel1': cifar10_complicated_ensemble_v2_submodel1,
    'cifar10_complicated_ensemble_v2_submodel2': cifar10_complicated_ensemble_v2_submodel2,
    'cifar10_complicated_ensemble_v2_submodel3': cifar10_complicated_ensemble_v2_submodel3,
    'cifar10_complicated_ensemble_v2_submodel4': cifar10_complicated_ensemble_v2_submodel4,
    'cifar10_complicated_ensemble_v2_submodel5': cifar10_complicated_ensemble_v2_submodel5,
    'cifar10_baseline_ensemble_v2': cifar10_baseline_ensemble_v2,
    'cifar10_baseline_ensemble_v2_submodel1': cifar10_baseline_ensemble_v2_submodel1,
    'cifar10_baseline_ensemble_v2_submodel2': cifar10_baseline_ensemble_v2_submodel2,
    'cifar10_baseline_ensemble_v2_submodel3': cifar10_baseline_ensemble_v2_submodel3,
    'cifar10_baseline_ensemble_v2_submodel4': cifar10_baseline_ensemble_v2_submodel4,
    'cifar10_baseline_ensemble_v2_submodel5': cifar10_baseline_ensemble_v2_submodel5
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
    'cifar100_student_weak': cifar100_student_weak,
    'cifar100_baseline_ensemble': cifar100_baseline_ensemble,
    'cifar100_architectures_diverse_ensemble': cifar100_architectures_diverse_ensemble,
    'cifar100_strong_ensemble': cifar100_strong_ensemble,
    'cifar100_weak_ensemble': cifar100_weak_ensemble,
    'cifar100_complicated_ensemble_v2': cifar100_complicated_ensemble_v2,
    'cifar100_complicated_ensemble_v2_submodel1': cifar100_complicated_ensemble_v2_submodel1,
    'cifar100_complicated_ensemble_v2_submodel2': cifar100_complicated_ensemble_v2_submodel2,
    'cifar100_complicated_ensemble_v2_submodel3': cifar100_complicated_ensemble_v2_submodel3,
    'cifar100_complicated_ensemble_v2_submodel4': cifar100_complicated_ensemble_v2_submodel4,
    'cifar100_complicated_ensemble_v2_submodel5': cifar100_complicated_ensemble_v2_submodel5,
    'cifar100_baseline_ensemble_v2': cifar100_baseline_ensemble_v2,
    'cifar100_baseline_ensemble_v2_submodel1': cifar100_baseline_ensemble_v2_submodel1,
    'cifar100_baseline_ensemble_v2_submodel2': cifar100_baseline_ensemble_v2_submodel2,
    'cifar100_baseline_ensemble_v2_submodel3': cifar100_baseline_ensemble_v2_submodel3,
    'cifar100_baseline_ensemble_v2_submodel4': cifar100_baseline_ensemble_v2_submodel4,
    'cifar100_baseline_ensemble_v2_submodel5': cifar100_baseline_ensemble_v2_submodel5
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
    'svhn_student_weak': svhn_student_weak,
    'svhn_baseline_ensemble': svhn_baseline_ensemble,
    'svhn_architectures_diverse_ensemble': svhn_architectures_diverse_ensemble,
    'svhn_complicated_ensemble_v2': svhn_complicated_ensemble_v2,
    'svhn_complicated_ensemble_v2_submodel1': svhn_complicated_ensemble_v2_submodel1,
    'svhn_complicated_ensemble_v2_submodel2': svhn_complicated_ensemble_v2_submodel2,
    'svhn_complicated_ensemble_v2_submodel3': svhn_complicated_ensemble_v2_submodel3,
    'svhn_complicated_ensemble_v2_submodel4': svhn_complicated_ensemble_v2_submodel4,
    'svhn_complicated_ensemble_v2_submodel5': svhn_complicated_ensemble_v2_submodel5
}

_caltech_networks: NetworksType = {
    'caltech_model1': caltech_model1,
    'caltech_model2': caltech_model2,
    'caltech_model3': caltech_model3,
    'caltech_complicated_ensemble': caltech_complicated_ensemble,
    'caltech_complicated_ensemble_submodel1': caltech_complicated_ensemble_submodel1,
    'caltech_complicated_ensemble_submodel2': caltech_complicated_ensemble_submodel2,
    'caltech_complicated_ensemble_submodel3': caltech_complicated_ensemble_submodel3,
    'caltech_complicated_ensemble_submodel4': caltech_complicated_ensemble_submodel4,
    'caltech_complicated_ensemble_submodel5': caltech_complicated_ensemble_submodel5,
    'caltech_pyramid_ensemble': caltech_pyramid_ensemble,
    'caltech_pyramid_ensemble_submodel_strong': caltech_pyramid_ensemble_submodel_strong,
    'caltech_pyramid_ensemble_submodel_weak1': caltech_pyramid_ensemble_submodel_weak1,
    'caltech_pyramid_ensemble_submodel_weak2': caltech_pyramid_ensemble_submodel_weak2,
    'caltech_student_strong': caltech_student_strong,
    'caltech_student_weak': caltech_student_weak
}

_omniglot_networks: NetworksType = {
    'omniglot_model1': omniglot_model1,
    'omniglot_model2': omniglot_model2,
    'omniglot_model3': omniglot_model3,
    'omniglot_complicated_ensemble': omniglot_complicated_ensemble,
    'omniglot_complicated_ensemble_submodel1': omniglot_complicated_ensemble_submodel1,
    'omniglot_complicated_ensemble_submodel2': omniglot_complicated_ensemble_submodel2,
    'omniglot_complicated_ensemble_submodel3': omniglot_complicated_ensemble_submodel3,
    'omniglot_complicated_ensemble_submodel4': omniglot_complicated_ensemble_submodel4,
    'omniglot_complicated_ensemble_submodel5': omniglot_complicated_ensemble_submodel5,
    'omniglot_pyramid_ensemble': omniglot_pyramid_ensemble,
    'omniglot_pyramid_ensemble_submodel_strong': omniglot_pyramid_ensemble_submodel_strong,
    'omniglot_pyramid_ensemble_submodel_weak1': omniglot_pyramid_ensemble_submodel_weak1,
    'omniglot_pyramid_ensemble_submodel_weak2': omniglot_pyramid_ensemble_submodel_weak2,
    'omniglot_student_strong': omniglot_student_strong,
    'omniglot_student_weak': omniglot_student_weak
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
        _labels_manipulation(cifar10_pyramid_ensemble_submodel_weak2_labels_manipulation),
    'cifar10_complicated_ensemble_v2_submodel1':
        _labels_manipulation(cifar10_complicated_ensemble_v2_submodel1_labels_manipulation),
    'cifar10_complicated_ensemble_v2_submodel2':
        _labels_manipulation(cifar10_complicated_ensemble_v2_submodel2_labels_manipulation),
    'cifar10_complicated_ensemble_v2_submodel3':
        _labels_manipulation(cifar10_complicated_ensemble_v2_submodel3_labels_manipulation),
    'cifar10_complicated_ensemble_v2_submodel4':
        _labels_manipulation(cifar10_complicated_ensemble_v2_submodel4_labels_manipulation),
    'cifar10_complicated_ensemble_v2_submodel5':
        _labels_manipulation(cifar10_complicated_ensemble_v2_submodel5_labels_manipulation)
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
        _labels_manipulation(cifar100_pyramid_ensemble_submodel_weak2_labels_manipulation),
    'cifar100_complicated_ensemble_v2_submodel1':
        _labels_manipulation(cifar100_complicated_ensemble_v2_submodel1_labels_manipulation),
    'cifar100_complicated_ensemble_v2_submodel2':
        _labels_manipulation(cifar100_complicated_ensemble_v2_submodel2_labels_manipulation),
    'cifar100_complicated_ensemble_v2_submodel3':
        _labels_manipulation(cifar100_complicated_ensemble_v2_submodel3_labels_manipulation),
    'cifar100_complicated_ensemble_v2_submodel4':
        _labels_manipulation(cifar100_complicated_ensemble_v2_submodel4_labels_manipulation),
    'cifar100_complicated_ensemble_v2_submodel5':
        _labels_manipulation(cifar100_complicated_ensemble_v2_submodel5_labels_manipulation)
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
        _labels_manipulation(svhn_pyramid_ensemble_submodel_weak2_labels_manipulation),
    'svhn_complicated_ensemble_v2_submodel1':
        _labels_manipulation(svhn_complicated_ensemble_v2_submodel1_labels_manipulation),
    'svhn_complicated_ensemble_v2_submodel2':
        _labels_manipulation(svhn_complicated_ensemble_v2_submodel2_labels_manipulation),
    'svhn_complicated_ensemble_v2_submodel3':
        _labels_manipulation(svhn_complicated_ensemble_v2_submodel3_labels_manipulation),
    'svhn_complicated_ensemble_v2_submodel4':
        _labels_manipulation(svhn_complicated_ensemble_v2_submodel4_labels_manipulation),
    'svhn_complicated_ensemble_v2_submodel5':
        _labels_manipulation(svhn_complicated_ensemble_v2_submodel5_labels_manipulation)
}

_caltech_subnetworks: SubnetworksType = {
    'caltech_complicated_ensemble_submodel1':
        _labels_manipulation(caltech_complicated_ensemble_submodel1_labels_manipulation),
    'caltech_complicated_ensemble_submodel2':
        _labels_manipulation(caltech_complicated_ensemble_submodel2_labels_manipulation),
    'caltech_complicated_ensemble_submodel3':
        _labels_manipulation(caltech_complicated_ensemble_submodel3_labels_manipulation),
    'caltech_complicated_ensemble_submodel4':
        _labels_manipulation(caltech_complicated_ensemble_submodel4_labels_manipulation),
    'caltech_complicated_ensemble_submodel5':
        _labels_manipulation(caltech_complicated_ensemble_submodel5_labels_manipulation),
    'caltech_pyramid_ensemble_submodel_weak1':
        _labels_manipulation(caltech_pyramid_ensemble_submodel_weak1_labels_manipulation),
    'caltech_pyramid_ensemble_submodel_weak2':
        _labels_manipulation(caltech_pyramid_ensemble_submodel_weak2_labels_manipulation)
}

_omniglot_subnetworks: SubnetworksType = {
    'omniglot_complicated_ensemble_submodel1':
        _labels_manipulation(omniglot_complicated_ensemble_submodel1_labels_manipulation),
    'omniglot_complicated_ensemble_submodel2':
        _labels_manipulation(omniglot_complicated_ensemble_submodel2_labels_manipulation),
    'omniglot_complicated_ensemble_submodel3':
        _labels_manipulation(omniglot_complicated_ensemble_submodel3_labels_manipulation),
    'omniglot_complicated_ensemble_submodel4':
        _labels_manipulation(omniglot_complicated_ensemble_submodel4_labels_manipulation),
    'omniglot_complicated_ensemble_submodel5':
        _labels_manipulation(omniglot_complicated_ensemble_submodel5_labels_manipulation),
    'omniglot_pyramid_ensemble_submodel_weak1':
        _labels_manipulation(omniglot_pyramid_ensemble_submodel_weak1_labels_manipulation),
    'omniglot_pyramid_ensemble_submodel_weak2':
        _labels_manipulation(omniglot_pyramid_ensemble_submodel_weak2_labels_manipulation)
}

networks: NetworksType = dict(_cifar10_networks, **_cifar100_networks)
networks.update(_svhn_networks)
networks.update(_caltech_networks)
networks.update(_omniglot_networks)

subnetworks: SubnetworksType = dict(_cifar10_subnetworks, **_cifar100_subnetworks)
subnetworks.update(_svhn_subnetworks)
subnetworks.update(_caltech_subnetworks)
subnetworks.update(_omniglot_subnetworks)
