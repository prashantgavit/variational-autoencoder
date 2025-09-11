import sys
import numpy as np
import warnings
from copy import deepcopy

from itertools import combinations
from ordered_set import OrderedSet

from torchvision.transforms import Compose
from torch.utils.data import Dataset as Dataset_


# from torchmeta.utils.data.task import ConcatTask
# from torchmeta.transforms import FixedCategory, Categorical, DefaultTargetTransform
# from torchmeta.transforms.utils import wrap_transform


class FixedCategory(object):
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, index):
        return (index, self.transform)

    def __repr__(self):
        return ('{0}({1})'.format(self.__class__.__name__, self.transform))

class ClassDataset(object):

    """Base class for a dataset of classes. Each item from a `ClassDataset` is 
    a dataset containing examples from the same class.

    Parameters
    ----------
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`, 
        `meta_val` and `meta_test`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g. `transforms.HorizontalFlip()`.
    """
    def __init__(self, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, class_augmentations=None):
        if meta_train + meta_val + meta_test == 0:
            if meta_split is None:
                raise ValueError('The meta-split is undefined. Use either the '
                    'argument `meta_train=True` (or `meta_val`/`meta_test`), or '
                    'the argument `meta_split="train"` (or "val"/"test").')
            elif meta_split not in ['train', 'val', 'test']:
                raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                    'must be in [`train`, `val`, `test`].'.format(meta_split))
            meta_train = (meta_split == 'train')
            meta_val = (meta_split == 'val')
            meta_test = (meta_split == 'test')
        elif meta_train + meta_val + meta_test > 1:
            raise ValueError('Multiple arguments among `meta_train`, `meta_val` '
                'and `meta_test` are set to `True`. Exactly one must be set to '
                '`True`.')
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test
        self._meta_split = meta_split

        if class_augmentations is not None:
            if not isinstance(class_augmentations, list):
                raise TypeError('Unknown type for `class_augmentations`. '
                    'Expected `list`, got `{0}`.'.format(type(class_augmentations)))
            unique_augmentations = OrderedSet()
            for augmentations in class_augmentations:
                for transform in augmentations:
                    if transform in unique_augmentations:
                        warnings.warn('The class augmentation `{0}` already '
                            'exists in the list of class augmentations (`{1}`). '
                            'To avoid any duplicate, this transformation is '
                            'ignored.'.format(transform, repr(transform)),
                            UserWarning, stacklevel=2)
                    unique_augmentations.add(transform)
            class_augmentations = list(unique_augmentations)
        else:
            class_augmentations = []
        self.class_augmentations = class_augmentations

    def get_class_augmentation(self, index):
        transform_index = (index // self.num_classes) - 1
        if transform_index < 0:
            return None
        return self.class_augmentations[transform_index]

    def get_transform(self, index, transform=None):
        class_transform = self.get_class_augmentation(index)
        if class_transform is None:
            return transform
        if transform is None:
            return class_transform
        return Compose([class_transform, transform])

    def get_target_transform(self, index):
        class_transform = self.get_class_augmentation(index)
        return FixedCategory(class_transform)

    @property
    def meta_split(self):
        if self._meta_split is None:
            if self.meta_train:
                self._meta_split = 'train'
            elif self.meta_val:
                self._meta_split = 'val'
            elif self.meta_test:
                self._meta_split = 'test'
            else:
                raise NotImplementedError()
        return self._meta_split

    def __getitem__(self, index):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_augmentations) + 1)

class Dataset(Dataset_):
    def __init__(self, index, transform=None, target_transform=None):
        self.index = index
        self.transform = transform
        self.target_transform = target_transform

    def target_transform_append(self, transform):
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])

    def __hash__(self):
        return hash(self.index)
