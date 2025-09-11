from collections import defaultdict
import torchvision.transforms.functional as F
import torch
import os
import json

from PIL import Image, ImageOps


def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data



class TargetTransform(object):
    def __call__(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)

class Categorical(TargetTransform):
    """Target transform to return labels in `[0, num_classes)`.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If `None`, then the number of classes is inferred
        from the number of individual labels encountered.

    Examples
    --------
    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True)
    >>> task = dataset.sample_task()
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11EC797F0>,
    ('images_evaluation/Glagolitic/character12', None))

    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True,
    ... target_transform=Categorical(5))
    >>> task = dataset.sample_task()
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11ED3F668>, 2)
    """
    def __init__(self, num_classes=None):
        super(Categorical, self).__init__()
        self.num_classes = num_classes
        self._classes = None
        self._labels = None

    def reset(self):
        self._classes = None
        self._labels = None

    @property
    def classes(self):
        if self._classes is None:
            self._classes = defaultdict(None)
            if self.num_classes is None:
                default_factory = lambda: len(self._classes)
            else:
                default_factory = lambda: self.labels[len(self._classes)]
            self._classes.default_factory = default_factory
        if (self.num_classes is not None) and (len(self._classes) > self.num_classes):
            raise ValueError('The number of individual labels ({0}) is greater '
                'than the number of classes defined by `num_classes` '
                '({1}).'.format(len(self._classes), self.num_classes))
        return self._classes

    @property
    def labels(self):
        if (self._labels is None) and (self.num_classes is not None):
            # TODO: Replace torch.randperm with seed-friendly counterpart
            self._labels = torch.randperm(self.num_classes).tolist()
        return self._labels

    def __call__(self, target):
        return self.classes[target]

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.num_classes or '')



class Rotation(object):
    def __init__(self, angle, resample=False, expand=False, center=None):
        super(Rotation, self).__init__()
        if isinstance(angle, (list, tuple)):
            self._angles = angle
            self.angle = None
        else:
            self._angles = [angle]
            self.angle = angle
            if angle % 360 == 0:
                import warnings
                warnings.warn('Applying a rotation of {0} degrees (`{1}`) as a '
                    'class augmentation on a dataset is equivalent to the original '
                    'dataset.'.format(angle, self), UserWarning, stacklevel=2)

        self.resample = resample
        self.expand = expand
        self.center = center

    def __iter__(self):
        return iter(Rotation(angle, resample=self.resample, expand=self.expand,
            center=self.center) for angle in self._angles)

    def __call__(self, image):
        if self.angle is None:
            raise ValueError('The value of the angle is unspecified.')
        # QKFIX: Explicitly compute the pixel fill value due to an
        # incompatibility between Torchvision 0.5 and Pillow 7.0.0
        # https://github.com/pytorch/vision/issues/1759#issuecomment-583826810
        # Will be fixed in Torchvision 0.6
        fill = tuple([0] * len(image.getbands()))
        return F.rotate(image, self.angle % 360, self.resample,
                        self.expand, self.center, fill=fill)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if (self.angle is None) or (other.angle is None):
            return self._angles == other._angles
        return (self.angle % 360) == (other.angle % 360)

    def __repr__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle % 360)

    def __str__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle)
