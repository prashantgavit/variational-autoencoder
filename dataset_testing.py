import sys
import os
from torchvision.transforms import Compose, Resize, ToTensor

from meta_dataset.utils import Categorical,Rotation

from meta_dataset.omniglot import OmniglotClassDataset

root = "data"
num_classes_per_task=5
transform=Compose([Resize(28), ToTensor()])
target_transform=Categorical(num_classes=5)
class_augmentations=[Rotation([90, 180, 270])]
meta_train=True
download=True

meta_val=False
meta_test=False 
meta_split=None 
use_vinyals_split=True
dataset_transform=None

dataset = OmniglotVAEDataset(root, meta_train=meta_train,
    meta_val=meta_val, meta_test=meta_test,
    use_vinyals_split=use_vinyals_split, transform=transform,
    meta_split=meta_split, class_augmentations=class_augmentations,
    download=download)