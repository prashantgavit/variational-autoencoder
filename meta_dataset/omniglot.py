from  meta_dataset.base_dataset import ClassDataset,Dataset
from torchvision.datasets.utils import list_dir, download_url
import os
import json
from torch.utils.data import Dataset as torch_dataset

from meta_dataset.utils import get_asset
import glob
import h5py
from PIL import Image, ImageOps
import torch

class OmniglotVAEDataset(torch_dataset):
    folder = 'omniglot_vae'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}{1}_labels.json'
    filename_classes = 'classes.json'

    def __init__(self, root, 
                 meta_train=False,
                 meta_test=False,
                 download=False):

        if (meta_train == meta_test):
            raise ValueError('both can not be the same, meta train and meta test')
        
        if meta_train:
            self.meta_split = 'train'
        else:
            self.meta_split = 'test'

        self.root = os.path.join(os.path.expanduser(root), self.folder)

        self.split_filename = os.path.join(self.root, self.filename)

        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format('vinyals_',
            self.meta_split))
        
        self.split_filename_classes = os.path.join(self.root,
            self.filename_classes)

        self._data = None
        self._labels = None
        self._classes = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        self._num_classes = len(self.labels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_data'] = None  # Don't pickle the file handle
        return state

    def __getitem__(self, index):
        # print(index)
        character_name = '/'.join(self.labels[index % self.num_classes][:-1])
        class_cat = '_'.join(self.labels[index % self.num_classes][:-1])
        # print(self.labels[index % self.num_classes])
        image_index = int(self.labels[index % self.num_classes][-1])
        data = self.data[character_name]
        # transform = self.get_transform(index, self.transform)
        # target_transform = self.get_target_transform(index)
        if self.classes.get(class_cat, None) == None:
            raise Exception('character_name does not exist in classes')
        else:
            img = data[image_index]
            img = torch.tensor(img, dtype=torch.float32) / 255.0  # <-- add this line
            return img, self.classes.get(class_cat)
        # return OmniglotDataset(index, data, character_name,
            # transform=transform, target_transform=target_transform)


    def __len__(self):
        return self.num_classes

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                # print(self.split_filename_labels)
                self._labels = json.load(f)
                # print(self._labels)
        return self._labels
    
    @property
    def classes(self):
        if self._classes is None:
            self._classes = {}
            with open(self.split_filename_classes, 'r') as f:
                loaded_classes = json.load(f)
                for index, class_cat in enumerate(loaded_classes):
                    self._classes['_'.join(class_cat)] = index
        return  self._classes

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import zipfile
        import shutil

        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = '{0}.zip'.format(name)
            filename = os.path.join(self.root, zip_filename)
            if os.path.isfile(filename):
                continue

            url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
            download_url(url, self.root, zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            for name in self.zips_md5:
                group = f.create_group(name)

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, name, alphabet))]

                split = 'train' if name == 'images_background' else 'test'
                labels_filename = os.path.join(self.root,
                    self.filename_labels.format('', split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                        alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                # shutil.rmtree(os.path.join(self.root, name))

        for split in ['train']:
            filename_train = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', 'train'))
            filename_test = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', 'test'))
            
            filename_class_pth = os.path.join(self.root, self.filename_classes)
            
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            labels_train = []
            labels_test = []
            classes = []
            for name, alphabets in data.items():
                for alphabet, characters in alphabets.items():
                    for character in characters:
                        classes.append(('images_{0}'.format(name), alphabet, character))
                        for i in range(15):
                            labels_train.append(('images_{0}'.format(name), alphabet, character, str(i)))
                        for i in range(15,20):
                            labels_test.append(('images_{0}'.format(name), alphabet, character, str(i)))
            labels = sorted(labels)
            classes = sorted(classes)

            with open(filename_train, 'w') as f:
                json.dump(labels_train, f)

            with open(filename_test, 'w') as f:
                json.dump(labels_test, f)

            with open(filename_class_pth,'w') as f:
                json.dump(classes,f)




class OmniglotClassDataset(ClassDataset):
    folder = 'omniglot'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}{1}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, use_vinyals_split=True, transform=None,
                 class_augmentations=None, download=False):
        super(OmniglotClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        if self.meta_val and (not use_vinyals_split):
            raise ValueError('Trying to use the meta-validation without the '
                'Vinyals split. You must set `use_vinyals_split=True` to use '
                'the meta-validation split.')

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format('vinyals_' if use_vinyals_split else '',
            self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index % self.num_classes])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniglotDataset(index, data, character_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import zipfile
        import shutil

        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = '{0}.zip'.format(name)
            filename = os.path.join(self.root, zip_filename)
            if os.path.isfile(filename):
                continue

            url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
            download_url(url, self.root, zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            for name in self.zips_md5:
                group = f.create_group(name)

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, name, alphabet))]

                split = 'train' if name == 'images_background' else 'test'
                labels_filename = os.path.join(self.root,
                    self.filename_labels.format('', split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                        alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                # shutil.rmtree(os.path.join(self.root, name))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', split))
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character)
                    for (name, alphabets) in data.items()
                    for (alphabet, characters) in alphabets.items()
                    for character in characters])
                json.dump(labels, f)

            vae_filename = os.path.join(self.root,'vae_' + self.filename_labels.format(
                'vinyals_', split))

            with open(vae_filename,'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character, str(i))
                    for (name, alphabets) in data.items()
                    for (alphabet, characters) in alphabets.items()
                    for character in characters]
                    for i in range(20))
                json.dump(labels, f)

class OmniglotDataset(Dataset):
    def __init__(self, index, data, character_name,
                 transform=None, target_transform=None):
        super(OmniglotDataset, self).__init__(index, transform=transform,
                                              target_transform=target_transform)
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print('I am here')
        image = Image.fromarray(self.data[index])
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)