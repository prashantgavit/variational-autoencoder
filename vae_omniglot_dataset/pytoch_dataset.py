import os
import sys
import json
import glob
import h5py
from PIL import Image, ImageOps
from torchvision.datasets.utils import list_dir, download_url


class omniglotDataset:
    folder = 'omniglot'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    filename = 'data.hdf5'
    filename_labels = '{0}{1}_labels.json'

    def __init__(self, root):
        self.zips_md5 = {
            'images_background': '68d2efa1b9178cc56df9314c21c6e718',
            'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
        }
        self.root = root

        if not os.path.exists(self.root):
            self.download()
        
    def download(self):
        print('Downloading omniglot data')
        import zipfile
        import shutil

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

                index = []

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
                    # index.extend(filenames)
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        index.append([name,alphabet,character,i])
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)
                        
                with open(f'{self.root}/{split}_index.json', 'w') as f_labels:
                    json.dump(index, f_labels)
                

                # shutil.rmtree(os.path.join(self.root, name))