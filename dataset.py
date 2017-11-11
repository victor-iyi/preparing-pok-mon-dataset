"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 11 November, 2017 @ 06:27 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import os
import sys
import pickle

import numpy as np

from PIL import Image


class Dataset:
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    
    def __init__(self, data_dir, dest_dir='datasets/save/', size=256, background=None):
        self.data_dir = data_dir
        self.dest_dir = dest_dir
        self.size = size
        self._background = background if background else self.COLOR_WHITE
        # Next batch
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    def create(self, force=False):
        """
        Creates the dataset. into self.images
        
        :param force: boolean
            if file has already been created before, 
            forcefully create it again.
        """
        self._process(force=force)
        self._num_examples = self._X.shape[0]
        
    def save(self, save_file, force=False):
        """
        Saves the dataset object

        :param save_file: str
            path to a pickle file
        :param force: bool
            force saving
        """
        if os.path.isfile(save_file) and not force:
            raise FileExistsError('{} already exist. Set `force=True` to override.'.format(save_file))
        dirs = save_file.split('/')
        if len(dirs) > 1 and not os.path.isdir('/'.join(dirs[:-1])):
            os.makedirs('/'.join(dirs[:-1]))
        with open(save_file, mode='wb') as f:
            pickle.dump(self, f)

    def load(self, save_file):
        """
        Load a saved Dataset object

        :param save_file:
            path to a pickle file
        :return: obj:
            saved instance of Dataset
        """
        if not os.path.isfile(save_file):
            raise FileNotFoundError('{} was not found.'.format(save_file))
        with open(save_file, 'rb') as f:
            self = pickle.load(file=f)
        return self

    
    def next_batch(self, batch_size, shuffle=True):
        """
        Get the next batch in the dataset

        :param batch_size: int
            Number of batches to be retrieved
        :param shuffle: bool
            Randomly shuffle the batches returned
        :return:
            Returns `batch_size` batches
            features - np.array([batch_size, ?])
        """
        start = self._index_in_epoch
        # Shuffle for first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            permute = np.arange(self._num_examples)
            np.random.shuffle(permute)
            self._X = self._X[permute]
        # Go to next batch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_examples = self._num_examples - start
            rest_features = self._X[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                permute = np.arange(self._num_examples)
                np.random.shuffle(permute)
                self._X = self._X[permute]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_examples
            end = self._index_in_epoch
            features = np.concatenate((rest_features, self._X[start:end]), axis=0)
            return features
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end]
    
    @property
    def images(self):
        return self._X
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def _process(self, force):
        imgs = []
        try:
            # convert rgba to rgb (with black background)
            self.__rgba2rgb(force=force)
            files = [os.path.join(self.dest_dir, d) for d in os.listdir(self.dest_dir)]
            for file in files:
                img = Image.open(file)
                img = img.resize(size=(self.size, self.size))
                img = np.array(img, dtype=np.float32)
                imgs.append(img)
        except Exception as e:
            sys.stderr.write('\r{}'.format(e))
        self._X = np.array(imgs)
    
    def __rgba2rgb(self, force=False):
        if os.path.isdir(self.dest_dir) and len(os.listdir(self.dest_dir)) > 1 and not force:
            sys.stderr.write('{} already exist.'.format(self.dest_dir))
            sys.stderr.flush()
            return
        
        import shutil
        shutil.rmtree(self.dest_dir, ignore_errors=True)
        os.makedirs(self.dest_dir)
        files = os.listdir(self.data_dir)
        for i, each in enumerate(files):
            try:
                png = Image.open(os.path.join(self.data_dir, each))
                if png.mode == 'RGBA':
                    png.load() # required for png.split()
                    background = Image.new("RGB", png.size, color=self._background)
                    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
                    background.save(os.path.join(self.dest_dir, each.split('.')[0] + '.jpg'), 'JPEG')
                else:
                    png.convert('RGB')
                    png.save(os.path.join(self.dest_dir, each.split('.')[0] + '.jpg'), 'JPEG')
            except Exception as e:
                sys.stderr.write('{} – {}\n'.format(e, png.filename))
                os.unlink(os.path.join(self.dest_dir, each.split('.')[0] + '.jpg'))
            finally:
                sys.stdout.write('\r{:,} of {:,}'.format(i+1, len(files)))
        del files