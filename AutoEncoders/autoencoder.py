
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import h5py

import lasagne as nn
from scipy.misc import imsave
from scipy.misc import toimage

from PIL import Image
L = nn.layers

def save_samples(dec_imgs, grid, path):
    width = dec_imgs.shape[2]
    height = dec_imgs.shape[3]
    # sample from latent space and save as grid
    im = Image.new('RGB', (width*grid,height*grid))
    for i in range(grid*grid):
        x = i % grid
        y = i / grid
        img = dec_imgs[i]
        if img.shape[0]==1:
            img = np.vstack([img, img, img])
        img = img.transpose(1, 2, 0)
        im.paste(toimage(img), (x*width,y*height))
    im.save(path)

def imgs_to_net(imgs):
    return np.array(imgs).transpose(0,3,1,2).astype('float32')/255

def imgs_from_net(imgs):
    return (imgs.transpose(0,2,3,1)*255).clip(0,255).astype(np.uint8)

def iterate_minibatches_hdf5(inputs, indices, batchsize, shuffle=False,
    forever=False):
    if shuffle:
        sub_indices = np.arange(len(indices))
        np.random.shuffle(sub_indices)
    while True:
        for start_idx in range(0, len(indices) - batchsize, batchsize):
            if shuffle:
                excerpt = sub_indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield imgs_to_net(inputs[indices[excerpt].tolist()])
        if not forever:
            break

class AutoEncoder(object):
    def __init__(self):
        self.have_encoder = False
        print('Inited')

    def save_net(self, path):
        print('not implemented')
        raise

    def load_net(self, path):
        print('not implemented')
        raise

    def train_init(self, num_epochs):
        print('not implemented')
        raise

    def train_epoch(self, epoch_size, batch_size, train_batches, test_batches):
        print('not implemented')
        raise

    def gen_imgs(self, n_samples):
        print('not implemented')
        raise

    def reconstruct_imgs(self, imgs):
        print('not implemented')
        raise

    def get_z(self, imgs, net_order=True):
        print('not implemented')
        raise

    def save_generated_data(self, path, imgs):
            # And finally, we plot some generated data
            samples = self.gen_imgs(100)
            samples = (samples*255).clip(0,255).astype('uint8')
            #                        .reshape(6*28, 7*28)
            if not os.path.exists(path):
                os.mkdir(path)
            save_samples(samples, 10, path+'/gen.jpg')
            if self.have_encoder:
                imgs_to_rec = imgs_to_net(imgs[0:50])
                samples = self.reconstruct_imgs(imgs_to_rec)
                samples = (samples*255).clip(0,255).astype('uint8')
                all_samples = np.vstack([imgs_to_rec, samples])
                save_samples(all_samples, 10, path+'/reconstruct.jpg')

    def train(self, test_samples, hdf5_path, base_path, num_epochs=1000, \
            epochsize=300, batchsize=64, **kwargs):
        self.train_init(num_epochs, batchsize, **kwargs)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        # Load the dataset
        f = h5py.File(hdf5_path,'r')
        imgs = f['imgs']
        test_indices = np.array(range(0, test_samples))
        train_indices = np.array(range(test_samples, imgs.shape[0]))
        train_batches = iterate_minibatches_hdf5(imgs, train_indices, batchsize,
            shuffle=False, forever=True)
        for epoch in range(num_epochs):
            start_time = time.time()
            test_batches = iterate_minibatches_hdf5(imgs, test_indices, batchsize,
                shuffle=False, forever=False)
            self.train_epoch(epochsize, batchsize, train_batches, test_batches)
            # save some samples after the iteration
            self.save_generated_data(base_path+'/%d'%(epoch+1), imgs)
            if epoch % 50 == 0:
                self.save_net(base_path+'/%d'%(epoch+1))
        # finally save network params
        self.save_net(base_path)
