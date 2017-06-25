from model_ae import EncoderModel

import numpy as np
import os
from glob import glob

import pickle

#import dataset.lfw
import aegan
import h5py
from scipy.misc import imsave
import deeppy as dp
from dataset.util import img_transform
import cudarray as ca
import random
import shutil
from glob import glob
from scipy.misc import imread, imsave, imresize
from sklearn.model_selection import train_test_split
import deeppy.expr as ex
from progress.bar import Bar
from sklearn.neighbors import NearestNeighbors
from cvxpy import *
import copy

class AugmentedFeed(dp.Feed):
    @property
    def x_shape(self):
        return (self.batch_size,) + (self.x.shape[3],self.x.shape[1],self.x.shape[2])

    def batches(self):
        x = ca.empty(self.x_shape, dtype=dp.float_)
        for start, stop in self._batch_slices():
            if stop > start:
                x_np = self.x[start:stop]
            else:
                x_np = np.concatenate((self.x[start:], self.x[:stop]))
            if random.randint(0, 1) == 0:
                x_np = x_np[:, :, :, ::-1]
            x_np = x_np.astype('float32')
            x_np = img_transform(x_np, to_bc01=True)
            x_np = np.ascontiguousarray(x_np)
            ca.copyto(x, x_np)
            if hasattr(self, 'bar'):
                self.bar.next()
            yield x,

class VAEGANEncoderModel(EncoderModel):
    def __init__(self, model_name, model_path):
        #vae_model_path = 'out/iGAN_Bags_reconganweight1.0e-06_recondepth9_realgenweight0.33_nodisaerecon/'
        vae_model_path = 'out/Bags_aegan_nhidden256_reconganweight1.0e-06_recondepth9_realgenweight0.33_nodisaerecon/'
        model_file = os.path.join(model_path, model_name)
        print('Loading model from disk')
        print(model_file)
        with open(model_file, 'r') as f:
            self.model = pickle.load(f)

    def features_get_from_image(self, img):
        if img.ndim == 2:
            img = np.repeat(img[:,:,None],3,2)
        if img.shape[2] == 4:
            img = img[:,:,:3]
        img = imresize(img,(64,64))[None,...]
        test_feed = AugmentedFeed(img, 1, 1)
        z = self.model.encode(test_feed)
        return z[0]

    def features_get_from_path(self, path):
        paths = glob(glob_path)
        z_arr = []
        imgs_arr = []
        for path in paths:
            # print path
            img = imread(path)
            if img.ndim == 2:
                img = np.repeat(img[:,:,None],3,2)
            if img.shape[2] == 4:
                img = img[:,:,:3]
            img = imresize(img,(64,64))[None,...]
            imgs_arr.append(img)
        imgs = np.vstack(imgs_arr)
        print 'batches '+str(imgs.shape[0])
        test_feed = AugmentedFeed(imgs, imgs.shape[0], 1)
        z = self.model.encode(test_feed)
        return z, paths

    def features_get_from_images(self, imgs):
        test_feed = AugmentedFeed(imgs, 1000)
        test_feed.bar = Bar('Processing', max=test_feed.epoch_size, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        z = self.model.encode(test_feed)
        test_feed.bar.finish()
        return z

    def image_from_features(self, features):
        x_t = self.model.decoder(features)
        graph = ex.graph.ExprGraph(x_t)
        graph.setup()
        graph.fprop()
        return x_t.array
