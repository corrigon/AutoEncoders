from AutoEncoders.vae import VAE
from model_ae import EncoderModel
from autoencoder import imgs_to_net, imgs_from_net

import numpy as np
import os
from glob import glob

import pickle

#import dataset.lfw
import h5py
from scipy.misc import imsave
import random
import shutil
from glob import glob
from scipy.misc import imread, imsave, imresize
from progress.bar import Bar
from sklearn.neighbors import NearestNeighbors
import copy

class VAEEncoderModel(EncoderModel):
    def __init__(self, model_name='', model_path=''):
        #vae_model_path = 'out/iGAN_Bags_reconganweight1.0e-06_recondepth9_realgenweight0.33_nodisaerecon/'
        vae_model_path = '/home/ubuntu/ALI_ours/snapshots_vae6_64_conv/51'
        self.vae = VAE(L=1, binary=False, imgshape=(64,64), channels=3, z_dim=256, n_hid=1024,
            encoder_type='conv2d_64_layer', decoder_type='conv2d_64_layer')
        print('Loading model from disk')
        self.vae.load_net(vae_model_path)

    def features_get_from_image(self, img):
        if img.ndim == 2:
            img = np.repeat(img[:,:,None],3,2)
        if img.shape[2] == 4:
            img = img[:,:,:3]
        img = imresize(img,(64,64))[None,...]
        z = self.vae.get_z(imgs_to_net(img[None,...]))
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
        z = self.vae.get_z(imgs, net_order=False)
        return z, paths

    def features_get_from_images(self, imgs):
        z = self.vae.get_z(imgs, net_order=False)
        return z

    def image_from_features(self, features):
        return self.vae.decode_imgs(features)
