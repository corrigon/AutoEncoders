from model_ae import EncoderModel

from glob import glob
from scipy.misc import imread, imsave, imresize
from model_def.dcgan_theano import Model
import numpy as np
import theano
import theano.tensor as T
from lib.theano_utils import floatX
from lib.rng import np_rng
import os
import shutil
from sklearn.model_selection import train_test_split
import pickle
import h5py
from progress.bar import Bar

class IGANEncoderModel(EncoderModel):
    def __init__(self, model_name, model_file):
        #Load model
        self.model = Model(model_name=model_name, model_file=model_file)

        #compile f_enc
        x_s = T.ftensor4('imgs')
        z_tilda_sym = self.model.model_P(x_s)
        self.f_enc = theano.function([x_s], z_tilda_sym)

        #compile f_enc
        s_z = T.fmatrix('z')
        gen_img = self.model.model_G(s_z)
        self.f_dec = theano.function([s_z], gen_img)


    def features_get_from_image(self, img):
        if img.ndim == 2:
            img = np.repeat(img[:,:,None],3,2)
        if img.shape[2] == 4:
            img = img[:,:,:3]
        img = imresize(img,(64,64))[None,...]
        t_img = self.model.transform(img)
        z = self.f_enc(t_img)
        return z

    def features_get_from_path(self, path):
        paths = glob(glob_path)
        z_arr = []

        for path in paths:
            print path
            img = imread(path)
            z = self.features_get_from_image(img)
            z_arr.append(z)

        z = np.vstack(z_arr) #1000x100
        return z, paths

    def features_get_from_images(self, imgs):
        batch_size = 1000
        i = 0
        z_arr = []
        bar = Bar('Processing', max=imgs.shape[0]/batch_size, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for i in range(0, imgs.shape[0], batch_size):
            img = imgs[i:i+batch_size,...]
            t_img = self.model.transform(img)
            z = self.f_enc(t_img)
            bar.next()
            z_arr.append(z)
        bar.finish()
        z = np.vstack(z_arr) #1000x100
        return z

def get_feats_hdf5(imgs):
    batch_size = 1000
    i = 0
    bar = Bar('Processing', max=imgs.shape[0]/batch_size, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    for i in range(0, imgs.shape[0], batch_size):
        img = imgs[i:i+batch_size,...]
        t_img = self.model.transform(img)
        z = self.f_enc(t_img)
        bar.next()
        z_arr.append(z)
    bar.finish()
    z = np.vstack(z_arr) #1000x100
    return z

def image_from_features(self, features):
    return self.f_dec(features)
