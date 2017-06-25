import numpy as np
import os
from glob import glob

import pickle

#import dataset.lfw
#import aegan
import h5py
from scipy.misc import imsave
import random
import shutil
from glob import glob
from scipy.misc import imread, imsave, imresize
from sklearn.model_selection import train_test_split
from progress.bar import Bar
from sklearn.neighbors import NearestNeighbors
from cvxpy import *
import copy
#from vae_gan_model import VAEGANEncoderModel
from vae_model import VAEEncoderModel
#from igan_model import IGANEncoderModel

def get_db(path):
    dirs = glob(path)
    dbs = []
    hdf5_indices = []
    for i in range(len(dirs)):
        db = {}
        db['path'] = dirs[i]
        imgs_paths = glob(dirs[i]+'/*')
        db['images'] = []
        for j in range(len(imgs_paths)):
            img = {}
            img['path'] = imgs_paths[j]
            print imgs_paths[j]
            img_name = os.path.basename(imgs_paths[j])
            fields = img_name.split('_')
            print fields
            if fields[0]=='img':
                continue
            img['gt_order'] = int(fields[0])
            img['alg_order'] = int(fields[2])
            img['hdf5_idx'] = int(fields[3])
            img['alg_dist'] = float(fields[4].split('.jpg')[0])
            if img['gt_order']==0:
                db['origin'] = img
            else:
                db['images'].append(img)
            hdf5_indices.append(img['hdf5_idx'])
        if len(db['images'])>0:
            dbs.append(db)
    return dbs, hdf5_indices

def calc_alg_cost(db):
    cost_ = 0
    for db_ in db:
        db_cost = 0
        imgs = db_['images']
        for img in imgs:
            db_cost = db_cost + np.abs(img['gt_order']-img['alg_order'])/float(len(imgs)-1)
        cost_ = cost_ + db_cost / float(len(db))
    return cost_

def optimize_weights(db, z_all, alpha):
    # prepare data for optimization
    D = []
    for dbe in db:
        imgs = dbe['images']
        z_orig = z_all[dbe['origin']['hdf5_idx']]
        for i1 in range(len(imgs)-1):
            z_i1 = z_all[imgs[i1]['hdf5_idx']]
            for i2 in range(i1+1, len(imgs)):
                if imgs[i1]['gt_order'] < imgs[i2]['gt_order']:
                    y = 1
                else:
                    y = -1
                z_i2 = z_all[imgs[i2]['hdf5_idx']]
                d = y*(np.abs(z_orig-z_i1)-np.abs(z_orig-z_i2))
                D.append(d)
    # Construct the problem.
    D = np.vstack(D)
    w = Variable(D.shape[1])
    objective = Minimize(sum(D*w)/D.shape[1]+alpha*sum_squares(w))
    constraints = [0 <= w, sum(w)==1]
    prob = Problem(objective, constraints)
    # The optimal objective is returned by prob.solve().
    print "Optimizing weights"
    result = prob.solve()
    W = w.value
    W = np.maximum(0, W)
    W = np.asarray(W).flatten()
    return W

def random_alg(db):
    db_rand = copy.deepcopy(db)
    for dbe in db_rand:
        imgs = dbe['images']
        order = range(1, len(imgs)+1)
        np.random.shuffle(order)
        for i1 in range(len(imgs)):
            imgs[i1]['alg_order'] = order[i1]
    return db_rand

def perfect_alg(db):
    db_perf = copy.deepcopy(db)
    for dbe in db_perf:
        imgs = dbe['images']
        order = range(1, len(imgs)+1)
        np.random.shuffle(order)
        for i1 in range(len(imgs)):
            imgs[i1]['alg_order'] = imgs[i1]['gt_order']
    return db_perf

def apply_weights(z_all, W, db):
    db_opt = copy.deepcopy(db)
    neib = NearestNeighbors(n_neighbors=19, algorithm='kd_tree')
    for dbe in db_opt:
        z_fit = []
        z_query = []
        imgs = dbe['images']
        z_query.append(z_all[dbe['origin']['hdf5_idx'], ...]*np.sqrt(W))
        for i1 in imgs:
            z_fit.append(z_all[i1['hdf5_idx'], ...]*np.sqrt(W))

        z_fit = np.vstack(z_fit)
        z_query = np.vstack(z_query)
        neib.fit(z_fit)
        nn_res = neib.kneighbors(z_query, n_neighbors=len(imgs))
        for i1 in range(len(imgs)):
            imgs[nn_res[1][0][i1]]['alg_order'] = i1+1

    cost = calc_alg_cost(db_opt)
    return db_opt

db, hdf5_indices = get_db('/home/ubuntu/Data/similarity/db1/*')
cost = calc_alg_cost(db)

experiment_name = 'iGAN_Bags'
hdf5_path ='/home/ubuntu/Data/iGAN_bags/handbag_64.hdf5'
np.random.seed(1)

print('experiment_name: %s' % experiment_name)
#vae_model_path = 'out/iGAN_Bags_reconganweight1.0e-06_recondepth9_realgenweight0.33_nodisaerecon/'

#model = VAEGANEncoderModel('arch.pickle',
#    '/home/ubuntu/VAE_GAN/autoencoding_beyond_pixels/out/Bags_aegan_nhidden256_reconganweight1.0e-06_recondepth9_realgenweight0.33_nodisaerecon/')
#model = IGANEncoderModel('handbag_64',
#    '/home/ubuntu/iGAN/models/handbag_64.dcgan_theano')
model = VAEEncoderModel()

f = h5py.File(hdf5_path,'r')
h5_imgs = f['imgs']
z_all = model.features_get_from_images(h5_imgs)
f.close()

W = optimize_weights(db, z_all, 10)
db_opt = apply_weights(z_all, W, db)
cost_opt = calc_alg_cost(db_opt)
db_rand = random_alg(db)
cost_rand = calc_alg_cost(db_rand)
db_perf = perfect_alg(db)
cost_perf = calc_alg_cost(db_perf)
print('perfect cost %s, random cost %s, alg_cost %s, weighted_cost %s'%(
    cost_perf, cost_rand, cost, cost_opt))
