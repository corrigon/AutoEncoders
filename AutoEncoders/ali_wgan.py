#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Wasserstein Generative Adversarial Networks
(WGANs, see https://arxiv.org/abs/1701.07875 for the paper and
https://github.com/martinarjovsky/WassersteinGAN for the "official" code).

It is based on a DCGAN example:
https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
This, in turn, is based on the MNIST example in Lasagne:
https://lasagne.readthedocs.io/en/latest/user/tutorial.html

Jan Schl�ter, 2017-02-02
"""

from __future__ import print_function
from autoencoder import AutoEncoder
from autoencoder import imgs_to_net

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
from progress.bar import Bar

from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, Conv2DLayer
try:
    from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
except ImportError:
    raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                      "version: http://lasagne.readthedocs.io/en/latest/"
                      "user/installation.html#bleeding-edge-version")
try:
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    from lasagne.layers import batch_norm
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import LeakyRectify
from PIL import Image
L = nn.layers

class ALI(AutoEncoder):
    def __init__(self, z_dim=100, channels=3, width=28, height=28, wgan_only=False, **kwargs):
        self.z_dim = z_dim
        self.channels = channels
        self.width = width
        self.height = height
        self.wgan_only = wgan_only
        self.have_encoder = not self.wgan_only
        self.build_model(**kwargs)
        self.encode_fn = None
        self.decode_fn = None

    def save_net(self, path):
        print('Saving network params from %s'%path)
        # Optionally, you could now dump the network weights to a file like this:
        np.savez(path+'/wgan_mnist_dec.npz', *L.get_all_param_values(self.decoder))
        np.savez(path+'/wgan_mnist_enc.npz', *L.get_all_param_values(self.encoder))
        np.savez(path+'/wgan_mnist_crit.npz', *L.get_all_param_values(self.critic))

    def load_net(self, path):
        print('Loading network params from %s'%path)
        #decoder
        values = np.load(path+'/wgan_mnist_dec.npz')
        l = []
        for i in range(len(values.files)):
            l.append(values['arr_%d'%i])
        L.set_all_param_values(self.decoder, l)
        #encoder
        values = np.load(path+'/wgan_mnist_enc.npz')
        l = []
        for i in range(len(values.files)):
            l.append(values['arr_%d'%i])
        L.set_all_param_values(self.encoder, l)
        #critic
        values = np.load(path+'/wgan_mnist_crit.npz')
        l = []
        for i in range(len(values.files)):
            l.append(values['arr_%d'%i])
        L.set_all_param_values(self.critic, l)

    def build_decoder_28(self, in_layer):
        lrelu = LeakyRectify(0.2)
        # fully-connected layer
        layer = batch_norm(DenseLayer(in_layer, 1024, nonlinearity=lrelu)) # original with relu
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 256*7*7, nonlinearity=lrelu)) # original with relu
        layer = ReshapeLayer(layer, ([0], 256, 7, 7))
        # two fractional-stride convolutions
        layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same',
                                         output_size=14, nonlinearity=lrelu)) # original with relu
        return Deconv2DLayer(layer, self.channels, 5, stride=2, crop='same', output_size=28,
              nonlinearity=None)

    def build_decoder(self, decoder_model='decoder_28', input_var=None):
        # input: 100dim
        layer = InputLayer(shape=(None, self.z_dim), input_var=input_var)
        if decoder_model=='decoder_28':
            layer = self.build_decoder_28(layer)
        else:
            raise 'unknown decoder model '+decoder_model
        print ("decoder output:", layer.output_shape)
        return layer


    def build_encoder_28(self, layer_in, encoder_mode='encoder_28'):
        lrelu = LeakyRectify(0.2)
        layer = batch_norm(Conv2DLayer(layer_in, 128, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        layer = batch_norm(Conv2DLayer(layer, 256, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        layer = ReshapeLayer(layer, ([0], 6272*2))
        layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
        return batch_norm(DenseLayer(layer, self.z_dim, nonlinearity=None))

    def build_encoder(self, encoder_model='encoder_28', input_var=None):
        # input: 3x28x28dim
        layer = InputLayer(shape=(None, self.channels, 28, 28), input_var=input_var)
        if encoder_model=='encoder_28':
            layer = self.build_encoder_28(layer)
        else:
            raise 'invalid encoder model: '+encoder_model
        print ("Encoder output:", layer.output_shape)
        return layer

    def build_critic_28(self, in_x_layer, in_z_layer):
        # two convolutions
        lrelu = LeakyRectify(0.2)
        layer = batch_norm(Conv2DLayer(in_x_layer, 128, 5, stride=2, pad='same',
                                       nonlinearity=lrelu))
        layer = batch_norm(Conv2DLayer(layer, 256, 5, stride=2, pad='same',
                                       nonlinearity=lrelu))
        layer = ReshapeLayer(layer, ([0], 6272*2))
        # fully-connected layer
        layer = L.ConcatLayer([layer, in_z_layer])
        layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
        # output layer (linear and without bias)
        return DenseLayer(layer, 1, nonlinearity=None, b=None)

    def build_critic(self, critic_model='critic_28', input_var=None, z_var=None):
        lrelu = LeakyRectify(0.2)
        # input: (None, 1, 28, 28)
        in_x_layer = InputLayer(shape=(None, self.channels, self.width, self.height),
            input_var=input_var)
        in_z_layer = InputLayer(shape=(None, self.z_dim), input_var=z_var)
        l_out_disc = None
        if critic_model=='critic_28':
            l_out_disc = self.build_critic_28(in_x_layer, in_z_layer)
        else:
            print('unknown critic '+critic_model)
            raise
        print ("critic output:", l_out_disc.output_shape)
        return l_out_disc, in_x_layer, in_z_layer

    # ############################## Main program ################################
    # Everything else will be handled in our main program now. We could pull out
    # more functions to better separate the code, but it wouldn't make it any
    # easier to read.
    def build_model(self, encoder_model=None, decoder_model=None, critic_model=None):
        # Prepare Theano variables for inputs and targets
        self.z_var = T.matrix('z')
        self.input_var = T.tensor4('inputs')
        self.z_zeros = T.zeros_like(self.z_var)

        # Create neural network model
        print("Building model and compiling functions...")
        self.decoder = self.build_decoder(decoder_model, self.z_var)
        self.encoder = self.build_encoder(encoder_model, self.input_var)
        self.encoder_out = L.get_output(self.encoder, self.input_var, deterministic=False)
        self.decoder_out = L.get_output(self.decoder, self.z_var, deterministic=False)
        self.critic, self.in_x_layer, self.in_z_layer = self.build_critic(critic_model, self.input_var, self.z_var)

    def gen_imgs(self, n_samples):
        z_samples = np.random.rand(n_samples, self.z_dim).astype('float32')
        return self.decode_fn(z_samples)

    def reconstruct_imgs(self, imgs):
        return self.decode_fn(self.encode_fn(imgs))

    def create_losses(self):
        if self.wgan_only:
            encoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.input_var, self.in_z_layer:self.z_zeros}, deterministic=False)
            decoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.decoder_out, self.in_z_layer:self.z_zeros}, deterministic=False)
        else:
            encoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.input_var, self.in_z_layer:self.encoder_out}, deterministic=False)
            decoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.decoder_out, self.in_z_layer:self.z_var}, deterministic=False)

        # Create score expressions to be maximized (i.e., negative losses)
        self.decoder_score = decoder_critic_out.mean()
        self.encoder_score = -encoder_critic_out.mean()
        self.critic_score = -self.encoder_score - self.decoder_score

    def compile_enc_dec(self):
        print('Compile encoder and decoder functions')
        # Compile another function generating some data
        self.decode_fn = theano.function([self.z_var], L.get_output(self.decoder,
                           deterministic=True))
        self.encode_fn = theano.function([self.input_var], L.get_output(self.encoder,
                           deterministic=True))

    def compile_critic(self):
        print('Compile critic function')
        # Compile another function generating some data
        self.critic_det_fn = theano.function([self.z_var, self.input_var],
            L.get_output(self.critic, {self.in_x_layer:self.input_var, self.in_z_layer:self.z_var},
            deterministic=True))
        if self.wgan_only:
            encoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.input_var, self.in_z_layer:self.z_zeros}, deterministic=True)
            decoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.decoder_out, self.in_z_layer:self.z_zeros}, deterministic=True)
        else:
            encoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.input_var, self.in_z_layer:self.encoder_out}, deterministic=True)
            decoder_critic_out = L.get_output(self.critic, {self.in_x_layer:self.decoder_out, self.in_z_layer:self.z_zeros}, deterministic=True)
        self.decoder_score_det = decoder_critic_out.mean()
        self.encoder_score_det = -encoder_critic_out.mean()
        self.critic_score_det = -self.encoder_score_det - self.decoder_score_det
        self.encoder_score_det_fn = theano.function([self.input_var],
            self.encoder_score_det)
        self.decoder_score_det_fn = theano.function([self.z_var],
            self.decoder_score_det)

    def train_init(self, num_epochs, batch_size, train_decoder=True,
        train_encoder=True, initial_eta=1e-4, initial_critic_steps=50000,
        start_eta_decrease=0):
        self.batch_size = batch_size
        self.train_decoder = train_decoder
        self.train_encoder = train_encoder
        self.create_losses()
#    def train(self, base_path, wgan_only=False, train_encoder=False, train_decoder=True, num_epochs=1000, initial_critic_steps=50):
        self.initial_eta=initial_eta
        self.eta_value = initial_eta
        self.initial_critic_steps = initial_critic_steps
        self.start_eta_decrease = start_eta_decrease
        clip=0.03
        # Create update expressions for training
        self.eta = theano.shared(nn.utils.floatX(self.initial_eta))
        decoder_params = L.get_all_params(self.decoder, trainable=True)
        self.decoder_updates = nn.updates.rmsprop(
                -self.decoder_score, decoder_params, learning_rate=self.eta)
        decoder_gradients = T.grad(self.decoder_score, decoder_params)
        critic_params = L.get_all_params(self.critic, trainable=True)
        critic_updates = nn.updates.rmsprop(
                -self.critic_score, critic_params, learning_rate=self.eta)
        critic_gradients = T.grad(self.critic_score, critic_params)
        if not self.wgan_only:
            encoder_params = L.get_all_params(self.encoder, trainable=True)
            encoder_updates = nn.updates.rmsprop(
                -self.encoder_score, encoder_params, learning_rate=self.eta)
            encoder_gradients = T.grad(self.encoder_score, encoder_params)
        # Clip critic parameters in a limited range around zero (except biases)
        for param in L.get_all_params(self.critic, trainable=True,
                                                   regularizable=True):
            critic_updates[param] = T.clip(critic_updates[param], -clip, clip)

        # Instantiate a symbolic noise decoder to use for training
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
        noise = srng.normal((self.batch_size, self.z_dim))
        print('Compile training functions')
        # Compile functions performing a training step on a mini-batch (according
        # to the updates dictionary) and returning the corresponding score:
        self.decoder_train_fn = theano.function([], self.decoder_score,
                                             givens={self.z_var: noise},
                                             updates=self.decoder_updates)
        self.critic_train_fn = theano.function([self.input_var], (self.critic_score,
                                        self.encoder_score, self.decoder_score),
                                          givens={self.z_var: noise},
                                          updates=critic_updates)

        all_updates = critic_updates.copy()
        if self.train_decoder:
            all_updates.update(self.decoder_updates)
        if self.train_encoder:
            all_updates.update(encoder_updates)
        self.all_train_fn = theano.function([self.input_var], (self.critic_score, self.decoder_score,
                                         self.encoder_score),
                                         givens={self.z_var: noise},
                                         updates=all_updates)
        self.compile_enc_dec()
        encode_fn = None
        self.encoder_train_fn = None
        if not self.wgan_only:
            self.encoder_train_fn = theano.function([self.input_var], self.encoder_score,
                                             updates=encoder_updates)
            encode_fn = self.encode_fn
        # Finally, launch the training loop.
        print("Prepare minibatches...")
        # We create an infinite supply of batches (as an iterable decoder):

        # We iterate over epochs:
        self.n_decoder_updates = 0
        self.n_encoder_updates = 0
        self.n_critic_updates = 0
        print("Starting training...")
        self.max_encoder_steps = 1
        self.max_decoder_steps = 1
        self.max_critic_steps = 1000
        self.epoch = 0
        self.num_epochs = num_epochs

        print('initial critic run %d steps'%self.initial_critic_steps)

    def train_epoch(self, epochsize, batch_size, train_batches, test_batches):
        if batch_size != self.batch_size:
            raise 'batch size cannot change'
        start_time = time.time()

        # In each epoch, we do `epochsize` decoder updates. Usually, the
        # critic is updated 5 times before every decoder update. For the
        # first 25 decoder updates and every 500 decoder updates, the
        # critic is updated 100 times instead, following the authors' code.
        critic_scores = []
        decoder_scores = []
        encoder_scores = []
        bar = Bar('Training epoch %d lr %f'%(self.epoch, self.eta_value),
            max=epochsize, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for ep_batch in range(epochsize):
            bar.next()
            # update
            batch = next(train_batches)
            inputs = batch
            critic_score, decoder_score, encoder_score = self.all_train_fn(inputs)
            critic_scores.append(critic_score)
            encoder_scores.append(encoder_score)
            decoder_scores.append(decoder_score)
            self.n_critic_updates += 1
            if self.train_decoder:
                self.n_decoder_updates += 1
            if self.train_encoder:
                self.n_encoder_updates += 1
            if (self.epoch % 10 != 0) or (ep_batch!=0):
                continue
            # train critic
            if (self.epoch == 0):
                critic_runs = self.initial_critic_steps
            else:
                critic_runs = self.max_critic_steps
            for cr_run in range(critic_runs):
                batch = next(train_batches)
                inputs = batch
                crit_score, enc_score, dec_score = self.critic_train_fn(inputs)
                critic_scores.append(crit_score)
                self.n_critic_updates += 1
                if cr_run % 1000 == 999:
                    print('critic run %d crit_score: %f enc_score %f dec_score %f'%(cr_run, crit_score, enc_score, dec_score))
                if enc_score<-0.1 and dec_score<-0.1:
                    break
            # train decoder
            if self.train_decoder:
                for _ in range(self.max_decoder_steps):
                    score = self.decoder_train_fn()
                    decoder_scores.append(score)
                    self.n_decoder_updates += 1
                    if score > 0:
                        break
            # train encoder
            if (not self.wgan_only) and self.train_encoder:
                for _ in range(self.max_encoder_steps):
                    batch = next(train_batches)
                    inputs = batch
                    score = self.encoder_train_fn(inputs)
                    encoder_scores.append(score)
                    self.n_encoder_updates += 1
                    if score > 0:
                        break
        bar.finish()
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            self.epoch + 1, self.num_epochs, time.time() - start_time))
        print("  decoder score:\t\t{}".format(np.mean(decoder_scores)))
        print("  encoder score:\t\t{}".format(np.mean(encoder_scores)))
        print("  Wasserstein distance:\t\t{}".format(np.mean(critic_scores)))
        print("  critic updates: %d\tdecoder updates %d\tencoder updates: %d"%(self.n_critic_updates, self.n_decoder_updates, self.n_encoder_updates))
        # save some samples after the iteration

        # After some epochs, we start decaying the learn rate towards zero
        if self.epoch >= self.start_eta_decrease:
            progress = float(self.epoch-self.start_eta_decrease) / \
                (self.num_epochs-self.start_eta_decrease)
            self.eta_value = nn.utils.floatX(self.initial_eta*(1 - progress))
            self.eta.set_value(self.eta_value)
        self.epoch += 1
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # L.set_all_param_values(network, param_values)

# critic_out = L.get_output(ali.critic, {ali.in_x_layer:ali.input_var, ali.in_z_layer:ali.z_var},
#     deterministic=True)
# ali.critic_fn = theano.function([ali.z_var, ali.input_var], critic_out)

# decoder_score = critic_out.mean()
# ali.encoder_score = -encoder_critic_out.mean()
# ali.critic_score = -ali.encoder_score - ali.decoder_score
