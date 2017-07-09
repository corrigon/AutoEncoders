import sys
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import time
import h5py
from PIL import Image
from scipy.stats import norm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.misc import imsave
from scipy.misc import toimage
from autoencoder import AutoEncoder
from autoencoder import imgs_to_net

# ############################################################################
# Tencia Lee
# Some code borrowed from:
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
#
# Implementation of variational autoencoder (AEVB) algorithm as in:
# [1] arXiv:1312.6114 [stat.ML] (Diederik P Kingma, Max Welling 2013)

# ################## Download and prepare the MNIST dataset ##################
# For the linked MNIST data, the autoencoder learns well only in binary mode.
# This is most likely due to the distribution of the values. Most pixels are
# either very close to 0, or very close to 1.
#
# Running this code with default settings should produce a manifold similar
# to the example in this directory. An animation of the manifold's evolution
# can be found here: https://youtu.be/pgmnCU_DxzM

class GaussianSampleLayer(nn.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)

# class GaussianSampleLayer(nn.layers.MergeLayer):
#     def __init__(self, mu, logsigma, rng=None, **kwargs):
#         self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1,2147462579))
#         super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)
#
#     def get_output_shape_for(self, input_shapes):
#         return input_shapes[0]
#
#     def get_output_for(self, inputs, deterministic=False, **kwargs):
#         mu, logsigma = inputs
#         shape=(self.input_shapes[0][0] or inputs[0].shape[0],
#                 self.input_shapes[0][1] or inputs[0].shape[1],
#                 self.input_shapes[0][2] or inputs[0].shape[2],
#                 self.input_shapes[0][3] or inputs[0].shape[3]
#                 )
#         if deterministic:
#             return mu
#         return mu + T.exp(logsigma) * self.rng.normal(shape)




# ##################### Custom layer for middle of VCAE ######################
# This layer takes the mu and sigma (both DenseLayers) and combines them with
# a random vector epsilon to sample values for a multivariate Gaussian


# ############################## Build Model #################################
# encoder has 1 hidden layer, where we get mu and sigma for Z given an inp X
# continuous decoder has 1 hidden layer, where we get mu and sigma for X given code Z
# binary decoder has 1 hidden layer, where we calculate p(X=1)
# once we have (mu, sigma) for Z, we sample L times
# Then L separate outputs are constructed and the final layer averages them
def log_likelihood(tgt, mu, ls):
    return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
            - 0.5 * T.sqr(tgt - mu) / T.exp(2 * ls))

class VAE(AutoEncoder):
    def __init__(self, **kwargs):
        self.have_encoder = True
        self.input_var = T.tensor4('inputs')
        self.z_var = T.matrix()
        self.l_z_mu, self.l_z_ls, self.l_x_mu_list, self.l_x_ls_list, self.l_x_list, \
            self.l_x = self.build_vae(self.input_var, **kwargs)
        self.compile_inference_funcs()

    def build_decoder_last_layer(self, l_dec_hid, params):
        x_dim = self.width * self.height * self.channels
        l_dec_mu = nn.layers.DenseLayer(l_dec_hid, num_units=x_dim,
                nonlinearity = None,
                W = nn.init.GlorotUniform() if params is None else params['W_dec_mu'],
                b = nn.init.Constant(0) if params is None else params['b_dec_mu'],
                name = 'dec_mu')
        # relu_shift is for numerical stability - if training data has any
        # dimensions where stdev=0, allowing logsigma to approach -inf
        # will cause the loss function to become NAN. So we set the limit
        # stdev >= exp(-1 * relu_shift)
        relu_shift = 10
        l_dec_logsigma = nn.layers.DenseLayer(l_dec_hid, num_units=x_dim,
                W = nn.init.GlorotUniform() if params is None else params['W_dec_ls'],
                b = nn.init.Constant(0) if params is None else params['b_dec_ls'],
                nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                name='dec_logsigma')
        if params is None:
            params = {}
            params['W_dec_mu'] = l_dec_mu.W
            params['b_dec_mu'] = l_dec_mu.b
            params['W_dec_ls'] = l_dec_logsigma.W
            params['b_dec_ls'] = l_dec_logsigma.b
        return l_dec_mu, l_dec_logsigma, params

    def build_encoder_single_hidden(self, l_input, n_hid):
        return nn.layers.DenseLayer(l_input, num_units=n_hid,
                nonlinearity=nn.nonlinearities.tanh if self.binary else T.nnet.softplus,
                name='enc_hid')

    def build_decoder_single_hidden(self, l_Z, n_hid, params):
        l_dec_hid = nn.layers.DenseLayer(l_Z, num_units=n_hid,
                nonlinearity = nn.nonlinearities.tanh if self.binary else T.nnet.softplus,
                W=nn.init.GlorotUniform() if params is None else params['W_hid'],
                b=nn.init.Constant(0.) if params is None else params['b_hid'],
                name='dec_hid')
        _params = params
        l_dec_mu, l_dec_logsigma, __params = self.build_decoder_last_layer(l_dec_hid, params)
        if params==None:
            _params = {}
            _params['W_hid'] = l_dec_hid.W
            _params['b_hid'] = l_dec_hid.b
        _params.update(__params)
        return l_dec_mu, l_dec_logsigma, _params

    def build_encoder_conv2d_32_hidden(self, l_input):
        from lasagne.nonlinearities import sigmoid
        from lasagne.nonlinearities import LeakyRectify
        from lasagne.layers import Conv2DLayer
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        # input: 3x28x28dim
        lrelu = LeakyRectify(0.2)
        layer = batch_norm(Conv2DLayer(l_input, 64, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        return ReshapeLayer(layer, ([0], 6272))

    def build_decoder_conv2d_32_hidden(self, l_Z, params):
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
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
        lrelu = LeakyRectify(0.2)
        # fully-connected layer
        layer = batch_norm(DenseLayer(l_Z, 1024, nonlinearity=lrelu,
                W=nn.init.GlorotUniform() if params is None else params['w1'],
                b=nn.init.Constant(0.) if params is None else params['b1'],
        )) # original with relu
        _params = {}
        _params['w1'] = layer.input_layer.input_layer.W
        _params['b1'] = layer.input_layer.input_layer.b
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 128*7*7, nonlinearity=lrelu,
            W=nn.init.GlorotUniform() if params is None else params['w2'],
            b=nn.init.Constant(0.) if params is None else params['b2'],
        )) # original with relu
        _params['w2'] = layer.input_layer.input_layer.W
        _params['b2'] = layer.input_layer.input_layer.b
        layer = ReshapeLayer(layer, ([0], 128, 7, 7))
        # two fractional-stride convolutions
        layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                        output_size=14, nonlinearity=lrelu,
                        W=nn.init.GlorotUniform() if params is None else params['w3'],
                        b=nn.init.Constant(0.) if params is None else params['b3']
                                         )) # original with relu

        _params['w3'] = layer.input_layer.input_layer.W
        _params['b3'] = layer.input_layer.input_layer.b
        layer = Deconv2DLayer(layer, self.channels, 5, stride=2, crop='same', output_size=28,
                            nonlinearity=sigmoid,
                            W=nn.init.GlorotUniform() if params is None else params['w4'],
                            b=nn.init.Constant(0.) if params is None else params['b4']
                              )
        _params['w4'] = layer.W
        _params['b4'] = layer.b
        l_dec_hid = ReshapeLayer(layer, ([0], self.width*self.height*self.channels))
        l_dec_mu, l_dec_logsigma, __params = self.build_decoder_last_layer(l_dec_hid, params)
        _params.update(__params)
        return l_dec_mu, l_dec_logsigma, _params

    def build_encoder_conv2d_64_hidden(self, l_input):
        from lasagne.nonlinearities import sigmoid
        from lasagne.nonlinearities import LeakyRectify
        from lasagne.layers import Conv2DLayer
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        # input: 3x64x64dim
        lrelu = LeakyRectify(0.2)
        layer = batch_norm(Conv2DLayer(l_input, 64, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        # shape 64x32x32
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        # shape 128x16x16
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        # shape 128x8x8=8192
        return ReshapeLayer(layer, ([0], 8192))


    def build_decoder_conv2d_64_hidden(self, l_Z, params):
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
        try:
            from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        except ImportError:
            raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                              "version: http://lasagne.readthedocs.io/en/latest/"
                              "user/installation.html#bleeding-edge-version")
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_nor
        from lasagne.nonlinearities import sigmoid
        from lasagne.nonlinearities import LeakyRectify
        lrelu = LeakyRectify(0.2)
        # fully-connected layer
        layer = batch_norm(DenseLayer(l_Z, 1024, nonlinearity=lrelu,
                W=nn.init.GlorotUniform() if params is None else params['w1'],
                b=nn.init.Constant(0.) if params is None else params['b1'],
        )) # original with relu
        _params = {}
        _params['w1'] = layer.input_layer.input_layer.W
        _params['b1'] = layer.input_layer.input_layer.b
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 128*8*8, nonlinearity=lrelu,
            W=nn.init.GlorotUniform() if params is None else params['w2'],
            b=nn.init.Constant(0.) if params is None else params['b2'],
        )) # original with relu
        _params['w2'] = layer.input_layer.input_layer.W
        _params['b2'] = layer.input_layer.input_layer.b
        layer = ReshapeLayer(layer, ([0], 128, 8, 8))
        # two fractional-stride convolutions
        layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same',
                        output_size=16, nonlinearity=lrelu,
                        W=nn.init.GlorotUniform() if params is None else params['w3'],
                        b=nn.init.Constant(0.) if params is None else params['b3']
                                         )) # original with relu
        _params['w3'] = layer.input_layer.input_layer.W
        _params['b3'] = layer.input_layer.input_layer.b
        layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                        output_size=32, nonlinearity=lrelu,
                        W=nn.init.GlorotUniform() if params is None else params['w4'],
                        b=nn.init.Constant(0.) if params is None else params['b4']
                                         )) # original with relu
        _params['w4'] = layer.input_layer.input_layer.W
        _params['b4'] = layer.input_layer.input_layer.b
        layer = Deconv2DLayer(layer, self.channels, 5, stride=2, crop='same', output_size=64,
                            nonlinearity=sigmoid,
                            W=nn.init.GlorotUniform() if params is None else params['w5'],
                            b=nn.init.Constant(0.) if params is None else params['b5']
                              )
        _params['w5'] = layer.W
        _params['b5'] = layer.b

        l_dec_hid = ReshapeLayer(layer, ([0], self.width*self.height*self.channels))
        l_dec_mu, l_dec_logsigma, __params = self.build_decoder_last_layer(l_dec_hid, params)
        _params.update(__params)
        return l_dec_mu, l_dec_logsigma, _params

    def build_encoder_conv2d_128_hidden(self, l_input):
        from lasagne.nonlinearities import sigmoid
        from lasagne.nonlinearities import LeakyRectify
        from lasagne.layers import Conv2DLayer
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        # input: 3x128x128dim
        lrelu = LeakyRectify(0.2)
        layer = batch_norm(Conv2DLayer(l_input, 128, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        # shape 128x64x64
        layer = batch_norm(Conv2DLayer(layer, 256, 5, stride=2, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        # shape 256x32x32
        layer = batch_norm(Conv2DLayer(layer, 256, 7, stride=4, pad='same',
                                       nonlinearity=lrelu)) # original with relu
        # shape 256x8x8=8192
        return ReshapeLayer(layer, ([0], 8192*2))

    def build_decoder_conv2d_128_hidden(self, l_Z, params):
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
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
        lrelu = LeakyRectify(0.2)
        # fully-connected layer
        layer = batch_norm(DenseLayer(l_Z, 4096, nonlinearity=lrelu,
                W=nn.init.GlorotUniform() if params is None else params['w1'],
                b=nn.init.Constant(0.) if params is None else params['b1'],
        )) # original with relu
        _params = {}
        _params['w1'] = layer.input_layer.input_layer.W
        _params['b1'] = layer.input_layer.input_layer.b
        # project and reshape
        # shape 1024
        layer = batch_norm(DenseLayer(layer, 256*8*8, nonlinearity=lrelu,
            W=nn.init.GlorotUniform() if params is None else params['w2'],
            b=nn.init.Constant(0.) if params is None else params['b2'],
        # shape 256x8x8
        )) # original with relu
        _params['w2'] = layer.input_layer.input_layer.W
        _params['b2'] = layer.input_layer.input_layer.b
        layer = ReshapeLayer(layer, ([0], 256, 8, 8))
        # two fractional-stride convolutions
        layer = batch_norm(Deconv2DLayer(layer, 256, 7, stride=4, crop='same',
                        output_size=32, nonlinearity=lrelu,
                        W=nn.init.GlorotUniform() if params is None else params['w3'],
                        b=nn.init.Constant(0.) if params is None else params['b3']
                                         )) # original with relu
        # shape 256x32x32
        _params['w3'] = layer.input_layer.input_layer.W
        _params['b3'] = layer.input_layer.input_layer.b
        _layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same',
                        output_size=64, nonlinearity=lrelu,
                        W=nn.init.GlorotUniform() if params is None else params['w4'],
                        b=nn.init.Constant(0.) if params is None else params['b4']
                                         )) # original with relu
        # shape 128x64x64
        _params['w4'] = _layer.input_layer.input_layer.W
        _params['b4'] = _layer.input_layer.input_layer.b
#                            nonlinearity=sigmoid,
        layer = Deconv2DLayer(_layer, self.channels, 5, stride=2, crop='same', output_size=128,
                            nonlinearity=None,
                            untie_biases = True,
                            W=nn.init.GlorotUniform() if params is None else params['w_mu'],
                            b=nn.init.Constant(0.) if params is None else params['b_mu']
                              )
        _params['w_mu'] = layer.W
        _params['b_mu'] = layer.b
        l_dec_mu = ReshapeLayer(layer, ([0], self.width*self.height*self.channels))
        # relu_shift is for numerical stability - if training data has any
        # dimensions where stdev=0, allowing logsigma to approach -inf
        # will cause the loss function to become NAN. So we set the limit
        # stdev >= exp(-1 * relu_shift)
        relu_shift = 10
        layer = Deconv2DLayer(_layer, self.channels, 5, stride=2, crop='same', output_size=128,
                            nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                            W=nn.init.GlorotUniform() if params is None else params['w_logsigma'],
                            b=nn.init.Constant(0.) if params is None else params['b_logsigma']
                              )
        _params['w_logsigma'] = layer.W
        _params['b_logsigma'] = layer.b
        l_dec_logsigma = ReshapeLayer(layer, ([0], self.width*self.height*self.channels))
        # shape 3x128x128
        return l_dec_mu, l_dec_logsigma, _params

    def build_vae(self, inputvar, L=2, binary=True, imgshape=(28,28), \
            encoder_type='single_layer', decoder_type='single_layer', channels=1, z_dim=2, n_hid=1024):
        self.L = L
        self.binary = binary
        self.width = imgshape[0]
        self.height = imgshape[1]
        self.channels = channels
        self.z_dim = z_dim
        self.n_hid = n_hid

        x_dim = imgshape[0] * imgshape[1] * channels
        l_input = nn.layers.InputLayer(shape=(None,channels,imgshape[0], imgshape[1]),
                input_var=inputvar, name='input')
        if encoder_type=='single_layer':
            l_enc_hid = self.build_encoder_single_hidden(l_input, n_hid)
        elif encoder_type=='conv2d_32_layer':
            l_enc_hid = self.build_encoder_conv2d_32_hidden(l_input)
        elif encoder_type=='conv2d_64_layer':
            l_enc_hid = self.build_encoder_conv2d_64_hidden(l_input)
        elif encoder_type=='conv2d_128_layer':
            l_enc_hid = self.build_encoder_conv2d_128_hidden(l_input)
        else:
            print('unknown encoder type '+encoder_type)
            raise

        l_enc_mu = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
                nonlinearity = None, name='enc_mu')
        l_enc_logsigma = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
                nonlinearity = None, name='enc_logsigma')
        l_dec_mu_list = []
        l_dec_logsigma_list = []
        l_output_list = []
        # tie the weights of all L versions so they are the "same" layer
        W_dec_mu = None
        b_dec_mu = None
        W_dec_ls = None
        b_dec_ls = None
        decoder_params = None
        for i in xrange(L):
            l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
            if decoder_type=='single_layer':
                l_dec_mu, l_dec_logsigma, decoder_params = self.build_decoder_single_hidden(l_Z, n_hid, decoder_params)
            elif decoder_type=='conv2d_32_layer':
                l_dec_mu, l_dec_logsigma, decoder_params = self.build_decoder_conv2d_32_hidden(l_Z, decoder_params)
            elif decoder_type=='conv2d_64_layer':
                l_dec_mu, l_dec_logsigma, decoder_params = self.build_decoder_conv2d_64_hidden(l_Z, decoder_params)
            elif decoder_type=='conv2d_128_layer':
                l_dec_mu, l_dec_logsigma, decoder_params = self.build_decoder_conv2d_128_hidden(l_Z, decoder_params)
            else:
                print('unknown decoder type '+decoder_type)
                raise
            if self.binary:
                l_output = l_dec_mu
                # nn.layers.DenseLayer(l_dec_hid, num_units = x_dim,
                #         nonlinearity = nn.nonlinearities.sigmoid,
                #         W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                #         b = nn.init.Constant(0.) if b_dec_mu is None else b_dec_mu,
                #         name = 'dec_output')
                l_output_list.append(l_output)
                # if W_dec_mu is None:
                #     W_dec_mu = l_output.W
                #     b_dec_mu = l_output.b
            else:
                l_output = GaussianSampleLayer(l_dec_mu, l_dec_logsigma,
                        name='dec_output')
                l_dec_mu_list.append(l_dec_mu)
                l_dec_logsigma_list.append(l_dec_logsigma)
                l_output_list.append(l_output)
        l_output = nn.layers.ElemwiseSumLayer(l_output_list, coeffs=1./L, name='output')
        return l_enc_mu, l_enc_logsigma, l_dec_mu_list, l_dec_logsigma_list, l_output_list, l_output

    # ############################## Main program ################################

    def gen_imgs(self, n_samples):
        z_samples = np.random.rand(n_samples, self.z_dim).astype('float32')
        return self.gen_fn(z_samples).reshape([n_samples, self.channels, self.width, self.height])

    def compile_inference_funcs(self):
        print('Compiling inference functions')
        if self.binary:
            self.generated_x = nn.layers.get_output(self.l_x, {self.l_z_mu:self.z_var}, deterministic=True)
        else:
            self.generated_x = nn.layers.get_output(self.l_x_mu_list[0], {self.l_z_mu:self.z_var},
                    deterministic=True)
        self.gen_fn = theano.function([self.z_var], self.generated_x)
        z_mu = nn.layers.get_output([self.l_z_mu], deterministic=True)
        self.encode_fn = theano.function([self.input_var], z_mu)
        self.test_loss, self.test_prediction = self.build_loss(deterministic=True)
        self.val_fn = theano.function([self.input_var], self.test_loss)
        self.pred_fn = theano.function([self.input_var], self.test_prediction)

    def reconstruct_imgs(self, imgs):
        return self.pred_fn(imgs).reshape(-1, self.channels, self.width, self.height)

    def decode_imgs(self, z):
        return self.gen_fn(z).reshape([n_samples, self.channels, self.width, self.height])

    def get_z(self, imgs, net_order=True):
        batch_size=500
        if imgs.shape[0]<1000:
            if net_order==False:
                imgs = imgs_to_net(imgs)
            return self.encode_fn(imgs)[0]
        z = np.zeros([imgs.shape[0], self.z_dim], dtype='float32')
        for idx in range(0, imgs.shape[0], batch_size):
            count = min(batch_size, imgs.shape[0]-idx)
            if net_order:
                batch = imgs[idx:idx+count]
            else:
                batch = imgs_to_net(imgs[idx:idx+count])
            z[idx:idx+count, :] = self.encode_fn(batch)[0]
        return z

    def build_loss(self, deterministic):
        layer_outputs = nn.layers.get_output([self.l_z_mu, self.l_z_ls] + self.l_x_mu_list + self.l_x_ls_list
                + self.l_x_list + [self.l_x], deterministic=deterministic)
        z_mu =  layer_outputs[0]
        z_ls =  layer_outputs[1]
        x_mu =  [] if self.binary else layer_outputs[2:2+self.L]
        x_ls =  [] if self.binary else layer_outputs[2+self.L:2+2*self.L]
        x_list =  layer_outputs[2:2+self.L] if self.binary else layer_outputs[2+2*self.L:2+3*self.L]
        x = layer_outputs[-1]
        # Loss expression has two parts as specified in [1]
        # kl_div = KL divergence between p_theta(z) and p(z|x)
        # - divergence between prior distr and approx posterior of z given x
        # - or how likely we are to see this z when accounting for Gaussian prior
        # logpxz = log p(x|z)
        # - log-likelihood of x given z
        # - in binary case logpxz = cross-entropy
        # - in continuous case, is log-likelihood of seeing the target x under the
        #   Gaussian distribution parameterized by dec_mu, sigma = exp(dec_logsigma)
        kl_div = 0.5 * T.sum(1 + 2*z_ls - T.sqr(z_mu) - T.exp(2 * z_ls))
        if self.binary:
            logpxz = sum(nn.objectives.binary_crossentropy(x,
                self.input_var.flatten(2)).sum() for x in x_list) * (-1./self.L)
            prediction = x_list[0] if deterministic else x
        else:
            logpxz = sum(log_likelihood(self.input_var.flatten(2), mu, ls) \
                for mu, ls in zip(x_mu, x_ls))/self.L
            prediction = x_mu[0] if deterministic else T.sum(x_mu, axis=0)/self.L
        loss = -1 * (logpxz + kl_div)
        return loss, prediction

    def train_init(self, num_epochs):
        # Create VAE model
        print("Building model and compiling functions...")
        print("L = {}, z_dim = {}, n_hid = {}, binary={}".format(self.L, self.z_dim,
            self.n_hid, self.binary))
        self.x_dim = self.width * self.height * self.channels
        #l_z_mu, l_z_ls, l_x_mu_list, l_x_ls_list, l_x_list, l_x = \
        #       build_vae(input_var, L=L, binary=binary, z_dim=z_dim, n_hid=n_hid, channels=3)


        # If there are dropout layers etc these functions return masked or non-masked expressions
        # depending on if they will be used for training or validation/test err calcs
        self.loss, _ = self.build_loss(deterministic=False)
        self.test_loss, self.test_prediction = self.build_loss(deterministic=True)

        # ADAM updates
        params = nn.layers.get_all_params(self.l_x, trainable=True)
        updates = nn.updates.adam(self.loss, params, learning_rate=1e-4)
        self.train_fn = theano.function([self.input_var], self.loss, updates=updates)
        self.generated_x = nn.layers.get_output(self.l_x_mu_list[0], {self.l_z_mu:self.z_var},
                deterministic=True)
        self.val_fn = theano.function([self.input_var], self.test_loss)
        self.pred_fn = theano.function([self.input_var], self.test_prediction)

        self.train_err = 0
        self.num_epochs = num_epochs
        self.epoch = 0

    def train_epoch(self, epoch_size, batch_size, train_batches, test_batches):
        start_time = time.time()
        n_train_batches = 0
        for _ in range(epoch_size):
            batch = next(train_batches)
            this_err = self.train_fn(batch)
            self.train_err += this_err
            n_train_batches += 1
        val_err = 0
        val_batches = 0
        for batch in test_batches:
            err = self.val_fn(batch)
            val_err += err
            val_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            self.epoch + 1, self.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(self.train_err / n_train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        self.epoch += 1

    def save_net(self, path):
        fn = path+'/vae_params.npz'
        np.savez(fn, *nn.layers.get_all_param_values(self.l_x))

    def load_net(self, path):
        print('Loading network params from %s'%path)
        #decoder
        values = np.load(path+'/vae_params.npz')
        l = []
        for i in range(len(values.files)):
            l.append(values['arr_%d'%i])
        nn.layers.set_all_param_values(self.l_x, l)

        # And finally, we plot some generated data
        # grid = 10
        # n_samples = grid*grid
        # z_samples = np.random.rand(n_samples, z_dim).astype('float32')
        # samples = gen_fn(z_samples)
        # samples = samples.reshape([n_samples, 3, 28,28])
        # samples = (samples*255).astype('uint8')
        # #                        .reshape(6*28, 7*28)
        # snapshot_path = base_path+'/%d'%(epoch+1)
        # if not os.path.exists(snapshot_path):
        #     os.mkdir(snapshot_path)
        # save_samples(samples, grid, snapshot_path+'/gen.jpg')
        # #reconstruction
        # example_batch_size = 50
        # X_comp = imgs_to_net(imgs[0:example_batch_size])
        # X_pred = (pred_fn(X_comp).reshape(-1, channels, width, height)*255).clip(0,255).astype('uint8')
        # save_samples(np.vstack([X_comp, X_pred]), 10, snapshot_path+'/reconstruct.jpg')
        #
        # # save the parameters so they can be loaded for next time
        # fn = snapshot_path+'/vae_params.npz'
        # np.savez(fn, *nn.layers.get_all_param_values(self.l_x))

    # sample from latent space if it's 2d
    # if z_dim == 2:
    #     # functions for generating images given a code (used for visualization)
    #     # for an given code z, we deterministically take x_mu as the generated data
    #     # (no Gaussian noise is used to either encode or decode).
    #     if channels == 1:
    #         im = Image.new('L', (width*19,height*19))
    #     else:
    #         im = Image.new('RGB', (width*19,height*19))
    #     for (x,y),val in np.ndenumerate(np.zeros((19,19))):
    #         z = np.asarray([norm.ppf(0.05*(x+1)), norm.ppf(0.05*(y+1))],
    #                 dtype=theano.config.floatX)
    #         x_gen = gen_fn(z).reshape(-1, channels, width, height)
    #         im.paste(Image.fromarray(get_image_array(x_gen,0)), (x*width,y*height))
    #         im.save('gen.jpg')
