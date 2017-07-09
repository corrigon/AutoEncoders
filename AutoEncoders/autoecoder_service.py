#!/usr/bin/python
import urllib3
urllib3.disable_warnings()
import copy
import json
import logging
import os
import sys
import time
import traceback
from StringIO import StringIO
from logging.handlers import RotatingFileHandler
from threading import Lock
import commands
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, Response
from scipy.misc import imresize, imsave, imread
from werkzeug.contrib.fixers import ProxyFix
from skimage.color import rgb2grey
import glob
import argparse
import time
from vae import VAE
from scipy.misc import imread, imresize

'''
Web service for calculating an image's feature vector from a encoder
'''
app = Flask(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--architecture', type=str,
    help='Architecture (vae_128_z_2048)', default='vae_128_z_2048')
parser.add_argument('--port', type=int,
    help='Port of the service', default='8080')
if not os.path.isdir('log'):
    os.makedirs('log')

args = parser.parse_args(sys.argv[1:])
if args.architecture=='vae_128_z_2048':
    encoder = VAE(L=1, binary=False, imgshape=(128,128), channels=3, z_dim=2048, n_hid=2048,
            encoder_type='conv2d_128_layer', decoder_type='conv2d_128_layer')
    encoder.load_net('vae_128_z_2048')
    #vae.load_net('/home/ubuntu/AutoEncoders/AutoEncoaders/snapshots_vae8_128_conv')
    NET_IMAGE_SIZE = 128
    z_dim = 2048
else:
    print('invalid architecture '+args.architecture)
    sys.exit(-1)
logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('log/autoencoder_fv_calc_@@num@@.log', maxBytes=2000000, backupCount=10)
formatter = logging.Formatter(
    "%(asctime)s [mobile_app:%(processName)s:%(process)d] [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

lock = Lock()

def calc_feat_vec(image_path):
    lock.acquire()
    start_ts = time.time()
    try:
        img = imread(image_path, mode='RGB')
        if img.ndim == 2:
            img = np.repeat(img[:,:,None],3,2)
        if img.shape[2] == 4:
            img = img[:,:,:3]
        img = imresize(img,(NET_IMAGE_SIZE,NET_IMAGE_SIZE))[None,...]
        feat = encoder.get_z(img, net_order=False)
        return np.asarray(feat.reshape(z_dim,), dtype='float32')
    except:
        return np.asarray([])
    finally:
        print 'Calc took %f'%(time.time()-start_ts)
        lock.release()

def download_image(img_url, crop=False, xyhw=True, coords=None):
    img_path = 'dl_' + repr(time.time()) + '.jpg'
    try:
        response = requests.get(img_url, timeout=2)
        if response.status_code == requests.codes.ok:
            if type(response.content) == str and response.content.startswith('<!'):
                raise ValueError("URL doesn't seem to lead to an image: " + response.content)
            dl_img = Image.open(StringIO(response.content))
            if crop:
                if xyhw:
                    dl_img = dl_img.crop(
                        (coords['x1'], coords['y1'], coords['x1'] + coords['w'], coords['y1'] + coords['h']))
                else:
                    dl_img = dl_img.crop((coords['x1'], coords['y1'], coords['x2'], coords['y2']))
            dl_img = np.array(dl_img)
            # dl_img = caffe.io.oversample([dl_img,],(227,227)).astype('float32')/255.
	    if not len(dl_img.shape) == 3:
                raise ValueError("Illegal Image Format")
            resize_ratio = max(dl_img.shape) / float(NET_IMAGE_SIZE)
            if resize_ratio > 1.0:
                dl_img = imresize(dl_img, (int(dl_img.shape[0] / resize_ratio), int(dl_img.shape[1] / resize_ratio)))

            # Convert to grey as RGB
            # im_g=rgb2grey(dl_img)
            # im_rgb = np.repeat(im_g[:,:,None],3,axis=2)
            # im_rgb = (im_rgb*255).astype('uint8')
            # imsave(img_path, im_rgb)


            imsave(img_path, dl_img)

            return img_path
        else:
            response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise NameError(img_path + ';;;' + e.message)
    except ValueError as e:
        raise NameError(img_path + ';;;' + e.message)
    except:
        raise NameError(img_path + ';;;' + traceback.format_exc())


@app.route('/fv')
def feature_vector_calculator():
    img_path = None
    try:
        print 'fv request url '+request.args.get('url')
        logger.info("Calculating FV for " + request.args.get('url'))
        img_path = download_image(request.args.get('url'))
        feats = calc_feat_vec(img_path)
    	resp_pred = {}
        #return Response(json.dumps({'status': 'ok'}),
        #                mimetype='application/json')
        return Response(json.dumps({'status': 'ok', 'fv': feats.tolist()}),
                        mimetype='application/json')
    except NameError as e:
        return Response(json.dumps({'status': 'error', 'msg': e}),
                        mimetype='application/json')
    except:
        print traceback.format_exc()
        return Response(json.dumps({'status': 'error', 'msg': traceback.format_exc()}),
                        mimetype='application/json')
    finally:
	if img_path:
	    commands.getoutput('rm -rf ' + img_path)

app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run('0.0.0.0', port=args.port)
