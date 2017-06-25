from vae import VAE
from autoencoder import imgs_to_net
import h5py

#vae = VAE(L=1, binary=False, imgshape=(28,28), channels=3, z_dim=100, n_hid=1024,
        #encoder_type='conv2d_32_layer', decoder_type='conv2d_32_layer')
#vae = VAE(L=1, binary=False, imgshape=(64,64), channels=3, z_dim=256, n_hid=1024,
#        encoder_type='conv2d_64_layer', decoder_type='conv2d_64_layer')
vae = VAE(L=1, binary=False, imgshape=(128,128), channels=3, z_dim=256, n_hid=1024,
        encoder_type='conv2d_128_layer', decoder_type='conv2d_128_layer')

#vae = VAE(L=1, binary=False, imgshape=(28,28), channels=3, z_dim=100, n_hid=1024)
hdf5_path='/home/ubuntu/Data/iGAN_bags/handbag_128.hdf5'
vae.train(1000, hdf5_path, 'snapshots_vae7_128_conv', num_epochs=100)
#hdf5_path='/home/ubuntu/Data/iGAN_bags/handbag_64.hdf5'
#vae.train(1000, hdf5_path, 'snapshots_vae6_64_conv', num_epochs=100)
f = h5py.File(hdf5_path,'r')
imgs = f['imgs']
vae.encode_fn(imgs_to_net(imgs[0:100]))
