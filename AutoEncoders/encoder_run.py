from vae import VAE
from ali_wgan import ALI
from autoencoder import imgs_to_net
import h5py
import numpy as np

#vae = VAE(L=1, binary=False, imgshape=(28,28), channels=3, z_dim=100, n_hid=1024,
#        encoder_type='single_layer', decoder_type='single_layer')
hdf5_path='/home/ubuntu/Data/iGAN_bags/handbag_28_c.hdf5'
ali = ALI(channels=3, z_dim=100, encoder_model='encoder_28',
    decoder_model='decoder_28', critic_model='critic_28')
ali.load_net('snapshots_ali8_28')
# ali.compile_enc_dec()
# ali.compile_critic()
# #
f = h5py.File(hdf5_path,'r')
imgs = f['imgs']
#
# imgs_net = imgs_to_net(imgs[:100])
# z = ali.encode_fn(imgs_net)
# z_rand = np.random.randn(100, 100).astype('float32')
# imgs_dec = ali.decode_fn(z_rand)
# np.mean(ali.critic_det_fn(z_rand, imgs_dec))
# np.mean(ali.critic_det_fn(z, imgs_net))
ali.train(1000, hdf5_path, 'snapshots_ali9_28', num_epochs=5000,
     train_encoder=True, train_decoder=True, epochsize=300, initial_eta=5e-5,
     initial_critic_steps=5000)
# vae = VAE(L=1, binary=False, imgshape=(28,28), channels=3, z_dim=100, n_hid=1024,
#         encoder_type='conv2d_32_layer', decoder_type='conv2d_32_layer')
# vae = VAE(L=1, binary=False, imgshape=(64,64), channels=3, z_dim=256, n_hid=1024,
    #   encoder_type='conv2d_64_layer', decoder_type='conv2d_64_layer')
# vae = VAE(L=1, binary=False, imgshape=(64,64), channels=3, z_dim=256, n_hid=1024,
#        encoder_type='conv2d_64_layer', decoder_type='conv2d_64_no_dense_layer')
vae = VAE(L=1, binary=False, imgshape=(128,128), channels=3, z_dim=2048, n_hid=2048,
        encoder_type='conv2d_128_layer', decoder_type='conv2d_128_layer')
vae.load_net('vae_128_z_2048')
hdf5_path='/home/ubuntu/Data/iGAN_bags/handbag_128.hdf5'
f = h5py.File(hdf5_path,'r')
imgs = f['imgs']
vae.save_generated_data('reconstructed_128.jpg',imgs)
#vae = VAE(L=1, binary=False, imgshape=(28,28), channels=3, z_dim=100, n_hid=1024)
#vae.train(1000, hdf5_path, 'snapshots_vae19_128_2048', num_epochs=200)
# hdf5_path='/home/ubuntu/Data/iGAN_bags/handbag_64.hdf5'
# vae.train(1000, hdf5_path, 'snapshots_vae18_64_no_dense_norm', num_epochs=200)
# hdf5_path='/home/ubuntu/Data/iGAN_bags/handbag_28_c.hdf5'
# vae.train(1000, hdf5_path, 'snapshots_vae14_28_conv', num_epochs=200)
# vae.encode_fn(imgs_to_net(imgs[0:100]))
