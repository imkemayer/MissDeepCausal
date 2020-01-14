# -*- coding: utf-8 -*-
# # !pip3 install --user --upgrade scikit-learn # We need to update it to run missForest
import tensorflow as tf
import numpy as np
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit
import os


def miwae(X_miss, d=3, d_miwae=3, h_miwae=128, add_mask=False, sig_prior = 1,
          num_samples_zmul=200, l_rate = 0.0001, n_epochs = 602, add_wy = False, w = None, y = None):
  # return xhat, zhat, zhat_mul
  # # ! xhat_rescaled is NOT computed (x is never scaled !)

  np.random.seed(1234)
  tf.set_random_seed(1234)


  n = X_miss.shape[0] # number of observations
  p = X_miss.shape[1] # number of features

  pwy = 0
  if add_wy:
    pwy = 2
    X_miss = np.column_stack([X_miss, w, y])

  print('X_miss.shape',X_miss.shape)
  X_miss = np.copy(X_miss)
  mask = np.isfinite(X_miss) # binary mask that indicates which values are missing

  print('mask.shape', mask.shape)
  
  


  # ##########

  xhat_0 = np.copy(X_miss)
  xhat_0[np.isnan(X_miss)] = 0
  
  p_mod = p
  if add_mask:
    mask_mod = np.copy(mask)
    #mask_mod = mask_mod.astype(float)
    #mask_mod = (mask_mod - np.mean(mask_mod,0))/np.std(mask_mod,0)
    xhat_0 = np.concatenate((xhat_0, mask_mod), axis=1)
    # xfull = np.concatenate((xfull, mask_mod), axis=1)
    mask = np.concatenate((mask, np.ones_like(mask).astype(bool)), axis = 1)
    p = p*2
    pwy = pwy*2
    print('[data mask]', xhat_0.shape)

  # ##########

  x = tf.placeholder(tf.float32, shape=[None, p+pwy]) # Placeholder for xhat_0
  learning_rate = tf.placeholder(tf.float32, shape=[])
  batch_size = tf.shape(x)[0]
  xmask = tf.placeholder(tf.bool, shape=[None, p+pwy])
  K= tf.placeholder(tf.int32, shape=[]) # Placeholder for the number of importance weights

  # ##########

  p_z = tfd.MultivariateNormalDiag(loc=tf.zeros(d_miwae, tf.float32),
                                   scale_diag = sig_prior*tf.ones(d_miwae, tf.float32))

  # ##########

  sigma = "relu"
  
  decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[d_miwae,]),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(3*(p+pwy),kernel_initializer="orthogonal") # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
  ])

  # ##########

  tiledmask = tf.tile(xmask,[K,1])
  tiledmask_float = tf.cast(tiledmask,tf.float32)
  mask_not_float = tf.abs(-tf.cast(xmask,tf.float32))

  iota = tf.Variable(np.zeros([1,p+pwy]),dtype=tf.float32)
  tilediota = tf.tile(iota,[batch_size,1])
  iotax = x + tf.multiply(tilediota,mask_not_float)

  # ##########

  encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[p+pwy,]),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(3*d_miwae,kernel_initializer="orthogonal")
  ])

  # ##########

  out_encoder = encoder(iotax)
  q_zgivenxobs = tfd.Independent(distribution=tfd.StudentT(loc=out_encoder[..., :d_miwae], scale=tf.nn.softplus(out_encoder[..., d_miwae:(2*d_miwae)]), df=3 + tf.nn.softplus(out_encoder[..., (2*d_miwae):(3*d_miwae)])))
  zgivenx = q_zgivenxobs.sample(K)
  zgivenx_flat = tf.reshape(zgivenx,[K*batch_size,d_miwae])
  data_flat = tf.reshape(tf.tile(x,[K,1]),[-1,1])

  # ##########

  out_decoder = decoder(zgivenx_flat)
  all_means_obs_model = out_decoder[..., :p+pwy]
  all_scales_obs_model = tf.nn.softplus(out_decoder[..., (p+pwy):(2*(p+pwy))]) + 0.001
  all_degfreedom_obs_model = tf.nn.softplus(out_decoder[..., (2*(p+pwy)):(3*(p+pwy))]) + 3
  all_log_pxgivenz_flat = tfd.StudentT(loc=tf.reshape(all_means_obs_model,[-1,1]),scale=tf.reshape(all_scales_obs_model,[-1,1]),df=tf.reshape(all_degfreedom_obs_model,[-1,1])).log_prob(data_flat)
  all_log_pxgivenz = tf.reshape(all_log_pxgivenz_flat,[K*batch_size,p+pwy])

  # ##########

  logpxobsgivenz = tf.reshape(tf.reduce_sum(tf.multiply(all_log_pxgivenz[:,0:p_mod],tiledmask_float[:,0:p_mod]),1),[K,batch_size])
  logpz = p_z.log_prob(zgivenx)
  logq = q_zgivenxobs.log_prob(zgivenx)

  # ##########

  miwae_loss = -tf.reduce_mean(tf.reduce_logsumexp(logpxobsgivenz + logpz - logq,0)) +tf.log(tf.cast(K,tf.float32))
  train_miss = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(miwae_loss)

  # ##########

  xgivenz = tfd.Independent(
        distribution=tfd.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model))

  # ##########

  imp_weights = tf.nn.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
  xms = tf.reshape(xgivenz.mean(),[K,batch_size,p+pwy])
  xm=tf.einsum('ki,kij->ij', imp_weights, xms) 

  # ##########

  z_hat = tf.einsum('ki,kij->ij', imp_weights, zgivenx) 

  # ##########

  sir_logits = tf.transpose(logpxobsgivenz + logpz - logq)
#   sirx = tfd.Categorical(logits = sir_logits).sample(num_samples_xmul)
  xmul = tf.reshape(xgivenz.sample(),[K,batch_size,p+pwy])

  sirz = tfd.Categorical(logits = sir_logits).sample(num_samples_zmul)
  zmul = tf.reshape(zgivenx,[K,batch_size,d_miwae])
  
  # ##########

  miwae_loss_train=np.array([])
#   mse_train=np.array([])
  bs = 64 # batch size
  xhat = np.copy(xhat_0) # This will be out imputed data matrix
#   x_mul_imp = np.tile(xhat_0,[num_samples_xmul,1,1])
  zhat = np.zeros([n,d_miwae]) # low-dimensional representations

  zhat_mul = np.tile(zhat, [num_samples_zmul,1,1])
      
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for ep in range(1,n_epochs):
        perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
        # print('debug n/bs = ', n/bs, int(n/bs))
        batches_data = np.array_split(xhat_0[perm,], int(n/bs))
        batches_mask = np.array_split(mask[perm,], int(n/bs))
        for it in range(len(batches_data)):
            train_miss.run(feed_dict={x: batches_data[it], learning_rate: l_rate, K:20, xmask: batches_mask[it]}) # Gradient step      
        if ep == n_epochs - 1:
            losstrain = np.array([miwae_loss.eval(feed_dict={x: xhat_0, K:20, xmask: mask})]) # MIWAE bound evaluation
            miwae_loss_train = np.append(miwae_loss_train,-losstrain,axis=0)
            print('Epoch %g' %ep)
            print('MIWAE likelihood bound  %g' %-losstrain)
            for i in range(n): # We impute the observations one at a time for memory reasons
                # # Single imputation:
                xhat[i,:][~mask[i,:]]=xm.eval(feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})[~mask[i,:].reshape([1,p+pwy])]
                # # Multiple imputation:
                # si, xmu = sess.run([sirx, xmul],feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})
                # x_mul_imp[:,i,:][~np.tile(mask[i,:].reshape([1,p+pwy]),[num_samples_xmul,1])] = np.squeeze(xmu[si,:,:])[~np.tile(mask[i,:].reshape([1,p+pwy]),[num_samples_xmul,1])]
                # Dimension reduction:
                zhat[i,:] = z_hat.eval(feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})
                # Z|X* sampling:
                si, zmu = sess.run([sirz, zmul],feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})    
                zhat_mul[:,i,:] = np.squeeze(zmu[si,:,:]).reshape((num_samples_zmul, d_miwae))
            # err = np.array([mse(xhat,xfull,mask)])
            # mse_train = np.append(mse_train,err,axis=0)
            # print('Imputation MSE  %g' %err)
  
  print('----- miwae training done -----')
  
#   if add_mask:
#     xhat_rescaled = xhat[:,0:int(p/2)]*np.std(data,0) + np.mean(data,0)
#   else:
#     xhat_rescaled = xhat*np.std(data,0) + np.mean(data,0)
  
  return xhat, zhat, zhat_mul


def miwae_cv(X_miss, d=3, d_miwae_list=[10,100], h_miwae=128, add_mask=False, sig_prior_list = [0.1, 1],
             num_samples_zmul=200, learning_rate = 0.0001, n_epochs = 602, add_wy = False, w = None, y = None,
             k_fold = 5):
  # return xhat, zhat, zhat_mul
  # # ! xhat_rescaled is NOT computed (x is never scaled !)

  np.random.seed(1234)
  tf.set_random_seed(1234)


  n = X_miss.shape[0] # number of observations
  p = X_miss.shape[1] # number of features

  pwy = 0
  if add_wy:
    pwy = 2
    X_miss = np.column_stack([X_miss, w, y])

  print('X_miss.shape',X_miss.shape)
  X_miss_all = np.copy(X_miss)
  mask_all = np.isfinite(X_miss) # binary mask that indicates which values are missing

  print('mask.shape', mask_all.shape)

  rs = ShuffleSplit(n_splits=k_fold, test_size=.25, random_state=1234)
  
  k = 0
  elbo = []
  for train_index, test_index in rs.split(X_miss_all):
    k+=1
    print('Fold n ' + str(k) + '\n')
    X_miss = np.copy(X_miss_all[train_index,])
    mask = np.copy(mask_all[train_index,])
    n = X_miss.shape[0] # number of observations

    X_miss_test = np.copy(X_miss_all[test_index,])
    mask_test = np.copy(mask_all[test_index,])

    xhat_0 = np.copy(X_miss)
    xhat_0[np.isnan(X_miss)] = 0

    xhat_0_test = np.copy(X_miss_test)
    xhat_0_test[np.isnan(X_miss_test)] = 0
    
    p_mod = p
    if add_mask:
      mask_mod = np.copy(mask)
      xhat_0 = np.concatenate((xhat_0, mask_mod), axis=1)
      mask = np.concatenate((mask, np.ones_like(mask).astype(bool)), axis = 1)

      p = p*2
      pwy = pwy*2
      print('[data mask]', xhat_0.shape)

      # same on test set
      mask_mod_test = np.copy(mask_test)
      xhat_0_test = np.concatenate((xhat_0_test, mask_mod_test), axis=1)
      mask_test = np.concatenate((mask_test, np.ones_like(mask_test).astype(bool)), axis = 1)
      

    

  # ##########
    for sig_index in range(len(sig_prior_list)):
      sig_prior = sig_prior_list[sig_index]

      for d_index in range(len(d_miwae_list)):
        d_miwae = d_miwae_list[d_index]


        # ##########

        x = tf.placeholder(tf.float32, shape=[None, p+pwy]) # Placeholder for xhat_0
        learning_rate = tf.placeholder(tf.float32, shape=[])
        batch_size = tf.shape(x)[0]
        xmask = tf.placeholder(tf.bool, shape=[None, p+pwy])
        K= tf.placeholder(tf.int32, shape=[]) # Placeholder for the number of importance weights

        # ##########

        p_z = tfd.MultivariateNormalDiag(loc=tf.zeros(d_miwae, tf.float32),
                                         scale_diag = sig_prior*tf.ones(d_miwae, tf.float32))

        # ##########

        sigma = "relu"
        
        decoder = tfk.Sequential([
          tfkl.InputLayer(input_shape=[d_miwae,]),
          tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
          tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
          tfkl.Dense(3*(p+pwy),kernel_initializer="orthogonal") # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        ])

        # ##########

        tiledmask = tf.tile(xmask,[K,1])
        tiledmask_float = tf.cast(tiledmask,tf.float32)
        mask_not_float = tf.abs(-tf.cast(xmask,tf.float32))

        iota = tf.Variable(np.zeros([1,p+pwy]),dtype=tf.float32)
        tilediota = tf.tile(iota,[batch_size,1])
        iotax = x + tf.multiply(tilediota,mask_not_float)

        # ##########

        encoder = tfk.Sequential([
          tfkl.InputLayer(input_shape=[p+pwy,]),
          tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
          tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
          tfkl.Dense(3*d_miwae,kernel_initializer="orthogonal")
        ])

        # ##########

        out_encoder = encoder(iotax)
        q_zgivenxobs = tfd.Independent(distribution=tfd.StudentT(loc=out_encoder[..., :d_miwae], scale=tf.nn.softplus(out_encoder[..., d_miwae:(2*d_miwae)]), df=3 + tf.nn.softplus(out_encoder[..., (2*d_miwae):(3*d_miwae)])))
        zgivenx = q_zgivenxobs.sample(K)
        zgivenx_flat = tf.reshape(zgivenx,[K*batch_size,d_miwae])
        data_flat = tf.reshape(tf.tile(x,[K,1]),[-1,1])

        # ##########

        out_decoder = decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :p+pwy]
        all_scales_obs_model = tf.nn.softplus(out_decoder[..., (p+pwy):(2*(p+pwy))]) + 0.001
        all_degfreedom_obs_model = tf.nn.softplus(out_decoder[..., (2*(p+pwy)):(3*(p+pwy))]) + 3
        all_log_pxgivenz_flat = tfd.StudentT(loc=tf.reshape(all_means_obs_model,[-1,1]),scale=tf.reshape(all_scales_obs_model,[-1,1]),df=tf.reshape(all_degfreedom_obs_model,[-1,1])).log_prob(data_flat)
        all_log_pxgivenz = tf.reshape(all_log_pxgivenz_flat,[K*batch_size,p+pwy])

        # ##########

        logpxobsgivenz = tf.reshape(tf.reduce_sum(tf.multiply(all_log_pxgivenz[:,0:p_mod],tiledmask_float[:,0:p_mod]),1),[K,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        # ##########

        miwae_loss = -tf.reduce_mean(tf.reduce_logsumexp(logpxobsgivenz + logpz - logq,0)) +tf.log(tf.cast(K,tf.float32))
        train_miss = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(miwae_loss)

        # ##########

        xgivenz = tfd.Independent(
              distribution=tfd.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model))

        # ##########

        imp_weights = tf.nn.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
        xms = tf.reshape(xgivenz.mean(),[K,batch_size,p+pwy])
        xm=tf.einsum('ki,kij->ij', imp_weights, xms) 

        # ##########

        z_hat = tf.einsum('ki,kij->ij', imp_weights, zgivenx) 

        # ##########

        sir_logits = tf.transpose(logpxobsgivenz + logpz - logq)
      #   sirx = tfd.Categorical(logits = sir_logits).sample(num_samples_xmul)
        xmul = tf.reshape(xgivenz.sample(),[K,batch_size,p+pwy])

        sirz = tfd.Categorical(logits = sir_logits).sample(num_samples_zmul)
        zmul = tf.reshape(zgivenx,[K,batch_size,d_miwae])
        
        # ##########

        miwae_loss_train=np.array([])
      #   mse_train=np.array([])
        bs = 64 # batch size
        xhat = np.copy(xhat_0) # This will be out imputed data matrix
      
            
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ep in range(1,n_epochs):
              perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
              # print('debug n/bs = ', n/bs, int(n/bs))
              batches_data = np.array_split(xhat_0[perm,], int(n/bs))
              batches_mask = np.array_split(mask[perm,], int(n/bs))
              for it in range(len(batches_data)):
                  train_miss.run(feed_dict={x: batches_data[it], learning_rate: learning_rate, K:20, xmask: batches_mask[it]}) # Gradient step      
              if ep == n_epochs - 1:
                  losstrain = np.array([miwae_loss.eval(feed_dict={x: xhat_0_test, K:20, xmask: mask_test})]) # MIWAE bound evaluation
                  miwae_loss_train = np.append(miwae_loss_train,-losstrain,axis=0)
                  elbo.append([d_miwae, sig_prior, k, -float(losstrain)])
                  print('Epoch %g' %ep)
                  print('MIWAE likelihood bound  %g' %-losstrain)
                  # for i in range(n): # We impute the observations one at a time for memory reasons
                  #     # # Single imputation:
                  #     xhat[i,:][~mask[i,:]]=xm.eval(feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})[~mask[i,:].reshape([1,p+pwy])]
                  #     # # Multiple imputation:
                  #     # si, xmu = sess.run([sirx, xmul],feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})
                  #     # x_mul_imp[:,i,:][~np.tile(mask[i,:].reshape([1,p+pwy]),[num_samples_xmul,1])] = np.squeeze(xmu[si,:,:])[~np.tile(mask[i,:].reshape([1,p+pwy]),[num_samples_xmul,1])]
                  #     # Dimension reduction:
                  #     zhat[i,:] = z_hat.eval(feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})
                  #     # Z|X* sampling:
                  #     si, zmu = sess.run([sirz, zmul],feed_dict={x: xhat_0[i,:].reshape([1,p+pwy]), K:10000, xmask: mask[i,:].reshape([1,p+pwy])})    
                  #     zhat_mul[:,i,:] = np.squeeze(zmu[si,:,:])
                  # # err = np.array([mse(xhat,xfull,mask)])
                  # # mse_train = np.append(mse_train,err,axis=0)
                  # # print('Imputation MSE  %g' %err)
                  
  print('----- miwae training done -----')
  
  
  return elbo

