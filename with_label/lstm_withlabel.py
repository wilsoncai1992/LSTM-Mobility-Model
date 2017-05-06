# export MPLBACKEND="agg"
# CUDA_VISIBLE_DEVICES=1 ipython2

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath('../src/models/lstm_mixture_density_model.py')))
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# load code
from lstm_mixture_density_model import tf_lstm_mixture_density_model
# =======================================================================================
# Section
# =======================================================================================
model_name = 'my_lstm_model'

# input_length = 5
input_length = 50
n_lstm_units = 32
# n_lstm_units = 256
n_layers = 1
# pred_x_dim = 4
# obs_x_dim = 5
pred_x_dim = 4
obs_x_dim = 5; use_feature = False # for label only
# obs_x_dim = 4 + 64; use_feature = True # for all features
# n_mixtures = 3
n_mixtures = 7
dropout_prob = 0.1
# y_dim = 8
y_dim = 55
# batch_size=1
batch_size=16
learning_rate = 0.01
start_time_sd = 0.01

# with tf.device('/cpu:0'):
with tf.device('/gpu:0'):
    lstm_DM = tf_lstm_mixture_density_model(name=model_name,
                                            input_length=input_length, # length of h_t
                                            n_lstm_units=n_lstm_units, # no. of hidden units
                                            n_layers=n_layers, # layer no.
                                            pred_x_dim=pred_x_dim, # predict x_t = 4 dim
                                            obs_x_dim=obs_x_dim, # x_t = 5dim
                                            y_dim=y_dim, # y dim
                                            batch_size=batch_size,
                                            n_loc_mixtures=n_mixtures, # no. of gaussian mixture
                                            dropout_prob = dropout_prob,
                                            learning_rate=learning_rate,
                                            start_time_sd=start_time_sd)
# =======================================================================================
# real data
# =======================================================================================
import pandas as pd

# data with label
Label = pd.read_csv('../data_new/Traj_Label_Data2.csv', index_col=[0]).reset_index()
colnames = ['Idx','TrueLabel','PredLabel1','PredLabel2','PredLabel3'] + ['Feature' + str(i) for i in range(1,65)]
Feature = pd.read_csv('../data_cnnout/PredictedLabelFeature.csv', names=colnames)
New_Label = Label.merge(Feature, left_on='index', right_on='Idx', how='left')
FID_with_missing_data = New_Label.loc[New_Label['PredLabel1'].index[New_Label['PredLabel1'].apply(np.isnan)]].FID.unique()
data_raw_withfeature = New_Label[~New_Label.FID.isin(FID_with_missing_data)]
# np.unique(New_Label.FID[New_Label.FID.isin(FID_with_missing_data)])

data_raw = data_raw_withfeature

# data with true label
# Label = pd.read_csv('../data_new/Traj_Label_Data2.csv', index_col=[0]).reset_index()
# colnames = ['Idx','TrueLabel','PredLabel1','PredLabel2','PredLabel3'] + ['Feature' + str(i) for i in range(1,65)]
# Feature = pd.read_csv('../data_cnnout/PredictedLabelFeature.csv', names=colnames)
# data_raw = Label.merge(Feature, left_on='index', right_on='Idx')

# no label
# data_raw = pd.read_csv('../data_new/Traj_Label_Data2.csv')

n_subj = len(np.unique(data_raw['FID']))

location_list = data_raw[['Lon','Lat']].as_matrix()

location_list1 = np.zeros((n_subj, 50, 2))
for it in xrange(0,n_subj):
    # print location_list[(50*it):(50*(it+1)), :].shape
    location_list1[it, :, :] = location_list[(50*it):(50*(it+1)), :]
# location_list1 = data_raw[['Lon','Lat']].as_matrix()



start_time_list1 = np.linspace(start = 1, stop = 50, num=50)
start_time_list1 = start_time_list1[np.newaxis, :, np.newaxis]
start_time_list1 = np.tile(start_time_list1, (n_subj, 1, 1))

# start_time_list1 = np.tile(np.linspace(start = 1, stop = 50, num=50), reps = n_subj)
# start_time_list1 = np.expand_dims(start_time_list1, axis = 1)


duration_list1 = np.repeat(1, repeats = 50)
duration_list1 = duration_list1[np.newaxis,:,np.newaxis]
duration_list1 = np.tile(duration_list1, (n_subj, 1, 1))

# duration_list1 = np.repeat(1, repeats = n_subj * 50)
# duration_list1 = np.expand_dims(duration_list1, axis = 1)

activity_type_list1 = np.eye(50)
activity_type_list1 = activity_type_list1[np.newaxis,:]
activity_type_list1 = np.tile(activity_type_list1, (n_subj, 1, 1))

# activity_type_list1 = np.reshape(np.tile(np.array([0, 1, 0]), 48), [48,3])
# activity_type_list1 = np.vstack(([1,0,0], activity_type_list1, [0,0,1]))
# activity_type_list1 = activity_type_list1[np.newaxis,:,:]
# activity_type_list1 = np.tile(activity_type_list1, (n_subj, 1, 1))

# activity_type_list1 = np.reshape(np.tile(np.array([0, 1, 0]), 48), [48,3])
# activity_type_list1 = np.vstack(([1,0,0], activity_type_list1, [0,0,1]))
# activity_type_list1 = np.tile(activity_type_list1, (n_subj, 1))


end_of_day_list1 = np.zeros((49, 1))
end_of_day_list1 = np.vstack((end_of_day_list1, 1))
end_of_day_list1 = end_of_day_list1[np.newaxis,:,:]
end_of_day_list1 = np.tile(end_of_day_list1, (n_subj, 1, 1))

# end_of_day_list1 = np.zeros((49, 1))
# end_of_day_list1 = np.vstack((end_of_day_list1, 1))
# end_of_day_list1 = np.tile(end_of_day_list1, (n_subj, 1))

activity_information1 = np.dstack((location_list1,
                                      start_time_list1,
                                      duration_list1,
                                      activity_type_list1,
                                      end_of_day_list1))

# activity_information1 = np.hstack((location_list1,
#                                   start_time_list1,
#                                   duration_list1,
#                                   activity_type_list1,
#                                   end_of_day_list1))
# activity_information1 = activity_information1[np.newaxis, :]

# ct: Contextual variables

home_location_list1 = np.array([-95.3,29.98333333])
work_location_list1 = np.array([-70.96666667,42.36666667])

contextual_variables1 = np.hstack((np.array([home_location_list1] * activity_information1.shape[1]),
                                  np.array([work_location_list1] * activity_information1.shape[1])))
contextual_variables1 = contextual_variables1[np.newaxis, :]
contextual_variables1 = np.tile(contextual_variables1, (n_subj, 1, 1))

#
# use true label
# ---------------------------------------------------------------------------------------
label_list = data_raw[['NewLabel']].as_matrix()
label_list = np.reshape(label_list, [n_subj, 50])
label_list = label_list[:, :, np.newaxis]
label_list.shape

contextual_variables1 = np.concatenate((contextual_variables1, label_list), axis = 2)
#
# use CNN feature
# ---------------------------------------------------------------------------------------
# feature_list = data_raw_withfeature.as_matrix()[:,14:]
# feature_list = np.reshape(feature_list, [n_subj, 50, 64])

# contextual_variables1 = np.concatenate((contextual_variables1, feature_list), axis = 2)
#
# use predicted label
# ---------------------------------------------------------------------------------------
# feature_list = data_raw_withfeature.as_matrix()[:,10]
# feature_list = np.reshape(feature_list, [n_subj, 50, 1])

# contextual_variables1 = np.concatenate((contextual_variables1, feature_list), axis = 2)

# Initilization for LSTM model
X_init = np.zeros((1, pred_x_dim))
X_init = np.tile(X_init, (n_subj, 1))
# =======================================================================================
# normalize data
# =======================================================================================
# Center latitude and longitude
lat_mean = np.mean(activity_information1[:, :, 0])
lon_mean = np.mean(activity_information1[:, :, 1])
activity_information1[:, :, 0] -= lat_mean
activity_information1[:, :, 1] -= lon_mean

contextual_variables1[:, :, 0] -= lat_mean
contextual_variables1[:, :, 2] -= lat_mean
contextual_variables1[:, :, 1] -= lon_mean
contextual_variables1[:, :, 3] -= lon_mean


# Normalize latitude and longitude to -1~1
# Normalize starting time and duration to 0~1
lat_max = np.max(np.abs(activity_information1[:, :, 0]))
lon_max = np.max(np.abs(activity_information1[:, :, 1]))

temp = np.array([lat_max,
                lon_max,
                24.,
                24.,
                1.])
temp = np.concatenate((temp , np.ones(50)))
activity_information1 /= temp
# activity_information1 /= np.array([lat_max,
#                                   lon_max,
#                                   24.,
#                                   24.,
#                                   1.,
#                                   1.,
#                                   1.,
#                                   1.])

# no label
# contextual_variables1 /= np.array([lat_max,
#                                   lon_max,
#                                   lat_max,
#                                   lon_max])

# for label only
if not use_feature:
  contextual_variables1 /= np.array([lat_max,
                                    lon_max,
                                    lat_max,
                                    lon_max,
                                    1])

# for all features
if use_feature:
  part1 = [lat_max,
          lon_max,
          lat_max,
          lon_max]
  to_divide = np.concatenate((part1, np.ones(64)))
  contextual_variables1 /= to_divide


# contextual_variables1 /= np.array([lat_max,
#                                   lon_max,
#                                   lat_max,
#                                   lon_max])

# =======================================================================================
# artificial eg.
# =======================================================================================
# We made up 5 activities with location, starting time, duration, and activity types
# The 5 activity types are home -> other -> work -> other -> home


# # xt: Activity information
# location_list = np.array([[37.750460, -122.429491],
#                           [37.944496, -122.351648],
#                           [37.856912, -122.288567],
#                           [37.754701, -122.188187],
#                           [37.750460, -122.429491]])

# start_time_list = np.array([[7.0],
#                             [8.4],
#                             [12.5],
#                             [16.0],
#                             [18.0]])
# start_time_list.shape

# duration_list = np.array([[1.4],
#                           [4.1],
#                           [5.5],
#                           [2.0],
#                           [12.0]])
# duration_list.shape

# activity_type_list = np.array([[1, 0, 0],
#                                [0, 0, 1],
#                                [0, 1, 0],
#                                [0, 0, 1],
#                                [1, 0, 0]])
# activity_type_list.shape

# end_of_day_list = np.array([[0],
#                             [0],
#                             [0],
#                             [0],
#                             [1]])
# end_of_day_list.shape

# activity_information = np.hstack((location_list,
#                                   start_time_list,
#                                   duration_list,
#                                   activity_type_list,
#                                   end_of_day_list))
# activity_information = activity_information[np.newaxis, :]

# # ct: Contextual variables
# home_location_list = np.array([37.750460, -122.429491])
# work_location_list = np.array([37.856912, -122.288567])

# contextual_variables = np.hstack((np.array([home_location_list] * 5),
#                                   np.array([work_location_list] * 5)))
# contextual_variables = contextual_variables[np.newaxis, :]

# # Initilization for LSTM model
# X_init = np.zeros((1, pred_x_dim))
# =======================================================================================
# normalize data
# =======================================================================================
# # Center latitude and longitude
# lat_mean = np.mean(activity_information[:, :, 0])
# lon_mean = np.mean(activity_information[:, :, 1])
# activity_information[:, :, 0] -= lat_mean
# activity_information[:, :, 1] -= lon_mean

# contextual_variables[:, :, 0] -= lat_mean
# contextual_variables[:, :, 2] -= lat_mean
# contextual_variables[:, :, 1] -= lon_mean
# contextual_variables[:, :, 3] -= lon_mean


# # Normalize latitude and longitude to -1~1
# # Normalize starting time and duration to 0~1
# lat_max = np.max(np.abs(activity_information[:, :, 0]))
# lon_max = np.max(np.abs(activity_information[:, :, 1]))

# activity_information /= np.array([lat_max,
#                                   lon_max,
#                                   24.,
#                                   24.,
#                                   1.,
#                                   1.,
#                                   1.,
#                                   1.])

# contextual_variables /= np.array([lat_max,
#                                   lon_max,
#                                   lat_max,
#                                   lon_max])
# =======================================================================================
# train
# =======================================================================================
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

location_sd_bias = 0.0
time_sd_bias = 0.0
pi_bias = 0.0

lstm_DM.train(X_init=X_init,
              X_input_seq=contextual_variables1, # c_t
              y=activity_information1, # x_t
              # epochs=1000,
              epochs=5000,
              sess=sess,
              start_time_list=start_time_list1[:,0,:]/24. ,
              per=1000,
              location_sd_bias=location_sd_bias,
              time_sd_bias=time_sd_bias,
              pi_bias=pi_bias)

# ho =start_time_list1[:,0,:]
# ho=start_time_list1[0,0]
# =======================================================================================
# generate sequence
# =======================================================================================
# config = tf.ConfigProto(allow_soft_placement = True)
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
# Restore variables from disk.
saver.restore(sess, "./model_save2/model.ckpt")
print("Model restored.")


gen_seq, \
gen_coef, \
gen_states, \
gen_mixture_coef = lstm_DM.generate_sequence_coefficients(sess=sess,
                                                          X_init=X_init,
                                                          X_input_seq=contextual_variables1,
                                                          # X_init=X_init[0,:],
                                                          # X_input_seq=contextual_variables1[0,:,:],
                                                          start_time_list=start_time_list1[:,0,:]/24.,
                                                          n=200)
contextual_variables1[0,:,:].shape
# =======================================================================================
# plot sequence
# =======================================================================================
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib as mpl
mpl.use('Agg')
# plt.interactive(True)

# Scale data back
activity_information1[:, :, 0] *= lat_max
activity_information1[:, :, 1] *= lon_max
activity_information1[:, :, 0] += lat_mean
activity_information1[:, :, 1] += lon_mean
activity_information1[:, :, 2] *= 24.
activity_information1[:, :, 3] *= 24.

gen_seq[:, :, 0] *= lat_max
gen_seq[:, :, 1] *= lon_max
gen_seq[:, :, 0] += lat_mean
gen_seq[:, :, 1] += lon_mean
gen_seq[:, :, 2] *= 24
gen_seq[:, :, 3] *= 24

# plot coordinate
plt.figure()

for i in xrange(200):
    plt.plot(gen_seq[i][:,1], gen_seq[i][:,0], 'b.', alpha =0.3)

# red center is truth coordinate
plt.plot(activity_information1[0][:,1], activity_information1[0][:,0], 'ro', lw=3)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('1.png')
#
# plot duration
# ---------------------------------------------------------------------------------------
plt.figure()

for i in xrange(200):
    plt.plot(gen_seq[i][:,2], gen_seq[i][:,3], 'b.', alpha =0.3)

plt.plot(activity_information1[0][:,2], activity_information1[0][:,3], 'ro', lw=3)


plt.xlabel('Starting Time')
plt.ylabel('Duration')
plt.xlim((0, 24))
plt.ylim((0, 24))

plt.savefig('2.png')

#
# plot path for single person
# ---------------------------------------------------------------------------------------
plt.figure()

i = 75
plt.plot(gen_seq[i][:,1], gen_seq[i][:,0], 'b-o', alpha =0.3)

# red center is truth coordinate
plt.plot(activity_information1[i][:,1], activity_information1[i][:,0], 'ro', lw=3)

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig('3.png')

#
# calculate distribution mean
# ---------------------------------------------------------------------------------------
gen_colmean = gen_seq.mean(axis=0)

# plot mean path
plt.figure()

plt.plot(gen_colmean[:,1], gen_colmean[:,0], 'b-o', alpha =0.3)

# red center is truth coordinate
mean_play = np.mean(activity_information1, axis=0)
# plt.plot(activity_information1[0][:,1], activity_information1[0][:,0], 'ro', lw=3)
plt.plot(mean_play[:,1], mean_play[:,0], 'ro', lw=3)

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig('4.png')

#
# calculate distribution median
# ---------------------------------------------------------------------------------------
gen_colmedian = np.median(gen_seq, axis=0)

# plot mean path
plt.figure()

plt.plot(gen_colmedian[:,1], gen_colmedian[:,0], 'b-o', alpha =0.3)

# red center is truth coordinate
median_play = np.median(activity_information1, axis=0)
plt.plot(median_play[:,1], median_play[:,0], 'ro', lw=3)
# plt.plot(activity_information1[10][:,1], activity_information1[10][:,0], 'ro', lw=3)


plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig('5.png')

#
# mean path for single person
# ---------------------------------------------------------------------------------------
contextual_variables2 = contextual_variables1[i,:,:][np.newaxis,:,:]
# contextual_variables2 = np.tile(contextual_variables2, n_subj)
# contextual_variables2.shape

gen_seq, \
gen_coef, \
gen_states, \
gen_mixture_coef = lstm_DM.generate_sequence_coefficients(sess=sess,
                                                          X_init=X_init,
                                                          X_input_seq=,
                                                          # X_init=X_init[0,:],
                                                          # X_input_seq=contextual_variables1[0,:,:],
                                                          start_time_list=start_time_list1[:,0,:]/24.,
                                                          n=200)

