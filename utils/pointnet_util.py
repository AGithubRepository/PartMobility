""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_trajectory'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping_trajectory'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/interpolation_trajectory'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/get_matrix'))
from sampling_trajectory import farthest_point_sample_trajectory, gather_trajectory
from grouping_trajectory import query_ball_trajectory, group_trajectory, knn_trajectory
from interpolation_trajectory import three_nn, three_interpolate
from get_matrix import get_motion_matrix,mul_motion_matrix
import tensorflow as tf
import numpy as np
import tf_util
from abspose import cal_mat


def sample_and_group(trajectory, trajectory_feature, msample, radius, kgroup, knn=False, use_xyz=True,centralization=True,use_fps=True):
    '''
    Input:
        trajectory: (batch_size, npoint, num_frame, 3)
        trajectory_feature: (batch_size, npoint, num_frame, channel)
        msample: int32
        radius: float32
        kgroup: int32
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        sampled_trajectory: (batch_size, msample, num_frame, 3) TF tensor
        new_trajectory_feature: (batch_size, msample, kgroup, num_frame, 3+channel) TF tensor
        idx: (batch_size, msample, kgroup) TF tensor, indices of grouped trajectory idx for each sampled trajectory
        grouped_trajectory: (batch_size, msample, kgroup, num_frame, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ in frame 1) in local regions
    '''
    batch_size = trajectory.get_shape()[0].value
    npoint = trajectory.get_shape()[1].value
    num_frame = trajectory.get_shape()[2].value
    if use_fps:
        sampled_trajectory = gather_trajectory(trajectory, farthest_point_sample_trajectory(trajectory, msample)) # (batch_size, msample, num_frame, 3)
    else:
        sampled_idx = tf.random_uniform(shape=(batch_size,msample),minval=0,maxval=npoint,dtype=tf.int32)
        sampled_trajectory = gather_trajectory(trajectory, sampled_idx) # (batch_size, msample, num_frame, 3)
    if knn:
        idx, _ = knn_trajectory(trajectory, sampled_trajectory, kgroup)
    else:
        idx, pts_cnt = query_ball_trajectory(trajectory, sampled_trajectory, radius, kgroup)
    grouped_trajectory = group_trajectory(trajectory, idx) # (batch_size, msample, kgroup, num_frame, 3)
    if centralization:
        grouped_trajectory -= tf.tile(tf.expand_dims(tf.expand_dims(sampled_trajectory[:,:,int(num_frame/2),:], 2),3), [1,1,kgroup,num_frame,1]) # translation normalization
    if trajectory_feature is not None:
        grouped_trajectory_feature = group_trajectory(trajectory_feature, idx) # (batch_size, msample, kgroup, num_frame, channel)
        if use_xyz:
            new_trajectory_feature = tf.concat([grouped_trajectory, grouped_trajectory_feature], axis=-1) # (batch_size, msample, kgroup, num_frame, 3+channel)
        else:
            new_trajectory_feature = grouped_trajectory_feature
    else:
        new_trajectory_feature = grouped_trajectory

    return sampled_trajectory, new_trajectory_feature, idx, grouped_trajectory


def sample_and_group_all(trajectory, trajectory_feature, use_xyz=True):
    '''
    Inputs:
        trajectory: (batch_size, msample, num_frame, 3)
        trajectory_feature: (batch_size, msample, num_frame, channel)
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        sampled_trajectory: (batch_size, 1, num_frame, 3) as (0,0,0)
        new_trajectory_feature: (batch_size, kmsample=1, group=npoint, num_frame, 3 + channel) TF tensor
    Note:
        Equivalent to sample_and_group with msample=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = trajectory.get_shape()[0].value
    npoint = trajectory.get_shape()[1].value
    num_frame = trajectory.get_shape()[2].value
    sampled_trajectory = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,1,3)), (batch_size,1,num_frame,1)),dtype=tf.float32) # (batch_size, 1, num_frame, 3)
    idx = tf.constant(np.tile(np.array(range(npoint)).reshape((1,1,npoint)), (batch_size,1,1)))
    grouped_trajectory = tf.expand_dims(trajectory, axis=1) # (batch_size, msample=1, kgroup=npoint, num_frame, 3)
    if trajectory_feature is not None:
        if use_xyz:
            new_trajectory_feature = tf.concat([trajectory, trajectory_feature], axis=-1) # (batch_size, kgroup=npoint, num_frame, 3 + channel)
        else:
            new_trajectory_feature = trajectory_feature
        new_trajectory_feature = tf.expand_dims(new_trajectory_feature, 1) # (batch_size, msample=1, kgroup=npoint, num_frame, ?)
    else:
        new_trajectory_feature = grouped_trajectory
    return sampled_trajectory, new_trajectory_feature, idx, grouped_trajectory



def pointnet_sa_module(trajectory, trajectory_feature, msample, radius, kgroup, mlp, mlp2, group_all, is_training, bn_decay, scope,num_layers = 2, hidden_size = 128, bn=True, pooling='max', knn=False, use_xyz=True,centralization = True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            trajectroy: (batch_size, npoint, num_frame, 3) TF tensor
            trajectory_feature: (batch_size, npoint, num_frame, channel) TF tensor
            msample: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            kgroup: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            num_layers: number of lstm layers
            hidden_size: hidden size of lstm layers
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            sampled_trajectory: (batch_size, msample, num_frame, 3) TF tensor
            new_trajectory_feature: (batch_size, msample, num_frame, ?) TF tensor
            idx: (batch_size, msample, kgroup) int32 -- indices for local regions
    '''

    batch_size = trajectory.get_shape()[0].value
    npoint = trajectory.get_shape()[1].value
    num_frame = trajectory.get_shape()[2].value
    channel = trajectory_feature.get_shape()[3].value
    with tf.variable_scope(scope) as sc:
        # RNN Feature Embedding(time_feature)

        encoder_input = tf.reshape(trajectory_feature ,[-1,num_frame,channel])
        with tf.variable_scope("encoder",initializer=tf.orthogonal_initializer()):
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size), input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(num_layers)])
            enc_output,enc_state = tf.nn.dynamic_rnn(encoder_cell,encoder_input,dtype=tf.float32) #enc_output (-1, num_frame, 128)
        enc_output = tf.reshape(enc_output ,[batch_size, npoint, num_frame,hidden_size])

        # Sample and Grouping
        if group_all:
            kgroup = trajectory.get_shape()[1].value
            sampled_trajectory, new_trajectory_feature, idx, grouped_trajectory = sample_and_group_all(trajectory, enc_output, use_xyz)
        else:
            sampled_trajectory, new_trajectory_feature, idx, grouped_trajectory = sample_and_group(trajectory, enc_output, msample, radius, kgroup, knn, use_xyz,centralization = True) #(batch_size, msample, kgroup, num_frame, channel)

        # CNN trajectory Feature Embedding(spatial_feature)
        if pooling=='max':
            new_trajectory_feature = tf.reduce_max(new_trajectory_feature, axis=[2], name='maxpool')
        elif pooling=='avg':
            new_trajectory_feature = tf.reduce_mean(new_trajectory_feature, axis=[2], name='avgpool')
        print(new_trajectory_feature)
        for i, num_out_channel in enumerate(mlp):
            rnn_channel = new_trajectory_feature.get_shape()[3].value
            new_trajectory_feature = tf.reshape(new_trajectory_feature ,[-1,num_frame,rnn_channel])
            with tf.variable_scope("decoder1_%d"%(i),initializer=tf.orthogonal_initializer()):
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_out_channel), input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(num_layers)])
                dec_output,dec_state = tf.nn.dynamic_rnn(decoder_cell,new_trajectory_feature,dtype=tf.float32) #dec_output (-1, num_frame, 128)
            new_trajectory_feature = tf.reshape(dec_output ,[batch_size, msample, num_frame,num_out_channel])

        # Pooling in Local Regions


        # [Optional] CNN region Feature Embedding
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                rnn_channel = new_trajectory_feature.get_shape()[3].value
                new_trajectory_feature = tf.reshape(new_trajectory_feature ,[-1,num_frame,rnn_channel])
                with tf.variable_scope("decoder2_%d"%(i),initializer=tf.orthogonal_initializer()):
                    decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_out_channel), input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(num_layers)])
                    dec_output,dec_state = tf.nn.dynamic_rnn(decoder_cell,new_trajectory_feature,dtype=tf.float32) #dec_output (-1, num_frame, 128)
                new_trajectory_feature = tf.reshape(dec_output ,[batch_size, msample, num_frame,num_out_channel])
        return sampled_trajectory, new_trajectory_feature, idx



def pointnet_fp_module(trajectory1, trajectory2, trajectory_feature1, trajectory_feature2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            trajectory1: (batch_size, npoint, num_frame, 3) TF tensor                                                              
            trajectory2: (batch_size, msample, num_frame, 3) TF tensor, sparser than xyz1                                           
            trajectory_feature1: (batch_size, npoint, num_frame, channel1) TF tensor                                                   
            trajectory_feature2: (batch_size, npoint, num_frame, channel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, npoint, num_frame, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(trajectory1, trajectory2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / (norm + 1e-6)
        interpolated_trajectory_feature = three_interpolate(trajectory_feature2, idx, weight)

        if trajectory_feature1 is not None:
            new_trajectory_feature = tf.concat(axis=-1, values=[interpolated_trajectory_feature, trajectory_feature1]) # (batch_size, npoints, num_frame, channel1 + channel2)
        else:
            new_trajectory_feature = interpolated_trajectory_feature
        for i, num_out_channel in enumerate(mlp):
            new_trajectory_feature = tf_util.conv2d(new_trajectory_feature, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         weight_decay=0.0005,bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay) #(batch_size, npoints, num_frame, mlp[-1])
        return new_trajectory_feature

def pointnet_rebuild_module(trajectory, trajectory_feature, msample, radius, kgroup, is_training, bn_decay, scope, num_layers = 2, hidden_size = 128, bn=True, pooling='avg', knn=True, use_xyz=True,centralization = False,use_fps=False):
    ''' PointNet rebuild Module
        Input:
            trajectroy: (batch_size, npoint, num_frame, 3) TF tensor
            trajectory_feature: (batch_size, npoint, num_frame, channel) TF tensor
            msample: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            kgroup: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            num_layers: number of lstm layers
            hidden_size: hidden size of lstm layers
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            rebuild_frame: (batch_size, msample, kgroup, 3)
            last_frame: (batch_size, msample, kgroup, 3)
    '''
    batch_size = trajectory.get_shape()[0].value
    npoint = trajectory.get_shape()[1].value
    num_frame = trajectory.get_shape()[2].value
    channel = trajectory_feature.get_shape()[3].value
    with tf.variable_scope(scope) as sc:
        sampled_trajectory, new_trajectory_feature, idx, grouped_trajectory = sample_and_group(trajectory, trajectory_feature, msample, radius, kgroup, knn, use_xyz,centralization = False,use_fps = use_fps) #(batch_size, msample, kgroup, num_frame, channel)
        if pooling=='max':
            new_trajectory_feature = tf.reduce_max(new_trajectory_feature, axis=[2], name='maxpool')
        elif pooling=='avg':
            new_trajectory_feature = tf.reduce_mean(new_trajectory_feature, axis=[2], name='avgpool')
        encoder_input = tf.reshape(new_trajectory_feature ,[-1,num_frame,channel+3])
        with tf.variable_scope("encoder",initializer=tf.orthogonal_initializer()):
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size), input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(num_layers)])
            enc_output,enc_state = tf.nn.dynamic_rnn(encoder_cell,encoder_input,dtype=tf.float32) #enc_output (-1, 128)
        lstm_state = enc_state[num_layers-1]
        lstm_state = lstm_state[1]
        net = tf.reshape(lstm_state ,[batch_size * msample,hidden_size])
        #axis (x y z u v w)
        # task1_xyz 
        task1_xyz = tf_util.fully_connected(net,128,scope='task1_xyz/fc1',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task1_xyz = tf_util.fully_connected(task1_xyz,128,scope='task1_xyz/fc2',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task1_xyz = tf_util.fully_connected(task1_xyz,3,scope='task1_xyz/fc3',weight_decay=0.0005,activation_fn=tf.nn.softsign)
        task1_xyz = tf.reshape(task1_xyz ,[batch_size , msample,3])
        '''
        task1_xyz = tf_util.conv1d(net, 128, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task1_xyz/fc1', bn_decay=bn_decay)
        task1_xyz = tf_util.conv1d(task1_xyz, 64, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task1_xyz/fc2', bn_decay=bn_decay)
        task1_xyz = tf_util.dropout(task1_xyz, keep_prob=0.5, is_training=is_training, scope='task1_xyz/dp1')
        task1_xyz = tf_util.conv1d(task1_xyz, 3, 1, padding='VALID', weight_decay=0.0005,activation_fn=tf.nn.softsign, scope='task1_xyz/fc3')
        '''
        # task1_uvw
        task1_uvw = tf_util.fully_connected(net,128,scope='task1_uvw/fc1',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task1_uvw = tf_util.fully_connected(task1_uvw,128,scope='task1_uvw/fc2',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task1_uvw = tf_util.fully_connected(task1_uvw,3,scope='task1_uvw/fc3',weight_decay=0.0005,activation_fn=tf.nn.softsign)
        task1_uvw = tf.reshape(task1_uvw ,[batch_size , msample,3])
        '''
        task1_uvw = tf_util.conv1d(net, 128, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task1_uvw/fc1', bn_decay=bn_decay)
        task1_uvw = tf_util.conv1d(task1_uvw, 64, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task1_uvw/fc2', bn_decay=bn_decay)
        task1_uvw = tf_util.dropout(task1_uvw, keep_prob=0.5, is_training=is_training, scope='task1_uvw/dp1')
        task1_uvw = tf_util.conv1d(task1_uvw, 3, 1, padding='VALID', weight_decay=0.0005,activation_fn=tf.nn.softsign, scope='task1_uvw/fc3')
        '''
        # task2 rspeed > 0
        task2_rs = tf_util.fully_connected(net,128,scope='task2_rs/fc1',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task2_rs = tf_util.fully_connected(task2_rs,128,scope='task2_rs/fc2',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task2_rs = tf_util.fully_connected(task2_rs,1,scope='task2_rs/fc3',weight_decay=0.0005,activation_fn=None)
        task2_rs = tf.reshape(task2_rs ,[batch_size , msample,1])
        '''
        task2_rs = tf_util.conv1d(net, 128, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task2_rs/fc1', bn_decay=bn_decay)
        task2_rs = tf_util.conv1d(task2_rs, 64, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task2_rs/fc2', bn_decay=bn_decay)
        task2_rs = tf_util.dropout(task2_rs, keep_prob=0.5, is_training=is_training, scope='task2_rs/dp1')
        task2_rs = tf_util.conv1d(task2_rs, 1, 1, padding='VALID',weight_decay=0.0005,activation_fn=tf.nn.softplus, scope='task2_rs/fc3')
        '''
        # task3 tspeed > 0
        task3_rs = tf_util.fully_connected(net,128,scope='task3_ts/fc1',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task3_rs = tf_util.fully_connected(task3_rs,128,scope='task3_ts/fc2',weight_decay=0.0005,bn=True,bn_decay=bn_decay,is_training=is_training)
        task3_ts = tf_util.fully_connected(task3_rs,1,scope='task3_ts/fc3',weight_decay=0.0005,activation_fn=None)
        task3_ts = tf.reshape(task3_ts ,[batch_size , msample,1])
        '''
        task3_ts = tf_util.conv1d(net, 128, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task3_ts/fc1', bn_decay=bn_decay)
        task3_ts = tf_util.conv1d(task3_ts, 64, 1, padding='VALID', weight_decay=0.0005,bn=True, is_training=is_training,scope='task3_ts/fc2', bn_decay=bn_decay)
        task3_ts = tf_util.dropout(task3_ts, keep_prob=0.5, is_training=is_training, scope='task3_ts/dp1')
        task3_ts = tf_util.conv1d(task3_ts, 1, 1, padding='VALID',weight_decay=0.0005,activation_fn=None, scope='task3_ts/fc3')
        '''
        first_frame = grouped_trajectory[:,:,:,0,:];
        last_frame = grouped_trajectory[:,:,:,num_frame-1,:];

        # motion_matrix     (batch_size, msample, 4, 4)
        matrix = get_motion_matrix(task1_xyz,task1_uvw, task2_rs, task3_ts)
        R_pred = matrix[:,:,0:3,0:3]
        #t = matrix[:,:,0:3,3:4]


        [R,_,m,uvw,theta,phi,R_diag,b] = cal_mat(first_frame,last_frame)
        rebuild_frame = mul_motion_matrix(matrix,first_frame)

        return rebuild_frame,last_frame,matrix,R_pred,R,m,uvw,theta,phi,R_diag,b,task1_xyz,task1_uvw,task2_rs,task3_ts,\
                  net,idx
    

if __name__=='__main__':
    trajectory = tf.zeros((2,1024,10,3))
    trajectory_feature = tf.zeros((2,1024,10,6))
    print(trajectory)
    print(trajectory_feature)
    msample = 128
    radius = 0.2
    kgroup = 64
    mlp = [128,128]
    mlp2 = [128,128,256]
    group_all = False
    knn = False
    is_training = True
    bn_decay = None
    scope = 'test_sa'
    sampled_trajectory, new_trajectory_feature, idx = pointnet_sa_module(trajectory, trajectory_feature, msample, radius, kgroup, mlp, mlp2, group_all, is_training, bn_decay, scope,knn=knn)
    print(sampled_trajectory)
    print(new_trajectory_feature)
    print(idx)
    mlp = [128,64,64]
    scope ='test_fp'
    new_trajectory_feature_fp = pointnet_fp_module(trajectory, sampled_trajectory, trajectory_feature, new_trajectory_feature, mlp, is_training, bn_decay, scope, bn=True)
    print(new_trajectory_feature_fp)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(sampled_trajectory))
        print(sess.run(new_trajectory_feature))
        print(sess.run(idx))
        print(sess.run(new_trajectory_feature_fp))





