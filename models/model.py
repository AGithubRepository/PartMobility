import os
import sys
import math
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module,pointnet_rebuild_module


class Model():
    def __init__(self,batch_size,max_frame,npoint,num_layers,hidden_size,max_grad_norm,batch,learning_rate,bn_decay):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.max_frame = max_frame
        self.npoint = npoint
        self.batch = batch
        self.learning_rate = learning_rate
        self.bn_decay = bn_decay
    def get_train_placeholder(self):
        self.train_trajectory_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.npoint, self.max_frame, 6))
        self.is_training_pl = tf.placeholder(tf.bool, shape=())

    def get_eval_placeholder(self):
        self.eval_trajectory_pl = tf.placeholder(tf.float32, shape=(1, self.npoint, self.max_frame, 6))
        self.is_eval_pl = tf.placeholder(tf.bool, shape=())

    def get_model(self):
        """ Part segmentation PointNet-RNN, input is BxNxTx6 (XYZ DirectionX DirectionY DirectionZ), output Bx? """
        end_points = {}
        l0_xyz = tf.slice(self.train_trajectory_pl, [0,0,0,0], [-1,-1,-1,3])
        l0_points = tf.slice(self.train_trajectory_pl, [0,0,0,3], [-1,-1,-1,3])
        print(l0_points)
        xxx = l0_points
        # Set Abstraction layers
        l1_xyz, l1_points, self.l1_indices = pointnet_sa_module(l0_xyz, l0_points, msample=128, radius=0.35, kgroup=64, mlp=[64,64], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=self.bn_decay, scope='layer1', num_layers=self.num_layers,hidden_size=self.hidden_size,knn=False,centralization = True)
        l2_xyz, l2_points, self.l2_indices = pointnet_sa_module(l1_xyz, l1_points, msample=32, radius=0.7, kgroup=16, mlp=[128,128], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=self.bn_decay, scope='layer2', num_layers=self.num_layers,hidden_size=self.hidden_size,knn=False,centralization = True)
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, msample=1, radius=None, kgroup=None, mlp=[256,256], mlp2=None, group_all=True, is_training=self.is_training_pl, bn_decay=self.bn_decay, scope='layer3', num_layers=self.num_layers,hidden_size=self.hidden_size,knn=False,centralization = True)
        
        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], self.is_training_pl, self.bn_decay, scope='fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128,128], self.is_training_pl, self.bn_decay, scope='fa_layer2')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128], self.is_training_pl, self.bn_decay, scope='fa_layer3') # batch_size * npoint * num_frame * 128

        # Rebuild layers
        rebuild_frame,last_frame,self.matrix,R_pred,R,self.m,self.uvw,self.theta,self.phi,R_diag,b,\
                    self.task1_xyz,self.task1_uvw,self.task2_rs,self.task3_ts,self.net,self.rebuild_indices=\
                    pointnet_rebuild_module(l0_xyz, l0_points, msample=64, radius=0.25, kgroup=16, \
                    is_training=self.is_training_pl, bn_decay=self.bn_decay, scope='rebuild_layers', \
                    num_layers=self.num_layers, hidden_size=self.hidden_size,knn=True,use_fps=True)

        self.matrix_loss = self.__get_matrix_loss(self.matrix,self.m)
        self.angle_loss = self.__get_angle_loss(R_pred,R)
        self.rebuild_loss = self.__get_p2p_loss(rebuild_frame,last_frame)
        self.theta_loss = self.__get_theta_loss(self.task2_rs,self.theta)
        self.phi_loss = self.__get_phi_loss(self.task3_ts,self.phi)
        self.xyz_loss = self.__get_xyz_loss(self.task1_xyz,R_diag,b)
        self.uvw_loss = self.__get_uvw_loss(self.task1_uvw,self.uvw)
        self.l2_loss = tf.add_n(tf.get_collection('losses'))
        w1 = 0
        w2 = 1
        w3 = 1
        w4 = 1
        w5 = 1
        w6 = 1
        w7 = 1
        w8 = 1

        self.loss = w1*self.matrix_loss + w2*self.angle_loss + w3*self.rebuild_loss + w4*self.theta_loss + w5*self.phi_loss + w6*self.xyz_loss + \
                    w7*self.uvw_loss + w8*self.l2_loss
        self.aa = tf.gradients(self.loss,xxx)
    def optimizer_loss(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainable_variables = tf.trainable_variables()
        for variable in tf.trainable_variables():
            print(variable.name)
        self.grads = tf.gradients(self.loss/tf.to_float(self.batch_size),self.trainable_variables)
        self.grads,_ = tf.clip_by_global_norm(self.grads, self.max_grad_norm)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.apply_gradients(zip(self.grads,self.trainable_variables),global_step = self.batch)

    def __get_theta_loss(self, pred_theta,theta):
        pred_theta = tf.clip_by_value(pred_theta,-0.2,3.1415926)
        theta = tf.clip_by_value(theta,-0.2,3.1415926)
        loss = tf.sqrt(tf.square(tf.squeeze(pred_theta,axis=-1)-theta)+1e-10)
        loss = tf.reduce_mean(loss)
        return loss

    def __get_phi_loss(self, pred_phi, phi):
        pred_phi = tf.clip_by_value(pred_phi,-0.05,1)
        phi = tf.clip_by_value(phi,-0.05,1)
        loss = tf.sqrt(tf.square(tf.squeeze(pred_phi,axis=-1)-phi)+1e-10)
        loss = tf.reduce_mean(loss)
        return loss

    def __get_xyz_loss(self, pred_xyz,R_diag,b):
        pred_b = tf.matmul(R_diag,tf.expand_dims(pred_xyz,axis=-1))
        loss = tf.reduce_mean(tf.sqrt(tf.square(tf.squeeze(pred_b,axis=-1)+b)+1e-10),axis=-1)
        loss = tf.reduce_mean(loss)
        return loss
    def __get_uvw_loss(self, pred_uvw,uvw):
        loss = tf.reduce_sum(tf.square(pred_uvw-uvw),axis=-1)
        cos_distance = tf.clip_by_value(self.__cosine(pred_uvw,uvw),-1,1)
        zeros_distance = tf.ones_like(cos_distance)
        self.cos_distance = tf.where(tf.less(self.theta,0.2)&tf.less(self.phi,0.05), zeros_distance,cos_distance)
        loss = tf.sqrt(tf.square(1-self.cos_distance)+1e-10)
        loss = tf.reduce_mean(loss)
        return loss

    def __cosine(self,q,a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, axis=-1)+1e-10)
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, axis=-1)+1e-10)
        pooled_mul_12 = tf.reduce_sum(q * a, axis=-1)+1e-10
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-6, name="cosine_scores")
        return score 

    def __get_p2p_loss(self,rebuild_frame,frame):
        frob_sqr = tf.reduce_sum(tf.square(rebuild_frame-frame),axis = -1)
        frob = tf.reduce_mean(tf.sqrt(frob_sqr+1e-10),axis = -1)
        loss = tf.reduce_mean(frob)
        return loss

    def __get_matrix_loss(self,matrix_pred,matrix_abspose):
        frob_sqr = tf.reduce_mean(tf.square(matrix_pred - matrix_abspose),axis=[2,3])
        frob = tf.sqrt(frob_sqr+1e-10)
        loss_matrix = tf.reduce_mean(frob)
        return loss_matrix

    def __get_angle_loss(self,R_pred,R_abspose):
        frob_sqr = tf.reduce_sum(tf.square(R_pred-R_abspose),axis=[2,3])
        frob = tf.sqrt(frob_sqr+1e-10)
        loss_angle = 2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0,frob/(2*math.sqrt(2)))))
        return loss_angle
