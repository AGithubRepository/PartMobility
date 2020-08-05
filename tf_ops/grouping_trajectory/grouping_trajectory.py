import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
grouping_trajectory_module=tf.load_op_library(os.path.join(BASE_DIR, 'grouping_trajectory_so.so'))

def query_ball_trajectory(xyz1, xyz2, radius, nsample):
    return grouping_trajectory_module.query_ball_trajectory(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPoint')
def select_top_k(k, dist):
    return grouping_trajectory_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')
def group_trajectory(inp, idx):
    return grouping_trajectory_module.group_trajectory(inp, idx)
@tf.RegisterGradient('GroupTrajectory')
def _group_trajectory_grad(op, grad_out):
    inp = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_trajectory_module.group_trajectory_grad(inp, idx, grad_out), None]

def knn_trajectory(xyz1, xyz2, k):
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    t = xyz1.get_shape()[2].value
    c = xyz1.get_shape()[3].value
    m = xyz2.get_shape()[1].value
    xyz1 = tf.expand_dims(xyz1,axis=1)
    xyz2 = tf.expand_dims(xyz2,axis=2)
    xyz1 = tf.tile(xyz1, [1,m,1,1,1])
    xyz2 = tf.tile(xyz2, [1,1,n,1,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    dist = tf.reduce_mean(dist,-1)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    return idx,val 
    

