import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
get_matrix_module=tf.load_op_library(os.path.join(BASE_DIR, 'get_matrix_so.so'))

def get_motion_matrix(inp_axis_xyz,inp_axis_uvw, inp_rspeed, inp_tspeed):
    inp_rspeed = inp_rspeed #* 3.14159265358979323846 / 180
    inp_uvw_norm = tf.sqrt(tf.reduce_sum(inp_axis_uvw * inp_axis_uvw,axis = -1)+1e-6) + 1e-6
    inp_axis_uvw = inp_axis_uvw / tf.expand_dims(inp_uvw_norm,axis = -1)
    return get_matrix_module.get_motion_matrix(inp_axis_xyz, inp_axis_uvw, inp_rspeed, inp_tspeed)
@tf.RegisterGradient('GetMotionMatrix')
def _get_motion_matrix_grad(op, grad_out):
    inp_axis_xyz = op.inputs[0]
    inp_axis_uvw = op.inputs[1]
    inp_rspeed = op.inputs[2]
    inp_tspeed = op.inputs[3]
    inp_axis_xyz_g, inp_axis_uvw_g, inp_rspeed_g, inp_tspeed_g = get_matrix_module.get_motion_matrix_grad(inp_axis_xyz, inp_axis_uvw, inp_rspeed,inp_tspeed, grad_out)
    return [inp_axis_xyz_g, inp_axis_uvw_g, inp_rspeed_g, inp_tspeed_g]
def mul_motion_matrix(inp_matrix,inp_point):
    return get_matrix_module.mul_motion_matrix(inp_matrix, inp_point)
@tf.RegisterGradient('MulMotionMatrix')
def _mul_motion_matrix_grad(op, grad_out):
    inp_matrix = op.inputs[0]
    inp_point = op.inputs[1]
    xxx = get_matrix_module.mul_motion_matrix_grad(inp_matrix, inp_point, grad_out)
    return [get_matrix_module.mul_motion_matrix_grad(inp_matrix, inp_point, grad_out), None]



if __name__=='__main__':
    import numpy as np
    import time
    import scipy.io as sio
    np.random.seed(100)
    t_inp_point = (np.random.random((32,16,16,3)).astype('float32') - 0.5)
    t_inp_axis = (np.random.random((32,16,6)).astype('float32') - 0.5)
    t_inp_type = np.random.randint(0,4,(32,16,1)).astype('int32')
    t_inp_rspeed = np.random.random((32,16,1)).astype('float32')
    t_inp_tspeed = np.random.random((32,16,1)).astype('float32')
    with tf.device('/gpu:0'):
        inp_point = tf.constant(t_inp_point)
        inp_axis = tf.constant(t_inp_axis)
        inp_axis_xyz = inp_axis[:,:,0:3]
        inp_axis_uvw = inp_axis[:,:,3:6]
        inp_type = tf.constant(t_inp_type)
        print(inp_type)
        inp_rspeed = tf.constant(t_inp_rspeed)
        inp_tspeed = tf.constant(t_inp_tspeed)
        matrix = get_motion_matrix(inp_axis_xyz,inp_axis_uvw, inp_type, inp_rspeed, inp_tspeed)
        print(matrix)
        out = mul_motion_matrix(matrix, inp_point)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            t_inp_axis,t_inp_type,t_inp_rspeed,t_inp_tspeed,t_matrix,t_inp_point,t_out = sess.run([inp_axis,inp_type,inp_rspeed,inp_tspeed,matrix,inp_point,out])
        print(time.time() - now)
        print(t_matrix.shape, t_matrix.dtype)
        #print ret
        sio.savemat('1.mat',{"inp_axis":t_inp_axis,"inp_type":t_inp_type,"inp_rspeed":t_inp_rspeed,"inp_tspeed":t_inp_tspeed,"matrix":t_matrix,"inp_point":t_inp_point,"out":t_out})
