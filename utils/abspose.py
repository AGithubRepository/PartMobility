import tensorflow as tf



def abspose(pc1,pc2):
    '''
    pc1:(batch_size,msample,kgroup,3)
    pc2:(batch_size,msample,kgroup,3)
    '''
    batch_size = pc1.get_shape()[0].value
    msample = pc1.get_shape()[1].value
    kgroup = pc1.get_shape()[2].value
    pc1 = tf.reshape(pc1,[-1,kgroup,3])
    pc2 = tf.reshape(pc2,[-1,kgroup,3])
    pbar = tf.reduce_mean(pc1,axis = 1,keep_dims = True)
    qbar = tf.reduce_mean(pc2,axis = 1,keep_dims = True)
    pbar_tile = tf.tile(pbar,[1,kgroup,1])
    qbar_tile = tf.tile(qbar,[1,kgroup,1])
    delta_0 = pc1 - pbar_tile
    delta_t = tf.transpose(pc2 - qbar_tile, [0,2,1])
    A = tf.matmul(delta_t,delta_0)
    s, u, v = tf.svd(A)
    v_T = tf.transpose(v, [0,2,1])
    uv_T = tf.matmul(u,v_T)
    det_uv_T = tf.matrix_determinant(uv_T)
    diag = tf.ones([batch_size*msample,2])
    diag = tf.concat([diag,tf.expand_dims(det_uv_T,axis=-1)],axis = -1)
    C = tf.matrix_diag(diag)
    R = tf.matmul(tf.matmul(u,C),v_T)
    t = qbar - tf.transpose(tf.matmul(R,tf.transpose(pbar,[0,2,1])),[0,2,1])
    R = tf.reshape(R,[batch_size,msample,3,3])
    t = tf.reshape(t,[batch_size,msample,1,3])
    return R,t

def cal_mat(pc1,pc2):
    R,t = abspose(pc1,pc2)
    batch_size = R.get_shape()[0].value
    msample = R.get_shape()[1].value
    t = tf.transpose(t,[0,1,3,2])
    m = tf.concat([R,t],axis = -1)
    t = tf.squeeze(t,axis=-1)

    temp_l = tf.zeros([batch_size,msample,3])
    temp_r = tf.ones([batch_size,msample,1])
    temp = tf.expand_dims(tf.concat([temp_l,temp_r],axis = -1),axis=2)
    m = tf.concat([m,temp],axis = 2)

    theta = tf.trace(R)
    theta = tf.clip_by_value((theta - 1)/2,-1,1)
    theta = tf.acos(theta)

    axis_u = R[:,:,2,1]-R[:,:,1,2]
    axis_v = R[:,:,0,2]-R[:,:,2,0]
    axis_w = R[:,:,1,0]-R[:,:,0,1]
    uvw = tf.concat([tf.expand_dims(axis_u,axis=-1),tf.expand_dims(axis_v,axis=-1),tf.expand_dims(axis_w,axis=-1)],axis = -1)
    theta_tile = tf.tile(tf.expand_dims(theta,axis=-1),[1,1,3])
    uvw = tf.where(tf.greater_equal(theta_tile,0.2),uvw,t)
    uvw = uvw / tf.expand_dims((tf.sqrt(tf.reduce_sum(uvw * uvw,axis = -1)) + 1e-8),axis=-1)

    phi = tf.reduce_sum(uvw * t,axis=-1)
    diag = tf.ones([batch_size,msample,3])
    diag = tf.matrix_diag(diag)
    constant = tf.zeros([batch_size,msample,3])
    b = tf.where(tf.greater_equal(theta_tile,0.2), t - uvw*tf.tile(tf.expand_dims(phi,axis=-1),[1,1,3]), constant)
    theta_tile = tf.tile(tf.expand_dims(theta_tile,axis=-1),[1,1,1,3])
    R_diag = tf.where(tf.greater_equal(theta_tile,0.2), R-diag, diag)
    

    return R,t,m,uvw,theta,phi,R_diag,b

if __name__=='__main__':
    import numpy as np
    import time
    import scipy.io as sio
    with tf.device('/gpu:0'):
        pc1 = tf.placeholder(tf.float32,shape=(1000,100,100,3))
        pc2 = tf.placeholder(tf.float32,shape=(1000,100,100,3))
        #[R,t] = abspose(pc1, pc2)
        R,t,m,uvw,theta,phi,R_diag,b = cal_mat(pc1,pc2)
    with tf.Session('') as sess:
        data = sio.loadmat('test_abspose.mat')
        t_pc1 = data['pc1'];
        t_pc2 = data['pc2'];
        m_gt = data['m_'];
        axis_gt = data['axis'];
        temp_m,temp_uvw,temp_theta,temp_phi,temp_R_diag,temp_b = sess.run([m,uvw,theta,phi,R_diag,b],feed_dict = {pc1:t_pc1,pc2:t_pc2})
        sio.savemat('abspose_gpu.mat',{'m_gpu':temp_m,'m':m_gt,'axis':axis_gt,'uvw':temp_uvw,'theta':temp_theta,'phi':temp_phi,'R_diag':temp_R_diag,'b':temp_b})
