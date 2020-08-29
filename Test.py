from Dataset import Dataset


import argparse
import tensorflow as tf
import numpy as np
import os
import importlib
from datetime import datetime
import scipy.io as sio
import sys
import time
import random
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_frame', type=int, default=11, help='Max number of pointcloud sequence [default: 10]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--num_layers', type=int, default=1, help='Encoder rnn layers [default: 2]')
parser.add_argument('--hidden_size', type=int, default=128, help='Encoder rnn hidden size [default: 128]')
parser.add_argument('--max_grad_norm', type=int, default=5, help='Rnn gradient clip [default: GPU 5]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam, momentum or RMSProp[default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--train_data_dir', default='train', help='Train data dir [default: data]')
parser.add_argument('--val_data_dir', default='val_data', help='Val data dir [default: data]')
parser.add_argument('--test_data_dir', default='test', help='Test dir [default: data]')
parser.add_argument('--npoint', type=int, default=2048, help='Point number of pointcloud [default: 1024]')
FLAGS = parser.parse_args()


GPU_INDEX = FLAGS.gpu
MAX_FRAME = FLAGS.max_frame
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
NUM_LAYERS = FLAGS.num_layers
HIDDEN_SIZE = FLAGS.hidden_size
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
NPOINT = FLAGS.npoint
MAX_GRAD_NORM = FLAGS.max_grad_norm


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
sys.path.append(MODEL_DIR)
MODEL_FILE = importlib.import_module(FLAGS.model) # import network module
MODEL_PATH = os.path.join(MODEL_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
# SAVED_MODEL_PATH = LOG_DIR + '/model.ckpt'
SAVED_MODEL_PATH = './log/model.ckpt'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_PATH, LOG_DIR)) # bkp of model def
os.system('cp Test.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
# Create output dir
if not os.path.exists('./pred'): os.mkdir('./pred')

TRAIN_DATA_DIR = FLAGS.train_data_dir
VAL_DATA_DIR = FLAGS.val_data_dir
TEST_DATA_DIR = FLAGS.test_data_dir


TEST_DATA = Dataset(TEST_DATA_DIR)
num_data = len(TEST_DATA)
print(num_data)
'''
idxs = np.arange(num_data).astype(np.int)
np.random.shuffle(idxs)

# random.shuffle(idxs)
f = open('idx.txt', 'w')
num = min(num_data, 200)
for i in range(num):
    f.write(str(idxs[i]) + '\n')
f.close()
print('------->Down!')
'''



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_batch_data(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    input_ = np.zeros((bsize, NPOINT, MAX_FRAME, 6), dtype=np.float)
    m_ = np.zeros((bsize,80, 4), dtype=np.float)
    axis_ = np.zeros((bsize,10, 9), dtype=np.float)
    T = np.zeros((bsize,), dtype=np.int32)
    label_ = np.zeros((bsize,NPOINT,1),dtype=np.int32)
    for i in range(bsize):
        tmp_T,tmp_input_,tmp_m_,tmp_axis_,tmp_label_ = dataset[idxs[i+start_idx]]
        shuffle_idx = np.random.permutation(len(tmp_input_))
        input_[i,...] = tmp_input_[shuffle_idx,...]
        label_[i,...] = tmp_label_[shuffle_idx,...]
        length_motion_part = tmp_m_.shape[0];
        m_[i,0:length_motion_part,...] = tmp_m_
        length_motion_part = tmp_axis_.shape[0];
        axis_[i,0:length_motion_part,...] = tmp_axis_
        T[i] = tmp_T
    return input_,m_,axis_,T,label_


def train_and_eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            learning_rate = get_learning_rate(batch)
            MODEL = MODEL_FILE.Model(BATCH_SIZE,MAX_FRAME,NPOINT,NUM_LAYERS,HIDDEN_SIZE,MAX_GRAD_NORM,batch,learning_rate,bn_decay)
            MODEL.get_train_placeholder()
            #MODEL.get_eval_placeholder()
            MODEL.get_model()
            #MODEL.inference()
            MODEL.optimizer_loss()
            saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        with tf.Session(config=config) as sess:
            saver.restore(sess, SAVED_MODEL_PATH)
            #best_acc = -1
            for i in range(MAX_EPOCH):
                start_time = time.time();
                f = open('idx.txt', 'r')
                idxs = [int(idx) for idx in f]
                f.close()
                idxs = np.array(idxs)
                print(idxs)
                for j in range(int(num_data/BATCH_SIZE)):
                    start_idx = BATCH_SIZE*j
                    end_idx = BATCH_SIZE*j+BATCH_SIZE
                    input_,m_,axis_,T_,label_ = get_batch_data(TEST_DATA, idxs, start_idx, end_idx)
                    feed_dict = {
                            MODEL.train_trajectory_pl: input_,
                            MODEL.is_training_pl: False,
                    } 
                    loss,step,abspose_m,matrix_t,axis_xyz,axis_uvw,rs,ts,rebuild_loss,matrix_loss, \
                    angle_loss,theta_loss,uvw_loss,theta,phi,abspose_axis,l2_loss,cos_distance,phi_loss,\
                    net,xyz_loss,idx=\
                    sess.run([MODEL.loss,batch,MODEL.m,MODEL.matrix,\
                    MODEL.task1_xyz,MODEL.task1_uvw,MODEL.task2_rs,MODEL.task3_ts,\
                    MODEL.rebuild_loss,MODEL.matrix_loss,MODEL.angle_loss,MODEL.theta_loss,\
                    MODEL.uvw_loss,MODEL.theta,MODEL.phi,MODEL.uvw,MODEL.l2_loss,MODEL.cos_distance,MODEL.phi_loss,\
                    MODEL.net,MODEL.xyz_loss,MODEL.rebuild_indices],feed_dict=feed_dict)
                    log_string('---abspose_angle: %f'%(theta[0,0]/3.1415926535*180))
                    log_string('---abspose_trans: %f'%(phi[0,0]))
                    log_string('---abspose_axis: %f %f %f'%(abspose_axis[0,0,0],abspose_axis[0,0,1],abspose_axis[0,0,2]))
                    log_string('---cos_distance %f'%(cos_distance[0,0]))
                    log_string('---axis_xyz: %f %f %f'%(axis_xyz[0,0,0],axis_xyz[0,0,1],axis_xyz[0,0,2]))
                    log_string('---axis_uvw: %f %f %f'%(axis_uvw[0,0,0],axis_uvw[0,0,1],axis_uvw[0,0,2]))
                    log_string('---rs: %f'%(rs[0,0,0]/3.1415926535*180))#/3.14*180))
                    log_string('---ts: %f'%(ts[0,0,0]))
                    sio.savemat(os.path.join('pred','pred'+str(j).zfill(4)+'.mat'),{'input_gt':input_,'m_gt':m_,'axis_gt':axis_,'T':T_, \
                                               'axis_xyz':axis_xyz,'axis_uvw':axis_uvw,'theta':rs,'phi':ts, \
                                               'abspose_angle':theta,'abspose_trans':phi,'abspose_axis':abspose_axis,\
                                               'abspose_m':abspose_m,'rebuild_indices':idx,'label':label_})
                    log_string('loss: %f \nrebuild_loss: %f \nmatrix_loss: %f \nangle_loss: %f \ntheta_loss: %f \nuvw_loss: %f \nl2_loss: %f \nphi_loss: %f \nxyz_loss: %f \n' \
           % (loss,rebuild_loss,matrix_loss,angle_loss,theta_loss,uvw_loss,l2_loss,phi_loss,xyz_loss))
                duration = time.time() - start_time
                examples_per_sec = num_data/duration
                sec_per_batch = duration/int(num_data/BATCH_SIZE)
                log_string('\t%s: epoch: %f step: %f\n duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),i,step,duration,examples_per_sec,sec_per_batch))
                #sio.savemat('net/g_net'+str(i)+'.mat',{'g':aa,'net':net})
                


if __name__=='__main__':
    train_and_eval()
