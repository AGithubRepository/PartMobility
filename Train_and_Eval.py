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
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_frame', type=int, default=11, help='Max number of pointcloud sequence [default: 10]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--num_layers', type=int, default=1, help='Encoder rnn layers [default: 2]')
parser.add_argument('--hidden_size', type=int, default=128, help='Encoder rnn hidden size [default: 128]')
parser.add_argument('--max_grad_norm', type=int, default=5, help='Rnn gradient clip [default: GPU 5]')
parser.add_argument('--learning_rate', type=float, default=0.0015, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam, momentum or RMSProp[default: adam]')
parser.add_argument('--decay_step', type=int, default=5000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--train_data_dir', default='data/train_data', help='Train data dir [default: data]')
parser.add_argument('--val_data_dir', default='data/val_data', help='Val data dir [default: data]')
parser.add_argument('--test_data_dir', default='data/test_data', help='Test dir [default: data]')
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

start_t = 0
end_t = 0


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
sys.path.append(MODEL_DIR)
MODEL_FILE = importlib.import_module(FLAGS.model) # import network module
MODEL_PATH = os.path.join(MODEL_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

# For visualization loss
TENSORBOARD_LOG = './Tensorboard_log'
if not os.path.exists(TENSORBOARD_LOG): os.mkdir(TENSORBOARD_LOG)

os.system('cp %s %s' % (MODEL_PATH, LOG_DIR)) # bkp of model def
os.system('cp Train_and_Eval.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_and_eval.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

TRAIN_DATA_DIR = FLAGS.train_data_dir
VAL_DATA_DIR = FLAGS.val_data_dir
TEST_DATA_DIR = FLAGS.test_data_dir


TRAIN_DATA = Dataset(TRAIN_DATA_DIR)
#EVAL_DARA = Dataset(VAL_DATA_DIR)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=False)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch,
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


def time_use_cal(time_use):

    time_use = int(time_use)
    secs = int(time_use % 60)
    mins = int(time_use / 60)
    hours = int(mins / 60)
    mins = int(mins % 60)
    time_info = 'Time use: ' + str(hours) + 'h ' + str(mins) + 'm ' + str(secs) + 's'

    return time_info


def train_and_eval():
    tol_t = 0
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            learning_rate = get_learning_rate(batch)
            # ------------------------------------------------------------
            tf.summary.scalar('bn_decay', bn_decay)
            tf.summary.scalar('learning_rate', learning_rate)
            # ------------------------------------------------------------
            MODEL = MODEL_FILE.Model(BATCH_SIZE,MAX_FRAME,NPOINT,NUM_LAYERS,HIDDEN_SIZE,MAX_GRAD_NORM,batch,learning_rate,bn_decay)
            MODEL.get_train_placeholder()
            #MODEL.get_eval_placeholder()
            MODEL.get_model()
            #MODEL.inference()
            MODEL.optimizer_loss()
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)
            #saver = tf.train.Saver(max_to_keep = 10)
            # ------------------------------------------------------------
            tf.summary.scalar('matrix_loss', MODEL.matrix_loss)
            tf.summary.scalar('angle_loss', MODEL.angle_loss)
            tf.summary.scalar('rebuild_loss', MODEL.rebuild_loss)
            tf.summary.scalar('phi_loss', MODEL.phi_loss)
            tf.summary.scalar('xyz_loss', MODEL.xyz_loss)
            tf.summary.scalar('uvw_loss', MODEL.uvw_loss)
            tf.summary.scalar('l2_loss', MODEL.l2_loss)
            tf.summary.scalar('theta_loss', MODEL.theta_loss)
            # ------------------------------------------------------------
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # ------------------------------------------------------------
        # Add summary writers
        merged = tf.summary.merge_all()
        # ------------------------------------------------------------
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            # ------------------------------------------------------------
            train_writer = tf.summary.FileWriter(os.path.join(TENSORBOARD_LOG, 'train'), sess.graph)
            # ------------------------------------------------------------
            sess.run(init)
            #best_acc = -1
            for i in range(MAX_EPOCH):
                start_time = time.time();
                num_data = len(TRAIN_DATA)
                print(num_data)
                print("TRAIN EPOCH %d:" %(i))
                idxs = np.random.permutation(num_data)
                all_loss = 0.0
                all_rebuild_loss = 0.0
                all_matrix_loss = 0.0
                all_angle_loss = 0.0
                all_theta_loss = 0.0
                all_phi_loss = 0.0
                all_uvw_loss = 0.0
                all_l2_loss = 0.0
                all_xyz_loss = 0.0
                batch_num = int(num_data/BATCH_SIZE)
                for j in range(int(num_data/BATCH_SIZE)):
                    start_idx = BATCH_SIZE*j
                    end_idx = BATCH_SIZE*j+BATCH_SIZE
                    input_,m_,axis_,T_,label_ = get_batch_data(TRAIN_DATA, idxs, start_idx, end_idx)
                    feed_dict = {
                            MODEL.train_trajectory_pl: input_,
                            MODEL.is_training_pl: True,
                    } 
                    _,loss,step,abspose_m,matrix_t,axis_xyz,axis_uvw,rs,ts,rebuild_loss,matrix_loss, \
                    angle_loss,theta_loss,uvw_loss,theta,phi,abspose_axis,l2_loss,cos_distance,phi_loss,\
                    net,xyz_loss,t_learning_rate,\
                    step,summary=\
                    sess.run([MODEL.train_op,MODEL.loss,batch,MODEL.m,MODEL.matrix,\
                    MODEL.task1_xyz,MODEL.task1_uvw,MODEL.task2_rs,MODEL.task3_ts,\
                    MODEL.rebuild_loss,MODEL.matrix_loss,MODEL.angle_loss,MODEL.theta_loss,\
                    MODEL.uvw_loss,MODEL.theta,MODEL.phi,MODEL.uvw,MODEL.l2_loss,MODEL.cos_distance,MODEL.phi_loss,\
                    MODEL.net,MODEL.xyz_loss,learning_rate,\
                    MODEL.batch,merged],feed_dict=feed_dict)
                    print('------> epoch:', i + 1, '(', round((i + 1)*100/MAX_EPOCH, 1), '%)', 'batch:', round((j + 1)*100/batch_num, 1), '%')
                    if i > 0:
                        per_epoch = tol_t / i
                        remain_t = per_epoch * (MAX_EPOCH - i - 1) + per_epoch / batch_num * (batch_num - j - 1)
                        print('Remain', time_use_cal(remain_t))
                    print('---cos_distance', cos_distance[0,0])
                    print('---abspose_axis', abspose_axis[0,0,:])
                    log_string('---axis_xyz: %f %f %f'%(axis_xyz[0,0,0],axis_xyz[0,0,1],axis_xyz[0,0,2]))
                    log_string('---axis_uvw: %f %f %f'%(axis_uvw[0,0,0],axis_uvw[0,0,1],axis_uvw[0,0,2]))
                    print('---abspose_angle', theta[0,0])#/3.14*180)
                    log_string('---rs: %f'%(rs[0,0,0]))#/3.14*180))
                    print('---abspose_trans', phi[0,0])
                    log_string('---ts: %f'%(ts[0,0,0]))
                    log_string('---rebuild_loss: %f'%(rebuild_loss))
                    log_string('---theta_loss: %f'%(theta_loss))
                    # log_string('------static_axes_num: %f'%(static_axes_num))
                    print('---matrix_t')
                    print(matrix_t[0,0,:,:])
                    print('---abspose_m_mid')
                    print(abspose_m[0,0,:,:])
                    log_string('---learning_rate: %f'%(t_learning_rate))
                    # '''
                    #----------------------------------------------------------
                    train_writer.add_summary(summary, step)
                    #----------------------------------------------------------
                    all_loss = all_loss + loss
                    all_rebuild_loss = all_rebuild_loss + rebuild_loss
                    all_matrix_loss = all_matrix_loss + matrix_loss
                    all_angle_loss = all_angle_loss + angle_loss
                    all_theta_loss = all_theta_loss + theta_loss
                    all_uvw_loss = all_uvw_loss + uvw_loss
                    all_phi_loss = all_phi_loss + phi_loss
                    all_xyz_loss = all_xyz_loss + xyz_loss
                    all_l2_loss = all_l2_loss + l2_loss
                duration = time.time() - start_time
                tol_t += duration
                examples_per_sec = num_data/duration
                sec_per_batch = duration/int(num_data/BATCH_SIZE)
                log_string('\t%s: epoch: %f step: %f loss: %f \n duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),i,step,all_loss/int(num_data/BATCH_SIZE),duration,examples_per_sec,sec_per_batch))
                log_string('rebuild_loss: %f \nmatrix_loss: %f \nangle_loss: %f \ntheta_loss: %f \nuvw_loss: %f \nl2_loss: %f \nphi_loss: %f \nxyz_loss: %f \n' \
           % (all_rebuild_loss/int(num_data/BATCH_SIZE),all_matrix_loss/int(num_data/BATCH_SIZE),all_angle_loss/int(num_data/BATCH_SIZE),all_theta_loss/int(num_data/BATCH_SIZE),all_uvw_loss/int(num_data/BATCH_SIZE),all_l2_loss/int(num_data/BATCH_SIZE),all_phi_loss/int(num_data/BATCH_SIZE),all_xyz_loss/int(num_data/BATCH_SIZE)))
                if i % 10 == 0:
                    save_path = saver.save(sess, LOG_DIR+"/model"+str(i)+".ckpt")
                    print("Model saved in file: %s" % save_path)
                    save_path2 = saver.save(sess, LOG_DIR+"/model.ckpt")
                    print("Model saved in file: %s" % save_path2)

    return tol_t
            

if __name__=='__main__':

    tol_t = train_and_eval()

    time_use = int(tol_t)
    time_info = time_use_cal(time_use) + '\n'
    epoch_info = 'Epoch: ' + str(MAX_EPOCH) + '\n'
    batch_size_info = 'Batch size: ' + str(BATCH_SIZE) + '\n'
    train_data_info = 'Train data: ' + str(len(TRAIN_DATA)) + '\n'
    model_path = './log/model.ckpt.data-00000-of-00001'
    model_size = round(os.path.getsize(model_path) * 1.0 / 1000/ 1000, 1)
    model_size_info = 'Model size: ' + str(model_size) + 'm\n'

    print(time_info)
    print(epoch_info)
    print(batch_size_info)
    print(train_data_info)
    print(model_size_info)
    # Write training info
    cur_path = os.path.abspath(__file__)
    net_name = cur_path.split('/')[-2]
    out_f = open(net_name + '.txt', 'w')
    out_f.write(time_info)
    out_f.write(epoch_info)
    out_f.write(batch_size_info)
    out_f.write(train_data_info)
    out_f.write(model_size_info)
                
