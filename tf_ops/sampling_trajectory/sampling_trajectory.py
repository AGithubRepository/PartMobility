import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_trajectory_module=tf.load_op_library(os.path.join(BASE_DIR, 'sampling_trajectory_so.so'))


def gather_trajectory(inp,idx):
    return sampling_trajectory_module.gather_trajecotry(inp,idx)
@tf.RegisterGradient('GatherTrajectory')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_trajectory_module.gather_trajectory_grad(inp,idx,out_g),None]

def farthest_point_sample_trajectory(inp,msample):
    return sampling_trajectory_module.farthest_point_sample_trajecotry(inp,msample)
ops.NoGradient('FarthestPointSample')

