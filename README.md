# PartMobility

## 1. Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.12 GPU version and Python 3.7 on Ubuntu 16.04. 

Before using the code, you should compile the customized TF operators which are included under  `tf_ops` first. Modify the file `tf_xxx_compile.sh` in each ops subfolder. Update `nvcc` and `python` path if necessary. 

Compile command：

    sh tf_xxx_compile.sh

## 2. Dataset

Our training and testing data are point cloud squence sampled from [Motion Dataset](http://motiondataset.zbuaa.com/).  You  can download the processed data from [here](https://drive.google.com/uc?export=download&id=1kfHhzf__gESS6k_yC6ntl5effv5MnhbV). For more categories of objects you can download the [enhance data](https://drive.google.com/uc?export=download&id=1TT7qjEOR80pMlIwW6R5n5ZPcxPLKW9s5).

## 3. Usage

To train the PartMobility model and predict the motion of point cloud sequence：

    python3 Train_and_Eval.py

After training, to test the predict result：

    python3 Test.py

Post-processing code is provided under `part_segment`, you can run the matlab code `segment_motion_part.m` to get the segment results and motion parameters of each part. 

More details of this work will be added in the future. 
