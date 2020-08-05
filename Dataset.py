import numpy as np
import scipy.io as sio
import os 
import copy
class Dataset():
    def __init__(self,filepath,npoint=2048,add_noise=False,noise_scale=0.02,rotate=False):
        self.__filepath = filepath
        self.__npoint = npoint
        self.__add_noise = add_noise
        self.__noise_scale = noise_scale
        self.__rotate = rotate
        self.__Get_Filename()
        self.__Load_Data()
    def __Get_Filename(self):
        filename = []
        for root,dirs,files in os.walk(self.__filepath):
            tmp = [os.path.join(root,name) for name in files]
            filename.extend(tmp)
        filename.sort()
        self.__filename = filename
    
    def __Add_Noise(self,length):
        noise = np.random.rand(self.__npoint,length,3)*self.__noise_scale-self.__noise_scale/2
        return noise        

    def __Rotate(self):
        if self.__rotate:
            x1, x2, x3 = np.random.rand(3)
            R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                           [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                           [0, 0, 1]])
            v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                           [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                           [np.sqrt(1 - x3)]])
            H = np.eye(3) - 2 * v * v.T
            M = -H * R
        else:
            M = np.eye(3)
        return M

    def __Load_Data(self):
        data = []
        for path in self.__filename:
            tmp = sio.loadmat(path)['data'][0]
            data.extend(tmp)
        self.__data = data

    def __getitem__(self,index):
        T = int(copy.deepcopy(self.__data[index]['T']))
        input_ = copy.deepcopy(self.__data[index]['input'])
        m_ = copy.deepcopy(self.__data[index]['m'])
        axis_ = copy.deepcopy(self.__data[index]['axis'])
        label_ = copy.deepcopy(self.__data[index]['label'])
        if self.__rotate:
            M = self.__Rotate()
            for i in range(T):
                input_[:,i,0:3] = np.matmul(input_[:,i,0:3],M)
        if self.__add_noise:
            noise = self.__Add_Noise(T)
            input_[:,:,0:3] = input_[:,:,0:3] + noise

        if self.__rotate or self.__add_noise:
            for i in range(T-1):
                input_[:,i,3:6] = input_[:,i+1,0:3] - input_[:,i,0:3]


        return T,input_,m_,axis_,label_


    def __len__(self):
        return len(self.__data)



if __name__=='__main__':
    dataset = Dataset("train_data",rotate=True)
    T,input_,m_,axis_ = dataset[10]

    data = {'input_':input_}
    sio.savemat('test.mat',data)
    print(len(dataset))
