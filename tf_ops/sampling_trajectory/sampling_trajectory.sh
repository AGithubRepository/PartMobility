/usr/local/cuda-9.0/bin/nvcc sampling_trajectory.cu -o sampling_trajectory.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC



g++ -std=c++11 sampling_trajectory.cpp sampling_trajectory.cu.o -o sampling_trajectory_so.so -shared -fPIC -I /home/vrlab/.local/lib/python3.5/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /home/vrlab/.local/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/home/vrlab/.local/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
