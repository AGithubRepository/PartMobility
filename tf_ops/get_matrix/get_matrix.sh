/usr/local/cuda-9.0/bin/nvcc get_matrix.cu -o get_matrix.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC



g++ -std=c++11 get_matrix.cpp get_matrix.cu.o -o get_matrix_so.so -shared -fPIC -I /home/vrlab/.local/lib/python3.5/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /home/vrlab/.local/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/home/vrlab/.local/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
