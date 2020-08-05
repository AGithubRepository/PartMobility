#include<stdio.h>
#include<time.h>
__global__ void gathertrajctoryKernel(int b,int n,int m,int t,const float * __restrict__ inp,const int * __restrict__ idx, float * __restrict__ out){
    for(int i = blockIdx.x;i<b;i+=gridDim.x){
        for(int j = threadIdx.x;j<m; j+=blockDim.x){
            int tmp = idx[i*m+j];
            for(int k = 0;k<t;k++){
                int tmp_idx1 = ((i*m+j)*t+k);
                int tmp_idx2 = ((i*n+tmp)*t+k);
                out[tmp_idx1*3+0]=inp[tmp_idx2*3+0];
                out[tmp_idx1*3+1]=inp[tmp_idx2*3+1];
                out[tmp_idx1*3+2]=inp[tmp_idx2*3+2];
            }
        }
    }
}

void gathertrajctoryLauncher(int b,int n,int m,int t,const float * inp,const int *idx, float *out){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    gathertrajctoryKernel<<<32,512>>>(b,n,m,t,inp,idx,out);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("gathertrajctoryKernel:%f b:%d n:%d m:%d t:%d \n",totaltime,b,n,m,t);

}

__global__ void gathertrajectorygradKernel(int b,int n,int m,int t,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
    for(int i = blockIdx.x;i<b;i+=gridDim.x){
        for(int j = threadIdx.x;j<m; j+=blockDim.x){
            int tmp = idx[i*m+j];
            for(int k = 0;k<t;k++){
                int tmp_idx1 = ((i*m+j)*t+k);
                int tmp_idx2 = ((i*n+tmp)*t+k);
                atomicAdd(&inp_g[tmp_idx2*3+0],out_g[tmp_idx1*3+0]);
                atomicAdd(&inp_g[tmp_idx2*3+1],out_g[tmp_idx1*3+1]);
                atomicAdd(&inp_g[tmp_idx2*3+2],out_g[tmp_idx1*3+2]);
            }
        }
    }
}

void gathertrajectorygradLauncher(int b,int n,int m,int t,const float * out_g,const int * idx,float * inp_g){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    gathertrajectorygradKernel<<<32,128>>>(b,n,m,t,out_g,idx,inp_g);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("gathertrajectorygradKernel:%f  \n",totaltime);

}

__global__ void farthestpointsamplingtrajectoryKernel(int b,int n,int m,int t,const float * __restrict__ trajectory,float * __restrict__ temp,int * __restrict__ sample_idx){
    const int BlockSize = 512;
    __shared__ float max_dists[BlockSize];
    __shared__ int dists_idx[BlockSize];
    const int BufferSize=2880;
    __shared__ float buf[BufferSize*3];
    const int framesize = 64;
    __shared__ float framebufx[framesize];
    __shared__ float framebufy[framesize];
    __shared__ float framebufz[framesize];
    for(int i=blockIdx.x;i<b;i+=gridDim.x){   //batch init
        int last = 0;
        if (threadIdx.x==0)
            sample_idx[i*m+0]=last;
        for(int j=threadIdx.x;j<n;j+=blockDim.x){
            temp[blockIdx.x*n+j]=1e38;
        }
        for(int j=threadIdx.x;j<min(BufferSize,n*t)*3;j+=blockDim.x){
            buf[j]=trajectory[i*n*t*3+j];
        }
        __syncthreads();
        for(int j=0;j<m;j++){ //each sample step
            float t_max_dists = -1;
            int t_dist_idx = 0;
            for(int k=0;k<min(t,framesize);k++){
                int tmp_idx = i*n*t*3 + last*t*3 + k*3;
                framebufx[k] = trajectory[tmp_idx + 0];
                framebufy[k] = trajectory[tmp_idx + 1];
                framebufz[k] = trajectory[tmp_idx + 2]; 
            }
            for(int k=threadIdx.x;k<n;k+=blockDim.x){ //compute dis
                float td=temp[blockIdx.x*n+k];
                float td_new = 0;
                float tx1=0,ty1=0,tz1=0,tx2=0,ty2=0,tz2=0;
                for(int u=0;u<t;u++){
                    if(u<framesize){
                        int tmp_idx = u;
                        tx1=framebufx[tmp_idx];
                        ty1=framebufy[tmp_idx];
                        tz1=framebufz[tmp_idx];
                    }else{
                        int tmp_idx = i*n*t*3 + last*t*3 + u*3;
                        tx1=trajectory[tmp_idx+0];
                        ty1=trajectory[tmp_idx+1];
                        tz1=trajectory[tmp_idx+2];
                    }
                    if(k*t+u<BufferSize){
                        int tmp_idx = (k*t+u)*3;
                        tx2=buf[tmp_idx+0];
                        ty2=buf[tmp_idx+1];
                        tz2=buf[tmp_idx+2];
                    }else{
                        int tmp_idx = i*n*t*3 + k*t*3 + u*3;
                        tx2=trajectory[tmp_idx+0];
                        ty2=trajectory[tmp_idx+1];
                        tz2=trajectory[tmp_idx+2];
                    }
                    td_new += max(((tx2-tx1)*(tx2-tx1)+(ty2-ty1)*(ty2-ty1)+(tz2-tz1)*(tz2-tz1)),1e-20f);
                }
                td_new/=t;
                float d2=min(td,td_new);
                if(d2!=td)
                    temp[blockIdx.x*n+k]=d2;
                if(d2>t_max_dists){
                    t_max_dists=d2;
                    t_dist_idx=k;
                }
            }
            max_dists[threadIdx.x]=t_max_dists;
            dists_idx[threadIdx.x]=t_dist_idx;
            for (int u=0;(1<<u)<blockDim.x;u++){ //reduce min
                __syncthreads();
               if (threadIdx.x<(blockDim.x>>(u+1))){
                   int i1=(threadIdx.x*2)<<u;
                   int i2=(threadIdx.x*2+1)<<u;
                   if (max_dists[i1]<max_dists[i2]){
                       max_dists[i1]=max_dists[i2];
                       dists_idx[i1]=dists_idx[i2];
                   }
               }
            }
            __syncthreads();
            last=dists_idx[0];
            if (threadIdx.x==0)
                sample_idx[i*m+j]=last;
        }
    }
}
//require 32*n working space
void farthestpointsamplingtrajectoryLauncher(int b,int n,int m,int t,const float * inp,float * temp,int *out){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    farthestpointsamplingtrajectoryKernel<<<32,512>>>(b,n,m,t,inp,temp,out);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("farthestpointsamplingtrajectoryKernel:%f  \n",totaltime);

}
