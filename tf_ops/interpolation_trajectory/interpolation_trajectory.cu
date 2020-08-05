#include<stdio.h>
#include<time.h>
__global__ void threennKernel(int b, int n, int m, int t, const float * __restrict__ xyz1, const float * __restrict__ xyz2, float * __restrict__ dist, int * __restrict__ idx) {

    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<n;j+=blockDim.x){
            float best1=1e20, best2=1e20, best3=1e20;
            int besti1=0, besti2 = 0, besti3 = 0;
            for(int u=0;u<m;u++){
                float t_dist = 0;
                for(int v=0;v<t;v++){
                    int tmp_idx1 = i*n*t*3 + j*t*3 + v*3;
                    int tmp_idx2 = i*m*t*3 + u*t*3 + v*3;
                    float x1 = xyz1[tmp_idx1+0];
                    float y1 = xyz1[tmp_idx1+1];
                    float z1 = xyz1[tmp_idx1+2];
                    float x2 = xyz2[tmp_idx2+0];
                    float y2 = xyz2[tmp_idx2+1];
                    float z2 = xyz2[tmp_idx2+2];
                    t_dist += max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
                }
                t_dist /= t;
                
                if (t_dist<best1) {
                    best3=best2;
                    besti3=besti2;
                    best2=best1;
                    besti2=besti1;
                    best1=t_dist;
                    besti1=u;
                } else if (t_dist<best2) {
                    best3=best2;
                    besti3=besti2;
                    best2=t_dist;
                    besti2=u;
                } else if (t_dist<best3) {
                    best3=t_dist;
                    besti3=u;
                }
            }
            int tmp_idx = i*n*3+j*3;
            dist[tmp_idx+0]=best1;
            idx[tmp_idx+0]=besti1;
            dist[tmp_idx+1]=best2;
            idx[tmp_idx+1]=besti2;
            dist[tmp_idx+2]=best3;
            idx[tmp_idx+2]=besti3;
        }
    }
} 

void threennLauncher(int b,int n,int m,int t,const float *xyz1, const float *xyz2, float *dist, int *idx){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    threennKernel<<<32,512>>>(b,n,m,t,xyz1,xyz2,dist,idx);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("threennKernel:%f  \n",totaltime);
}



__global__ void threeinterpolateKernel(int b, int n, int m, int t,int c, const float * __restrict__ points, const int * __restrict__ idx, const float * __restrict__ weight, float * __restrict__ out) {

    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<n;j+=blockDim.x){
            int tmp_idx = i*n*3+j*3;
            float w1=weight[tmp_idx+0];
            float w2=weight[tmp_idx+1];
            float w3=weight[tmp_idx+2];
            int i1=idx[tmp_idx+0];
            int i2=idx[tmp_idx+1];
            int i3=idx[tmp_idx+2];
            for(int u=0;u<t;u++){
                for(int v=0;v<c;v++){
                    int tmp_idx1 = i*n*t*c + j*t*c + u*c + v;
                    int tmp_idx2 = i*m*t*c + i1*t*c + u*c + v;
                    int tmp_idx3 = i*m*t*c + i2*t*c + u*c + v;
                    int tmp_idx4 = i*m*t*c + i3*t*c + u*c + v;
                    out[tmp_idx1] = points[tmp_idx2]*w1 + points[tmp_idx3]*w2 + points[tmp_idx4]*w3;
                }
            }
        }
    }
}

void threeinterpolateLauncher(int b, int n, int m, int t, int c,const float *points, const int *idx, const float *weight, float *out){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    threeinterpolateKernel<<<32,512>>>(b,n,m,t,c,points,idx,weight,out);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("threeinterpolateKernel:%f  \n",totaltime);
}

// input: grad_out (b,n,c), idx (b,n,3), weight (b,n,3)
// output: grad_points (b,m,c)
__global__ void threeinterpolategradKernel(int b, int n, int m, int t, int c, const float * __restrict__ grad_out, const int * __restrict__ idx, const float * __restrict__ weight, float * __restrict__ grad_points) {



    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<n;j+=blockDim.x){
            int tmp_idx = i*n*3+j*3;
            float w1=weight[tmp_idx+0];
            float w2=weight[tmp_idx+1];
            float w3=weight[tmp_idx+2];
            int i1=idx[tmp_idx+0];
            int i2=idx[tmp_idx+1];
            int i3=idx[tmp_idx+2];
            for(int u=0;u<t;u++){
                for(int v=0;v<c;v++){
                    int tmp_idx1 = i*n*t*c + j*t*c + u*c + v;
                    int tmp_idx2 = i*m*t*c + i1*t*c + u*c + v;
                    int tmp_idx3 = i*m*t*c + i2*t*c + u*c + v;
                    int tmp_idx4 = i*m*t*c + i3*t*c + u*c + v;
                    atomicAdd(&grad_points[tmp_idx2],grad_out[tmp_idx1]*w1);
                    atomicAdd(&grad_points[tmp_idx3],grad_out[tmp_idx1]*w2);
                    atomicAdd(&grad_points[tmp_idx4],grad_out[tmp_idx1]*w3);
                }
            }
        }
    }
}

void threeinterpolategradLauncher(int b, int n, int m, int t, int c, const float *grad_out, const int *idx, const float *weight, float *grad_points){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    threeinterpolategradKernel<<<32,128>>>(b,n,m,t,c,grad_out,idx,weight,grad_points);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("threeinterpolategradKernel:%f  \n",totaltime);
}
