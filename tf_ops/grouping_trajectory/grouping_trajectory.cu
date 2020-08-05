#include<stdio.h>
#include<time.h>
__global__ void grouptrajectoryKernel(int b, int n, int c, int m,int t, int k, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<m;j+=blockDim.x){
            for(int u=0;u<k;u++){
                int tidx = idx[i*m*k+j*k+u];
                for(int v=0;v<t;v++){
                    for(int w=0;w<c;w++){
                        out[i*m*k*t*c+j*k*t*c+u*t*c+v*c+w] = inp[i*n*t*c+tidx*t*c+v*c+w];
                    }
                }
            }
        }
    }

}

void grouptrajectoryLauncher(int b, int n, int c, int m,int t, int k, const float *inp, const int *idx, float *out){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    grouptrajectoryKernel<<<32,512>>>(b,n,c,m,t,k,inp,idx,out);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("grouptrajectoryKernel:%f b:%d n:%d c:%d m:%d t:%d k:%d\n",totaltime,b,n,c,m,t,k);

}

__global__ void grouptrajectorygradKernel(int b, int n, int c, int m, int t, int k,const float * __restrict__ out_g, const int * __restrict__ idx, float * __restrict__ inp_g) {
    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<m;j+=blockDim.x){
            for(int u=0;u<k;u++){
                int tidx = idx[i*m*k+j*k+u];
                for(int v=0;v<t;v++){
                    for(int w=0;w<c;w++){
                        atomicAdd(&inp_g[i*n*t*c+tidx*t*c+v*c+w],out_g[i*m*k*t*c+j*k*t*c+u*t*c+v*c+w]);
                    }
                }
            }
        }
    }
}

void grouptrajectorygradLauncher(int b, int n, int c, int m, int t, int k,const float *out_g, const int *idx, float *inp_g){
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    grouptrajectorygradKernel<<<32,512>>>(b,n,c,m,t,k,out_g,idx,inp_g);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("grouptrajectorygradKernel:%f b:%d n:%d c:%d m:%d t:%d k:%d\n",totaltime,b,n,c,m,t,k);
}



// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void queryballtrajectoryKernel(int b, int n, int m,int t, float radius, int kgroup, const float * __restrict__ xyz1, const float * __restrict__ xyz2, int * __restrict__ idx, int * __restrict__ pts_cnt) {

    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<m;j+=blockDim.x){
            int cnt = 0;
            for(int u=0;u<n;u++){
                if(cnt == kgroup) break;
                float t_sum = 0;
                for(int v=0;v<t;v++){
                    int tmp_idx1 = i*n*t*3 + u*t*3 + v*3;
                    int tmp_idx2 = i*m*t*3 + j*t*3 + v*3;
                    float x2=xyz2[tmp_idx2+0];
                    float y2=xyz2[tmp_idx2+1];
                    float z2=xyz2[tmp_idx2+2];
                    float x1=xyz1[tmp_idx1+0];
                    float y1=xyz1[tmp_idx1+1];
                    float z1=xyz1[tmp_idx1+2];
    	            t_sum+=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
                }
                t_sum/=t;
                if(t_sum<radius){
                    if(cnt==0){
                        for(int w=0;w<kgroup;w++){
                            idx[i*m*kgroup+j*kgroup+w] = u;
                        }
                    }
                    idx[i*m*kgroup+j*kgroup+cnt] = u;
                    cnt+=1;
                }
            }
            pts_cnt[i*m+j] = cnt;
        }
    }
}


void queryballtrajectoryLauncher(int b, int n, int m,int t, float radius, int kgroup, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    queryballtrajectoryKernel<<<32,512>>>(b,n,m,t,radius,kgroup,xyz1,xyz2,idx,pts_cnt);
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("queryballtrajectoryKernel:%f  \n",totaltime);
    //cudaDeviceSynchronize();
}


__global__ void selectionsortKernel(int b, int n, int m, int k, const float * __restrict__ dist, int * __restrict__ outi, float * __restrict__ out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

void selectionsortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    //clock_t start,finish;
    //double totaltime;
    //start=clock();
    selectionsortKernel<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //finish=clock();
    //totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    //printf("selectionsortKernel:%f  \n",totaltime);
    //cudaDeviceSynchronize();
}

