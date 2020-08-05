#include<stdio.h>
#include<time.h>
#include<cmath>
__global__ void getmotionmatrixKernel(int b, int n, const float * __restrict__ inp_axis_xyz, const float * __restrict__ inp_axis_uvw, const float * __restrict__ inp_rspeed, const float * __restrict__ inp_tspeed, float * __restrict__ out) {
    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<n;j+=blockDim.x){
            int tmp_idx = i*n+j;
            int matrix_idx = i*n*16 + j*16;
            int axis_idx = i*n*3 + j*3;
            float tmp_rspeed = inp_rspeed[tmp_idx];
            float tmp_tspeed = inp_tspeed[tmp_idx];
            float x = inp_axis_xyz[axis_idx + 0];
            float y = inp_axis_xyz[axis_idx + 1];
            float z = inp_axis_xyz[axis_idx + 2];
            float u = inp_axis_uvw[axis_idx + 0];
            float v = inp_axis_uvw[axis_idx + 1];
            float w = inp_axis_uvw[axis_idx + 2];
            float cosaa = cos(tmp_rspeed);
            float sinaa = sin(tmp_rspeed);
            float tx = u * tmp_tspeed;
            float ty = v * tmp_tspeed;
            float tz = w * tmp_tspeed;
            out[matrix_idx + 0] = u*u+(v*v+w*w)*cosaa;
            out[matrix_idx + 1] = u*v*(1-cosaa)-w*sinaa;
            out[matrix_idx + 2] = u*w*(1-cosaa)+v*sinaa;
            out[matrix_idx + 3] = (x*(v*v+w*w)-u*(y*v+z*w))*(1-cosaa)+(y*w-z*v)*sinaa+tx;
            out[matrix_idx + 4] = u*v*(1-cosaa)+w*sinaa;
            out[matrix_idx + 5] = v*v+(u*u+w*w)*cosaa;
            out[matrix_idx + 6] = v*w*(1-cosaa)-u*sinaa;
            out[matrix_idx + 7] = (y*(u*u+w*w)-v*(x*u+z*w))*(1-cosaa)+(z*u-x*w)*sinaa+ty;
            out[matrix_idx + 8] = u*w*(1-cosaa)-v*sinaa;
            out[matrix_idx + 9] = v*w*(1-cosaa)+u*sinaa;
            out[matrix_idx +10] = w*w+(u*u+v*v)*cosaa;
            out[matrix_idx +11] = (z*(u*u+v*v)-w*(x*u+y*v))*(1-cosaa)+(x*v-y*u)*sinaa+tz;
            out[matrix_idx +12] = 0;
            out[matrix_idx +13] = 0;
            out[matrix_idx +14] = 0;
            out[matrix_idx +15] = 1;
            //printf("getmotionmatrix:---\naxis:-----\n%.6f %.6f %.6f %.6f %.6f %.6f\n-----axis\nrspeed: %.6f\ntspeed: %.6f\ntype: %d\nmatrix:-----\n%.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n-----:matrix\n",x,y,z,u,v,w,tmp_rspeed,tmp_tspeed,type,out[matrix_idx+0],out[matrix_idx+1],out[matrix_idx+2],out[matrix_idx+3],out[matrix_idx+4],out[matrix_idx+5],out[matrix_idx+6],out[matrix_idx+7],out[matrix_idx+8],out[matrix_idx+9],out[matrix_idx+10],out[matrix_idx+11],out[matrix_idx+12],out[matrix_idx+13],out[matrix_idx+14],out[matrix_idx+15]);
        }
    }

}


void getmotionmatrixLauncher(int b, int n, const float *inp_axis_xyz, const float *inp_axis_uvw, const float *inp_rspeed, const float *inp_tspeed, float *out){
    getmotionmatrixKernel<<<32,512>>>(b,n,inp_axis_xyz,inp_axis_uvw,inp_rspeed,inp_tspeed,out);
}



__global__ void getmotionmatrixgradKernel(int b, int n, const float * __restrict__ inp_axis_xyz, const float * __restrict__ inp_axis_uvw, const float * __restrict__ inp_rspeed, const float * __restrict__ inp_tspeed, const float * __restrict__ out_g, float * __restrict__ inp_axis_xyz_g, float * __restrict__ inp_axis_uvw_g, float * __restrict__ inp_rspeed_g,float * __restrict__ inp_tspeed_g) {
    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<n;j+=blockDim.x){
            int tmp_idx = i*n+j;
            int axis_idx = i*n*3 + j*3;
            int matrix_idx = i*n*16 + j*16;
            float tmp_tspeed = inp_tspeed[tmp_idx];
            float tmp_rspeed = inp_rspeed[tmp_idx];
            float cosaa = cos(tmp_rspeed);
            float sinaa = sin(tmp_rspeed);
            float x = inp_axis_xyz[axis_idx + 0];
            float y = inp_axis_xyz[axis_idx + 1];
            float z = inp_axis_xyz[axis_idx + 2];
            float u = inp_axis_uvw[axis_idx + 0];
            float v = inp_axis_uvw[axis_idx + 1];
            float w = inp_axis_uvw[axis_idx + 2];

            float d1 = out_g[matrix_idx + 0];
            float d2 = out_g[matrix_idx + 1];
            float d3 = out_g[matrix_idx + 2];
            float d4 = out_g[matrix_idx + 3];
            float d5 = out_g[matrix_idx + 4];
            float d6 = out_g[matrix_idx + 5];
            float d7 = out_g[matrix_idx + 6];
            float d8 = out_g[matrix_idx + 7];
            float d9 = out_g[matrix_idx + 8];
            float d10= out_g[matrix_idx + 9];
            float d11= out_g[matrix_idx +10];
            float d12= out_g[matrix_idx +11];

            //out[matrix_idx + 0] = u*u+(v*v+w*w)*cosaa;
            float du1 = d1*2*u;
            float dv1 = d1*cosaa*2*v;
            float dw1 = d1*cosaa*2*w;
            float daa1 = d1*(v*v+w*w)*(-1)*sinaa;

            //out[matrix_idx + 1] = u*v*(1-cosaa)-w*sinaa;
            float du2 = d2*v*(1-cosaa);
            float dv2 = d2*u*(1-cosaa);
            float dw2 = d2*(-1)*sinaa;
            float daa2 =d2*(u*v*sinaa-w*cosaa);

            //out[matrix_idx + 2] = u*w*(1-cosaa)+v*sinaa;
            float du3 = d3*w*(1-cosaa);
            float dv3 = d3*sinaa;
            float dw3 = d3*u*(1-cosaa);
            float daa3 =d3*(u*w*sinaa+v*cosaa);

            //out[matrix_idx + 3] = (x*(v*v+w*w)-u*(y*v+z*w))*(1-cosaa)+(y*w-z*v)*sinaa;
            float dx4 = d4*(v*v+w*w)*(1-cosaa);
            float dy4 = d4*(((-1)*u*v)*(1-cosaa)+w*sinaa);
            float dz4 = d4*(((-1)*u*w)*(1-cosaa)-v*sinaa);
            float du4 = d4*(-1)*(y*v+z*w)*(1-cosaa);
            float dv4 = d4*((2*x*v-u*y)*(1-cosaa)-z*sinaa);
            float dw4 = d4*((2*x*w-u*z)*(1-cosaa)+y*sinaa);
            float daa4 =d4*((x*(v*v+w*w)-u*(y*v+z*w))*sinaa+(y*w-z*v)*cosaa);

            //out[matrix_idx + 4] = u*v*(1-cosaa)+w*sinaa;
            float du5 = d5*v*(1-cosaa);
            float dv5 = d5*u*(1-cosaa);
            float dw5 = d5*sinaa;
            float daa5 =d5*(u*v*sinaa+w*cosaa);

            //out[matrix_idx + 5] = v*v+(u*u+w*w)*cosaa;
            float du6 = d6*2*u*cosaa;
            float dv6 = d6*2*v;
            float dw6 = d6*2*w*cosaa;
            float daa6 =d6*(u*u+w*w)*(-1)*sinaa;

            //out[matrix_idx + 6] = v*w*(1-cosaa)-u*sinaa;
            float du7 = d7*(-1)*sinaa;
            float dv7 = d7*w*(1-cosaa);
            float dw7 = d7*v*(1-cosaa);
            float daa7 =d7*(v*w*sinaa-u*cosaa);

            //out[matrix_idx + 7] = (y*(u*u+w*w)-v*(x*u+z*w))*(1-cosaa)+(z*u-x*w)*sinaa;
            float dx8 = d8*((-1)*v*u*(1-cosaa)-w*sinaa);
            float dy8 = d8*(u*u+w*w)*(1-cosaa);
            float dz8 = d8*((-1)*v*w*(1-cosaa)+u*sinaa);
            float du8 = d8*((2*y*u-v*x)*(1-cosaa)+z*sinaa);
            float dv8 = d8*(-1)*(x*u+z*w)*(1-cosaa);
            float dw8 = d8*((2*y*w-v*z)*(1-cosaa)-x*sinaa);
            float daa8 =d8*((y*(u*u+w*w)-v*(x*u+z*w))*sinaa+(z*u-x*w)*cosaa);

            //out[matrix_idx + 8] = u*w*(1-cosaa)-v*sinaa;
            float du9 = d9*w*(1-cosaa);
            float dv9 = d9*(-1)*sinaa;
            float dw9 = d9*u*(1-cosaa);
            float daa9 =d9*(u*w*sinaa-v*cosaa);

            //out[matrix_idx + 9] = v*w*(1-cosaa)+u*sinaa;
            float du10= d10*sinaa;
            float dv10= d10*w*(1-cosaa);
            float dw10= d10*v*(1-cosaa);
            float daa10=d10*(v*w*sinaa+u*cosaa);

            //out[matrix_idx +10] = w*w+(u*u+v*v)*cosaa;
            float du11= d11*2*u*cosaa;
            float dv11= d11*2*v*cosaa;
            float dw11= d11*2*w;
            float daa11=d11*(u*u+v*v)*(-1)*sinaa;

            //out[matrix_idx +11] = (z*(u*u+v*v)-w*(x*u+y*v))*(1-cosaa)+(x*v-y*u)*sinaa;
            float dx12= d12*((-1)*w*u*(1-cosaa)+v*sinaa);
            float dy12= d12*((-1)*w*v*(1-cosaa)-u*sinaa);
            float dz12= d12*(u*u+v*v)*(1-cosaa);
            float du12= d12*((2*z*u-w*x)*(1-cosaa)-y*sinaa);
            float dv12= d12*((2*z*v-w*y)*(1-cosaa)+x*sinaa);
            float dw12= d12*(-1)*(x*u+y*v)*(1-cosaa);
            float daa12=d12*((z*(u*u+v*v)-w*(x*u+y*v))*sinaa+(x*v-y*u)*cosaa);

            float du = du1+du2+du3+du4+du5+du6+du7+du8+du9+du10+du11+du12; 
            float dv = dv1+dv2+dv3+dv4+dv5+dv6+dv7+dv8+dv9+dv10+dv11+dv12;
            float dw = dw1+dw2+dw3+dw4+dw5+dw6+dw7+dw8+dw9+dw10+dw11+dw12;
            float daa= daa1+daa2+daa3+daa4+daa5+daa6+daa7+daa8+daa9+daa10+daa11+daa12;
            float dx = dx4+dx8+dx12;
            float dy = dy4+dy8+dy12;
            float dz = dz4+dz8+dz12;
            float dtx = d4;
            float dty = d8;
            float dtz = d12;
            float dts = dtx * u + dty * v + dtz * w;

            du += dtx * tmp_tspeed;
            dv += dty * tmp_tspeed;
            dw += dtz * tmp_tspeed;

            atomicAdd(&inp_axis_xyz_g[axis_idx + 0], dx);
            atomicAdd(&inp_axis_xyz_g[axis_idx + 1], dy);
            atomicAdd(&inp_axis_xyz_g[axis_idx + 2], dz);
            atomicAdd(&inp_axis_uvw_g[axis_idx + 0], du);
            atomicAdd(&inp_axis_uvw_g[axis_idx + 1], dv);
            atomicAdd(&inp_axis_uvw_g[axis_idx + 2], dw);
            atomicAdd(&inp_rspeed_g[tmp_idx], daa);
            atomicAdd(&inp_tspeed_g[tmp_idx], dts);
            
            //printf("getmotionmatrixgrad:---\nmatrix grad: %.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\naxis:-----\n%.6f %.6f %.6f %.6f %.6f %.6f\n-----axis\nrspeed: %.6f\ntspeed: %.6f\ntype: %d\naxis grad:-----\n%.6f %.6f %.6f %.6f %.6f %.6f\n-----axis grad\nspeed grad-----\nrspeed grad:%.6f tspeed grad:%.6f\n-----:speed grad\n",d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,x,y,z,u,v,w,tmp_rspeed,tmp_tspeed,type,inp_axis_xyz_g[axis_idx+0],inp_axis_xyz_g[axis_idx+1],inp_axis_xyz_g[axis_idx+2],inp_axis_uvw_g[axis_idx+0],inp_axis_uvw_g[axis_idx+1],inp_axis_uvw_g[axis_idx+2],inp_rspeed_g[tmp_idx],inp_tspeed_g[tmp_idx]);

        }
    }

}





void getmotionmatrixgradLauncher(int b, int n, const float *inp_axis_xyz, const float *inp_axis_uvw, const float *inp_rspeed, const float *inp_tspeed, const float *out_g, float *inp_axis_xyz_g, float *inp_axis_uvw_g, float *inp_rspeed_g, float *inp_tspeed_g){
    getmotionmatrixgradKernel<<<32,512>>>(b,n,inp_axis_xyz,inp_axis_uvw,inp_rspeed,inp_tspeed,out_g,inp_axis_xyz_g,inp_axis_uvw_g,inp_rspeed_g,inp_tspeed_g);
}


__global__ void mulmotionmatrixKernel(int b, int m,int k, const float * __restrict__ inp_matrix, const float * __restrict__ inp_point, float * __restrict__ out) {
    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<m;j+=blockDim.x){
            int matrix_idx = i*m*16 + j*16;
            float A11 = inp_matrix[matrix_idx + 0];
            float A12 = inp_matrix[matrix_idx + 1];
            float A13 = inp_matrix[matrix_idx + 2];
            float A14 = inp_matrix[matrix_idx + 3];
            float A21 = inp_matrix[matrix_idx + 4];
            float A22 = inp_matrix[matrix_idx + 5];
            float A23 = inp_matrix[matrix_idx + 6];
            float A24 = inp_matrix[matrix_idx + 7];
            float A31 = inp_matrix[matrix_idx + 8];
            float A32 = inp_matrix[matrix_idx + 9];
            float A33 = inp_matrix[matrix_idx +10];
            float A34 = inp_matrix[matrix_idx +11];
            int point_idx = i*m*k*3 + j*k*3;
            for(int u = 0;u<k;u++){
                float x = inp_point[point_idx + u*3 + 0];
                float y = inp_point[point_idx + u*3 + 1];
                float z = inp_point[point_idx + u*3 + 2];
                out[point_idx + u*3 + 0] = A11*x + A12*y + A13*z + A14;
                out[point_idx + u*3 + 1] = A21*x + A22*y + A23*z + A24;
                out[point_idx + u*3 + 2] = A31*x + A32*y + A33*z + A34;
            }
        }
    }

}

void mulmotionmatrixLauncher(int b, int m,int k, const float *inp_matrix, const float *inp_point, float *out){
    mulmotionmatrixKernel<<<32,512>>>(b,m,k,inp_matrix,inp_point,out);
}


__global__ void mulmotionmatrixgradKernel(int b, int m,int k, const float * __restrict__ out_g, const float * __restrict__ inp_point, float * __restrict__ inp_g) {
    for(int i=blockIdx.x;i<b;i+=gridDim.x){
        for(int j=threadIdx.x;j<m;j+=blockDim.x){
            int point_idx = i*m*k*3 + j*k*3;
            int inp_g_idx = i*m*16 + j*16;
            float temp_d1 = 0;
            float temp_d2 = 0;
            float temp_d3 = 0;
            for(int u = 0;u<k;u++){
                float x = inp_point[point_idx + u*3 + 0];
                float y = inp_point[point_idx + u*3 + 1];
                float z = inp_point[point_idx + u*3 + 2];
                float d1 = out_g[point_idx + u*3 + 0];
                float d2 = out_g[point_idx + u*3 + 1];
                float d3 = out_g[point_idx + u*3 + 2];
                temp_d1 += d1;
                temp_d2 += d2;
                temp_d3 += d3;
                atomicAdd(&inp_g[inp_g_idx + 0],d1 * x );
                atomicAdd(&inp_g[inp_g_idx + 1],d1 * y );
                atomicAdd(&inp_g[inp_g_idx + 2],d1 * z );
                atomicAdd(&inp_g[inp_g_idx + 3],d1 );
                atomicAdd(&inp_g[inp_g_idx + 4],d2 * x );
                atomicAdd(&inp_g[inp_g_idx + 5],d2 * y );
                atomicAdd(&inp_g[inp_g_idx + 6],d2 * z );
                atomicAdd(&inp_g[inp_g_idx + 7],d2 );
                atomicAdd(&inp_g[inp_g_idx + 8],d3 * x );
                atomicAdd(&inp_g[inp_g_idx + 9],d3 * y );
                atomicAdd(&inp_g[inp_g_idx +10],d3 * z );
                atomicAdd(&inp_g[inp_g_idx +11],d3 );
            }
            //printf("mulmotionmatrixgrad:---\npoint grad:-----\n%.6f %.6f %.6f\n-----point grad\nmatrix grad:-----\n%.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n %.6f %.6f %.6f %.6f\n-----:matrix grad\n",temp_d1,temp_d2,temp_d3,inp_g[inp_g_idx+0],inp_g[inp_g_idx+1],inp_g[inp_g_idx+2],inp_g[inp_g_idx+3],inp_g[inp_g_idx+4],inp_g[inp_g_idx+5],inp_g[inp_g_idx+6],inp_g[inp_g_idx+7],inp_g[inp_g_idx+8],inp_g[inp_g_idx+9],inp_g[inp_g_idx+10],inp_g[inp_g_idx+11],inp_g[inp_g_idx+12],inp_g[inp_g_idx+13],inp_g[inp_g_idx+14],inp_g[inp_g_idx+15]);
        
        }
    }

}


void mulmotionmatrixgradLauncher(int b, int m, int k,const float *out_g, const float *inp_point, float *inp_g){
    mulmotionmatrixgradKernel<<<32,512>>>(b,m,k,out_g,inp_point,inp_g);
}

