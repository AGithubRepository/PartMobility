#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include<stdio.h>
using namespace tensorflow;
using namespace std;


REGISTER_OP("GetMotionMatrix")
    .Input("inp_axis_xyz: float32")
    .Input("inp_axis_uvw: float32")
    .Input("inp_rspeed: float32")
    .Input("inp_tspeed: float32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3 (x,y,z)
        c->WithRank(c->input(0), 3, &dims1);
        // batch_size * npoint * 4 * 4
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), 4, 4});
        c->set_output(0, output);
        return Status::OK();
    });



void getmotionmatrixLauncher(int b, int n, const float *inp_axis_xyz,const float *inp_axis_uvw, const float *inp_rspeed, const float *inp_tspeed, float *out);
class GetMotionMatrixGpuOp: public OpKernel{
    public:
        explicit GetMotionMatrixGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& inp_axis_xyz_tensor=context->input(0);
            OP_REQUIRES(context, inp_axis_xyz_tensor.dims()==3 && inp_axis_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("GetMotionMatrix expects inp_axis_xyz (batch_size * npoint * 3) input shape"));
            const Tensor& inp_axis_uvw_tensor=context->input(1);
            OP_REQUIRES(context, inp_axis_uvw_tensor.dims()==3 && inp_axis_uvw_tensor.shape().dim_size(2)==3, errors::InvalidArgument("GetMotionMatrix expects inp_axis_uvw (batch_size * npoint * 3) input shape"));
            int b = inp_axis_xyz_tensor.shape().dim_size(0);
            int n = inp_axis_xyz_tensor.shape().dim_size(1);

            const Tensor& inp_rspeed_tensor=context->input(2);
            OP_REQUIRES(context, inp_rspeed_tensor.dims()==3 && inp_rspeed_tensor.shape().dim_size(0)==b && inp_rspeed_tensor.shape().dim_size(1)==n && inp_rspeed_tensor.shape().dim_size(2)==1, errors::InvalidArgument("GetMotionMatrix expects inp_rspeed (batch_size * npoint * 1) input shape"));

            const Tensor& inp_tspeed_tensor=context->input(3);
            OP_REQUIRES(context, inp_tspeed_tensor.dims()==3 && inp_tspeed_tensor.shape().dim_size(0)==b && inp_tspeed_tensor.shape().dim_size(1)==n && inp_tspeed_tensor.shape().dim_size(2)==1, errors::InvalidArgument("GetMotionMatrix expects inp_tspeed (batch_size * npoint * 1) input shape"));


            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,4,4}, &out_tensor));

            auto inp_axis_xyz_flat = inp_axis_xyz_tensor.flat<float>();
            const float *inp_axis_xyz = &(inp_axis_xyz_flat(0));
            auto inp_axis_uvw_flat = inp_axis_uvw_tensor.flat<float>();
            const float *inp_axis_uvw = &(inp_axis_uvw_flat(0));
            auto inp_rspeed_flat = inp_rspeed_tensor.flat<float>();
            const float *inp_rspeed = &(inp_rspeed_flat(0));
            auto inp_tspeed_flat = inp_tspeed_tensor.flat<float>();
            const float *inp_tspeed = &(inp_tspeed_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            getmotionmatrixLauncher(b,n,inp_axis_xyz,inp_axis_uvw,inp_rspeed,inp_tspeed,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GetMotionMatrix").Device(DEVICE_GPU),GetMotionMatrixGpuOp);


REGISTER_OP("GetMotionMatrixGrad")
    .Input("inp_axis_xyz: float32")
    .Input("inp_axis_uvw: float32")
    .Input("inp_rspeed: float32")
    .Input("inp_tspeed: float32")
    .Input("out_g: float32")
    .Output("inp_axis_axis_g: float32")
    .Output("inp_axis_uvw_g: float32")
    .Output("inp_rspeed_g: float32")
    .Output("inp_tspeed_g: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        c->set_output(2, c->input(2));
        c->set_output(3, c->input(3));
        return Status::OK();
    });

void getmotionmatrixgradLauncher(int b, int n, const float *inp_axis_xyz, const float *inp_axis_uvw, const float *inp_rspeed, const float *inp_tspeed, const float *out_g, float *inp_axis_xyz_g, float *inp_axis_uvw_g, float *inp_rspeed_g, float *inp_tspeed_g);
class GetMotionMatrixGradGpuOp: public OpKernel{
    public:
        explicit GetMotionMatrixGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& inp_axis_xyz_tensor=context->input(0);
            OP_REQUIRES(context, inp_axis_xyz_tensor.dims()==3 && inp_axis_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("GetMotionMatrix expects inp_axis_xyz (batch_size * npoint * 3) input shape"));
            const Tensor& inp_axis_uvw_tensor=context->input(1);
            OP_REQUIRES(context, inp_axis_uvw_tensor.dims()==3 && inp_axis_uvw_tensor.shape().dim_size(2)==3, errors::InvalidArgument("GetMotionMatrix expects inp_axis_uvw (batch_size * npoint * 3) input shape"));
            int b = inp_axis_xyz_tensor.shape().dim_size(0);
            int n = inp_axis_xyz_tensor.shape().dim_size(1);

            const Tensor& inp_rspeed_tensor=context->input(2);
            OP_REQUIRES(context, inp_rspeed_tensor.dims()==3 && inp_rspeed_tensor.shape().dim_size(0)==b && inp_rspeed_tensor.shape().dim_size(1)==n && inp_rspeed_tensor.shape().dim_size(2)==1, errors::InvalidArgument("GetMotionMatrix expects inp_rspeed (batch_size * npoint * 1) input shape"));

            const Tensor& inp_tspeed_tensor=context->input(3);
            OP_REQUIRES(context, inp_tspeed_tensor.dims()==3 && inp_tspeed_tensor.shape().dim_size(0)==b && inp_tspeed_tensor.shape().dim_size(1)==n && inp_tspeed_tensor.shape().dim_size(2)==1, errors::InvalidArgument("GetMotionMatrix expects inp_tspeed (batch_size * npoint * 1) input shape"));


            const Tensor& out_g_tensor=context->input(4);
            OP_REQUIRES(context,out_g_tensor.dims()==4 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==n && out_g_tensor.shape().dim_size(2)==4 && out_g_tensor.shape().dim_size(3)==4, errors::InvalidArgument("GetMotionMatrix expects (batch_size * msample * 4 * 4) grad_out shape"));

            Tensor * inp_axis_xyz_g_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,3}, &inp_axis_xyz_g_tensor));
            Tensor * inp_axis_uvw_g_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1,TensorShape{b,n,3}, &inp_axis_uvw_g_tensor));
            Tensor * inp_rspeed_g_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2,TensorShape{b,n,1}, &inp_rspeed_g_tensor));
            Tensor * inp_tspeed_g_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3,TensorShape{b,n,1}, &inp_tspeed_g_tensor));

            auto inp_axis_xyz_flat = inp_axis_xyz_tensor.flat<float>();
            const float *inp_axis_xyz = &(inp_axis_xyz_flat(0));
            auto inp_axis_uvw_flat = inp_axis_uvw_tensor.flat<float>();
            const float *inp_axis_uvw = &(inp_axis_uvw_flat(0));
            auto inp_rspeed_flat = inp_rspeed_tensor.flat<float>();
            const float *inp_rspeed = &(inp_rspeed_flat(0));
            auto inp_tspeed_flat = inp_tspeed_tensor.flat<float>();
            const float *inp_tspeed = &(inp_tspeed_flat(0));
            auto out_g_flat = out_g_tensor.flat<float>();
            const float *out_g = &(out_g_flat(0));

            auto inp_axis_xyz_g_flat = inp_axis_xyz_g_tensor->flat<float>();
            float *inp_axis_xyz_g = &(inp_axis_xyz_g_flat(0));
            auto inp_axis_uvw_g_flat = inp_axis_uvw_g_tensor->flat<float>();
            float *inp_axis_uvw_g = &(inp_axis_uvw_g_flat(0));
            auto inp_rspeed_g_flat = inp_rspeed_g_tensor->flat<float>();
            float *inp_rspeed_g = &(inp_rspeed_g_flat(0));
            auto inp_tspeed_g_flat = inp_tspeed_g_tensor->flat<float>();
            float *inp_tspeed_g = &(inp_tspeed_g_flat(0));
            cudaMemset(inp_axis_xyz_g, 0, sizeof(float)*b*n*3);
            cudaMemset(inp_axis_uvw_g, 0, sizeof(float)*b*n*3);
            cudaMemset(inp_rspeed_g, 0, sizeof(float)*b*n*1);
            cudaMemset(inp_tspeed_g, 0, sizeof(float)*b*n*1);
            getmotionmatrixgradLauncher(b,n,inp_axis_xyz,inp_axis_uvw,inp_rspeed,inp_tspeed,out_g,inp_axis_xyz_g,inp_axis_uvw_g,inp_rspeed_g,inp_tspeed_g);
        }
};
REGISTER_KERNEL_BUILDER(Name("GetMotionMatrixGrad").Device(DEVICE_GPU),GetMotionMatrixGradGpuOp);





REGISTER_OP("MulMotionMatrix")
    .Input("inp_matrix: float32")
    .Input("inp_point: float32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * msample * 4 * 4
        c->WithRank(c->input(0), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * msample * kgroup * 3
        c->WithRank(c->input(1), 4, &dims2);
        // batch_size * msample * kgroup * 3
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims2, 2), 3});
        c->set_output(0, output);
        return Status::OK();
    });



void mulmotionmatrixLauncher(int b, int m,int k, const float *inp_matrix, const float *inp_point, float *out);
class MulMotionMatrixGpuOp: public OpKernel{
    public:
        explicit MulMotionMatrixGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& inp_matrix_tensor=context->input(0);
            OP_REQUIRES(context, inp_matrix_tensor.dims()==4 && inp_matrix_tensor.shape().dim_size(2)==4 && inp_matrix_tensor.shape().dim_size(3)==4, errors::InvalidArgument("MulMotionMatrix expects inp_matrix (batch_size * msample * 4 * 4) input shape"));
            int b = inp_matrix_tensor.shape().dim_size(0);
            int m = inp_matrix_tensor.shape().dim_size(1);

            const Tensor& inp_point_tensor=context->input(1);
            OP_REQUIRES(context, inp_point_tensor.dims()==4 && inp_point_tensor.shape().dim_size(0)==b && inp_point_tensor.shape().dim_size(1)==m && inp_point_tensor.shape().dim_size(3)==3, errors::InvalidArgument("MulMotionMatrix expects inp_point (batch_size * msample * kgroup * 3) input shape"));
            int k = inp_point_tensor.shape().dim_size(2);
            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,k,3}, &out_tensor));
            
            auto inp_matrix_flat = inp_matrix_tensor.flat<float>();
            const float *inp_matrix = &(inp_matrix_flat(0));
            auto inp_point_flat = inp_point_tensor.flat<float>();
            const float *inp_point = &(inp_point_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            mulmotionmatrixLauncher(b,m,k,inp_matrix,inp_point,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("MulMotionMatrix").Device(DEVICE_GPU),MulMotionMatrixGpuOp);

REGISTER_OP("MulMotionMatrixGrad")
    .Input("inp_matrix: float32")
    .Input("inp_point: float32")
    .Input("out_g: float32")
    .Output("inp_g: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void mulmotionmatrixgradLauncher(int b, int m, int k,const float *out_g, const float *inp_point, float *inp_g);
class MulMotionMatrixGradGpuOp: public OpKernel{
    public:
        explicit MulMotionMatrixGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& inp_matrix_tensor=context->input(0);
            OP_REQUIRES(context, inp_matrix_tensor.dims()==4 && inp_matrix_tensor.shape().dim_size(2)==4 && inp_matrix_tensor.shape().dim_size(3)==4, errors::InvalidArgument("MulMotionMatrix expects inp_matrix (batch_size * msample * 4 * 4) input shape"));
            int b = inp_matrix_tensor.shape().dim_size(0);
            int m = inp_matrix_tensor.shape().dim_size(1);

            const Tensor& inp_point_tensor=context->input(1);
            OP_REQUIRES(context, inp_point_tensor.dims()==4 && inp_point_tensor.shape().dim_size(0)==b && inp_point_tensor.shape().dim_size(1)==m && inp_point_tensor.shape().dim_size(3)==3, errors::InvalidArgument("MulMotionMatrix expects inp_point (batch_size * msample * kgroup * 3) input shape"));
            int k = inp_point_tensor.shape().dim_size(2);

            const Tensor& out_g_tensor=context->input(2);
            OP_REQUIRES(context,out_g_tensor.dims()==4 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==m && out_g_tensor.shape().dim_size(2)==k && out_g_tensor.shape().dim_size(3)==3, errors::InvalidArgument("MulMotionMatrix expects (batch_size * msample * kgroup * 3) grad_out shape"));

            Tensor * inp_g_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,4,4}, &inp_g_tensor));

            auto inp_matrix_flat = inp_matrix_tensor.flat<float>();
            const float *inp_matrix = &(inp_matrix_flat(0));
            auto inp_point_flat = inp_point_tensor.flat<float>();
            const float *inp_point = &(inp_point_flat(0));
            auto out_g_flat = out_g_tensor.flat<float>();
            const float *out_g = &(out_g_flat(0));
            auto inp_g_flat = inp_g_tensor->flat<float>();
            float *inp_g = &(inp_g_flat(0));
            cudaMemset(inp_g, 0, sizeof(float)*b*m*4*4);
            mulmotionmatrixgradLauncher(b,m,k,out_g,inp_point,inp_g);
        }
};
REGISTER_KERNEL_BUILDER(Name("MulMotionMatrixGrad").Device(DEVICE_GPU),MulMotionMatrixGradGpuOp);
