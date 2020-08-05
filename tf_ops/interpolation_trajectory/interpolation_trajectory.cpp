#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include<stdio.h>
using namespace tensorflow;

REGISTER_OP("ThreeNN")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("dist: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * numframe *  3
        c->WithRank(c->input(0), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * msample * numframe *  3
        c->WithRank(c->input(1), 4, &dims2);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), 3});
        c->set_output(0, output);
        c->set_output(1, output);
        return Status::OK();
    });


void threennLauncher(int b,int n,int m,int t,const float *xyz1, const float *xyz2, float *dist, int *idx);
class ThreeNNOp : public OpKernel {
    public:
        explicit ThreeNNOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==4 && xyz1_tensor.shape().dim_size(3)==3, errors::InvalidArgument("ThreeNN expects (batch_size * npoint * numframe *  3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);
            int t = xyz1_tensor.shape().dim_size(2);
            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==4 && xyz2_tensor.shape().dim_size(3)==3, errors::InvalidArgument("ThreeNN expects (batch_size * msample * numframe *  3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *dist_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,3}, &dist_tensor));
            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,3}, &idx_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto dist_flat = dist_tensor->flat<float>();
            float *dist = &(dist_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            threennLauncher(b,n,m,t,xyz1,xyz2,dist,idx);
        }
};
REGISTER_KERNEL_BUILDER(Name("ThreeNN").Device(DEVICE_GPU), ThreeNNOp);


REGISTER_OP("ThreeInterpolate")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // (b,m,t,c)
        c->WithRank(c->input(0), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // (b,n,3)
        c->WithRank(c->input(1), 3, &dims2);
        // (b,n,t,c)
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2), c->Dim(dims1, 3)});
        c->set_output(0, output);
        return Status::OK();
    });

void threeinterpolateLauncher(int b,int n,int m,int t,int c,const float *points, const int *idx, const float *weight, float *out);
class ThreeInterpolateOp: public OpKernel{
    public:
        explicit ThreeInterpolateOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==4, errors::InvalidArgument("ThreeInterpolate expects (b,m,t,c) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int m = points_tensor.shape().dim_size(1);
            int t = points_tensor.shape().dim_size(2);
            int c = points_tensor.shape().dim_size(3);
            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b && idx_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeInterpolate expects (b,n,3) idx shape"));
            int n = idx_tensor.shape().dim_size(1);
            const Tensor& weight_tensor=context->input(2);
            OP_REQUIRES(context,weight_tensor.dims()==3 && weight_tensor.shape().dim_size(0)==b && weight_tensor.shape().dim_size(1)==n && weight_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeInterpolate expects (b,n,3) weight shape"));

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,t,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto weight_flat = weight_tensor.flat<float>();
            const float *weight = &(weight_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            threeinterpolateLauncher(b,n,m,t,c,points,idx,weight,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("ThreeInterpolate").Device(DEVICE_GPU),ThreeInterpolateOp);



REGISTER_OP("ThreeInterpolateGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void threeinterpolategradLauncher(int b, int n, int m, int t, int c, const float *grad_out, const int *idx, const float *weight, float *grad_points);
class ThreeInterpolateGradOp: public OpKernel{
    public:
        explicit ThreeInterpolateGradOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==4, errors::InvalidArgument("ThreeInterpolateGrad expects (b,m,t,c) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int m = points_tensor.shape().dim_size(1);
            int t = points_tensor.shape().dim_size(2);
            int c = points_tensor.shape().dim_size(3);
            
            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("ThreeInterpolateGrad expects (b,n,3) idx shape"));
            int n = idx_tensor.shape().dim_size(1);
            const Tensor& weight_tensor=context->input(2);
            OP_REQUIRES(context,weight_tensor.dims()==3 && weight_tensor.shape().dim_size(0)==b && weight_tensor.shape().dim_size(1)==n && weight_tensor.shape().dim_size(2)==3, errors::InvalidArgument("ThreeInterpolateGrad expects (b,n,3) weight shape"));

            const Tensor& grad_out_tensor=context->input(3);
            OP_REQUIRES(context,grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==n && grad_out_tensor.shape().dim_size(3)==c, errors::InvalidArgument("ThreeInterpolateGrad expects (b,n,t,c) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,t,c}, &grad_points_tensor));
            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto weight_flat = weight_tensor.flat<float>();
            const float *weight = &(weight_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            cudaMemset(grad_points, 0, sizeof(float)*b*m*t*c);
            threeinterpolategradLauncher(b,n,m,t,c,grad_out,idx,weight,grad_points);
        }
};
REGISTER_KERNEL_BUILDER(Name("ThreeInterpolateGrad").Device(DEVICE_GPU),ThreeInterpolateGradOp);


