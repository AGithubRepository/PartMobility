#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include<stdio.h>
using namespace tensorflow;


REGISTER_OP("GroupTrajectory")
    .Input("inp: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * numframe *  channels
        c->WithRank(c->input(0), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * msample * kgroup
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * msample * kgroup * numframe * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims2, 2), c->Dim(dims1, 2), c->Dim(dims1, 3)});
        c->set_output(0, output);
        return Status::OK();
    });


void grouptrajectoryLauncher(int b, int n, int c, int m,int t, int k, const float *inp, const int *idx, float *out);
class GroupTrajectoryGpuOp: public OpKernel{
    public:
        explicit GroupTrajectoryGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& inp_tensor=context->input(0);
            OP_REQUIRES(context, inp_tensor.dims()==4, errors::InvalidArgument("GroupTrajectory expects (batch_size * npoint * numframe *  channels) input shape"));
            int b = inp_tensor.shape().dim_size(0);
            int n = inp_tensor.shape().dim_size(1);
            int t = inp_tensor.shape().dim_size(2);
            int c = inp_tensor.shape().dim_size(3);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupTrajectory expects (batch_size * msample * kgroup) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int k = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,k,t,c}, &out_tensor));

            auto inp_flat = inp_tensor.flat<float>();
            const float *inp = &(inp_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            grouptrajectoryLauncher(b,n,c,m,t,k,inp,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupTrajectory").Device(DEVICE_GPU),GroupTrajectoryGpuOp);





REGISTER_OP("GroupTrajectoryGrad")
    .Input("inp: float32")
    .Input("idx: int32")
    .Input("out_g: float32")
    .Output("inp_g: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void grouptrajectorygradLauncher(int b, int n, int c, int m, int t, int k,const float *out_g, const int *idx, float *inp_g);
class GroupTrajectoryGradGpuOp: public OpKernel{
    public:
        explicit GroupTrajectoryGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& inp_tensor=context->input(0);
            OP_REQUIRES(context, inp_tensor.dims()==4, errors::InvalidArgument("GroupTrajectory expects (batch_size * npoint * numframe *  channels) input shape"));
            int b = inp_tensor.shape().dim_size(0);
            int n = inp_tensor.shape().dim_size(1);
            int t = inp_tensor.shape().dim_size(2);
            int c = inp_tensor.shape().dim_size(3);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupTrajectory expects (batch_size * msample * kgroup) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int k = idx_tensor.shape().dim_size(2);

            const Tensor& out_g_tensor=context->input(2);
            OP_REQUIRES(context,out_g_tensor.dims()==5 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==m && out_g_tensor.shape().dim_size(2)==k && out_g_tensor.shape().dim_size(3)==t && out_g_tensor.shape().dim_size(4)==c, errors::InvalidArgument("GroupTrajectoryGrad expects (batch_size * msample * kgroup * numframe * channels) grad_out shape"));

            Tensor * inp_g_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,t,c}, &inp_g_tensor));

            auto inp_flat = inp_tensor.flat<float>();
            const float *inp = &(inp_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_g_flat = out_g_tensor.flat<float>();
            const float *out_g = &(out_g_flat(0));
            auto inp_g_flat = inp_g_tensor->flat<float>();
            float *inp_g = &(inp_g_flat(0));
            cudaMemset(inp_g, 0, sizeof(float)*b*n*t*c);
            grouptrajectorygradLauncher(b,n,c,m,t,k,out_g,idx,inp_g);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupTrajectoryGrad").Device(DEVICE_GPU),GroupTrajectoryGradGpuOp);


REGISTER_OP("QueryBallTrajectory")
    .Attr("radius: float")
    .Attr("kgroup: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * msample * t * 3
        c->WithRank(c->input(1), 4, &dims2);
        int kgroup;
        TF_RETURN_IF_ERROR(c->GetAttr("kgroup", &kgroup));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), kgroup});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });


void queryballtrajectoryLauncher(int b, int n, int m,int t, float radius, int kgroup, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
class QueryBallTrajectoryGpuOp : public OpKernel {
    public:
        explicit QueryBallTrajectoryGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("kgroup", &kgroup_));
            OP_REQUIRES(context, kgroup_ > 0, errors::InvalidArgument("QueryBallPoint expects positive kgroup"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==4 && xyz1_tensor.shape().dim_size(3)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size * npoint * numframe *  3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);
            int t = xyz1_tensor.shape().dim_size(2);
            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==4 && xyz2_tensor.shape().dim_size(3)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size * msample * numframe *  3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,kgroup_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryballtrajectoryLauncher(b,n,m,t,radius_,kgroup_,xyz1,xyz2,idx,pts_cnt);
        }
    private:
        float radius_;
        int kgroup_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallTrajectory").Device(DEVICE_GPU), QueryBallTrajectoryGpuOp);



REGISTER_OP("SelectionSort")
    .Attr("k: int")
    .Input("dist: float32")
    .Output("outi: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });


void selectionsortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out);
class SelectionSortGpuOp : public OpKernel {
    public:
        explicit SelectionSortGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("SelectionSort expects positive k"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& dist_tensor = context->input(0);
            OP_REQUIRES(context, dist_tensor.dims()==3, errors::InvalidArgument("SelectionSort expects (b,m,n) dist shape."));
            int b = dist_tensor.shape().dim_size(0);
            int m = dist_tensor.shape().dim_size(1);
            int n = dist_tensor.shape().dim_size(2);

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,n}, &outi_tensor));
            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,n}, &out_tensor));

            auto dist_flat = dist_tensor.flat<float>();
            const float *dist = &(dist_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            selectionsortLauncher(b,n,m,k_,dist,outi,out);
        }
    private:
        int k_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionSort").Device(DEVICE_GPU), SelectionSortGpuOp);

