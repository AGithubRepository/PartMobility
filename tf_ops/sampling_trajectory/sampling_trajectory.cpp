#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include<stdio.h>
using namespace tensorflow;
REGISTER_OP("GatherTrajecotry")
  .Input("inp: float32")
  .Input("idx: int32")
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * numframe * 3
    c->WithRank(c->input(0), 4, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * msample
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * msample * numframe * 3
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2), c->Dim(dims1, 3)});
    c->set_output(0, output);
    return Status::OK();
  });



void gathertrajctoryLauncher(int b,int n,int m,int t,const float * inp,const int *idx, float *out);
class GatherTrajecotryGpuOp: public OpKernel{
  public:
    explicit GatherTrajecotryGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==4 && inp_tensor.shape().dim_size(3)==3,errors::InvalidArgument("GatherPointTrajecotry expects (batch_size,npoint,numframe,3) inp shape"));
      auto inp_flat=inp_tensor.flat<float>();
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int t=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b && idx_tensor.shape().dim_size(1)<=n,errors::InvalidArgument("GatherPointTrajecotry expects (batch_size,msample) idx shape which msample<=npoint"));
      auto idx_flat=idx_tensor.flat<int>();
      int m=idx_tensor.shape().dim_size(1);
      const float * inp=&(inp_flat(0));
      const int * idx=&(idx_flat(0));
      Tensor * out_tensor = NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,t,3},&out_tensor));
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      gathertrajctoryLauncher(b,n,m,t,inp,idx,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherTrajecotry").Device(DEVICE_GPU), GatherTrajecotryGpuOp);


REGISTER_OP("GatherTrajectoryGrad")
  .Input("inp: float32")
  .Input("idx: int32")
  .Input("out_g: float32")
  .Output("inp_g: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


void gathertrajectorygradLauncher(int b,int n,int m,int t,const float * out_g,const int * idx,float * inp_g);
class GatherTrajectoryGradGpuOp: public OpKernel{
  public:
    explicit GatherTrajectoryGradGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==4 && inp_tensor.shape().dim_size(3)==3,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,npoint,numframe,3) inp"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int t=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,msample) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      const Tensor& out_g_tensor=context->input(2);
      OP_REQUIRES(context,out_g_tensor.dims()==4 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==m && out_g_tensor.shape().dim_size(3)==3,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,msample,numframe,3) out_g shape"));
      auto out_g_flat=out_g_tensor.flat<float>();
      const float * out_g=&(out_g_flat(0));
      Tensor * inp_g_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,t,3},&inp_g_tensor));
      auto inp_g_flat=inp_g_tensor->flat<float>();
      float * inp_g=&(inp_g_flat(0));
      cudaMemset(inp_g,0,sizeof(float)*b*n*t*3);
      gathertrajectorygradLauncher(b,n,m,t,out_g,idx,inp_g);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherTrajectoryGrad").Device(DEVICE_GPU),GatherTrajectoryGradGpuOp);








REGISTER_OP("FarthestPointSampleTrajecotry")
  .Attr("msample: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * numframe * 3
    c->WithRank(c->input(0), 4, &dims1);
    int msample;
    TF_RETURN_IF_ERROR(c->GetAttr("msample", &msample));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), msample});
    c->set_output(0, output);
    return Status::OK();
  });

void farthestpointsamplingtrajectoryLauncher(int b,int n,int m,int t,const float * inp,float * temp,int * out);
class FarthestPointSampleTrajectoryGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleTrajectoryGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("msample", &msample_));
                    OP_REQUIRES(context, msample_ > 0, errors::InvalidArgument("FarthestPointSample expects positive msample"));
                }
    void Compute(OpKernelContext * context)override{
      //get input
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==4 && inp_tensor.shape().dim_size(3)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,npoint,numframe,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int t=inp_tensor.shape().dim_size(2);
      OP_REQUIRES(context,msample_<=n,errors::InvalidArgument("FarthestPointSample expects sampling_point <= n"));
      auto inp_flat=inp_tensor.flat<float>();
      //get input address
      const float * inp=&(inp_flat(0));
      //allocate output
      Tensor * out_tensor = NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,msample_},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;


      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingtrajectoryLauncher(b,n,msample_,t,inp,temp,out);
    }
    private:
        int msample_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSampleTrajecotry").Device(DEVICE_GPU),FarthestPointSampleTrajectoryGpuOp);
