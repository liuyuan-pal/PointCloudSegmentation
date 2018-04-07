//
// Created by pal on 18-3-29
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>

using namespace tensorflow;

void computeDiffXYZImpl(
        float * xyzs,               // [pn1,3]
        float *cxyzs,               // [pn2,3]
        float *dxyzs,               // [pn1,3]
        int *cidxs,                 // [pn1]
        int pn1,
        int pn2
);

REGISTER_OP("ComputeDiffXyz")
    .Input("xyzs: float32")       // [pn1,3]
    .Input("cxyzs: float32")      // [pn2,3]
    .Input("cidxs: int32")        // [pn1]
    .Output("dxyzs: float32")     // [pn1,3]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle xyzs_shape;
        ::tensorflow::shape_inference::ShapeHandle cxyzs_shape;
        ::tensorflow::shape_inference::ShapeHandle cidxs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&xyzs_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&cxyzs_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&cidxs_shape));

        c->set_output(0,xyzs_shape);
        return Status::OK();
});


class ComputeDiffXyzGPUOp: public OpKernel
{
public:
    explicit ComputeDiffXyzGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& xyzs=context->input(0);       // [pn1,3]
        const Tensor& cxyzs=context->input(1);      // [pn2,3]
        const Tensor& cidxs=context->input(2);      // [pn1]

        unsigned int pn1=xyzs.dim_size(0),
                     pn2=cxyzs.dim_size(0);
        OP_REQUIRES(context,xyzs.dim_size(1)==3,errors::InvalidArgument("xyzs dim 1"));
        OP_REQUIRES(context,cxyzs.dim_size(1)==3,errors::InvalidArgument("cxyzs dim 1"));
        OP_REQUIRES(context,cidxs.dim_size(0)==pn1,errors::InvalidArgument("cidxs dim 0"));

        std::initializer_list<int64> dims={pn1,3};
        Tensor *dxyzs;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&dxyzs));

        auto xyzs_data=const_cast<float*>(xyzs.shaped<float,2>({pn1,3}).data());
        auto cxyzs_data=const_cast<float*>(cxyzs.shaped<float,2>({pn2,3}).data());
        auto cidxs_data=const_cast<int*>(cidxs.shaped<int,1>({pn1}).data());
        auto dxyzs_data=reinterpret_cast<float*>(dxyzs->shaped<float,2>({pn1,3}).data());

        computeDiffXYZImpl(xyzs_data,cxyzs_data,dxyzs_data,cidxs_data,pn1,pn2);
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputeDiffXyz").Device(DEVICE_GPU), ComputeDiffXyzGPUOp);