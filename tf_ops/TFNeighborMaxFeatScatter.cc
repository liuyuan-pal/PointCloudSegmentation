//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("NeighborMaxFeatScatter")
        .Input("gfeats: float32")        // [pn2,fd]
        .Input("ifeats: float32")        // [pn1,fd]
        .Input("max_idxs: int32")        // [pn2,fd]
        .Input("vlens_bg: int32")        // [pn2]
        .Output("sfeats: float32")       // [pn1,fd]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle gfeats_shape;
            ::tensorflow::shape_inference::ShapeHandle ifeats_shape;
            ::tensorflow::shape_inference::ShapeHandle max_idxs_shape;
            ::tensorflow::shape_inference::ShapeHandle vlens_bg_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&gfeats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&ifeats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),2,&max_idxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&vlens_bg_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(ifeats_shape,0),c->Dim(gfeats_shape,1)};
            c->set_output(0,c->MakeShape(dims));
            return Status::OK();
        });


class NeighborMaxFeatScatterGPUOp: public OpKernel
{
public:
    explicit NeighborMaxFeatScatterGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& gfeats=context->input(0);     // [pn2,fd]
        const Tensor& ifeats=context->input(1);     // [pn1,fd]
        const Tensor& max_idxs=context->input(2);   // [pn2,fd]
        const Tensor& vlens_bgs=context->input(3);  // [pn2]

        unsigned int pn2=gfeats.dim_size(0),
                    fd=gfeats.dim_size(1),
                    pn1=ifeats.dim_size(0);


        OP_REQUIRES(context,ifeats.dim_size(1)==fd,errors::InvalidArgument("ifeats dim 1"));
        OP_REQUIRES(context,max_idxs.dim_size(0)==pn2,errors::InvalidArgument("max_idxs dim 0"));
        OP_REQUIRES(context,max_idxs.dim_size(1)==fd,errors::InvalidArgument("max_idxs dim 1"));
        OP_REQUIRES(context,vlens_bgs.dim_size(0)==pn2,errors::InvalidArgument("vlens_bgs dim 0"));

        std::initializer_list<int64> sfeats_dims={pn1,fd};
        Tensor *sfeats=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(sfeats_dims),&sfeats));

        auto gfeats_p=const_cast<float*>(gfeats.shaped<float,2>({pn2,fd}).data());
        auto max_idxs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(max_idxs.shaped<int,2>({pn2,fd}).data()));
        auto vlens_bgs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(vlens_bgs.shaped<int,1>({pn2}).data()));
        auto sfeats_p=sfeats->shaped<float,2>({pn1,fd}).data();

        neighborMaxFeatScatterGPU(gfeats_p,max_idxs_p,vlens_bgs_p,pn1,pn2,fd,sfeats_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("NeighborMaxFeatScatter").Device(DEVICE_GPU), NeighborMaxFeatScatterGPUOp);