//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("NeighborMaxFeatGather")
        .Input("feats: float32")     // [pn1,fd]
        .Input("vlens_lens: int32")  // [pn2]
        .Input("vlens_bgs: int32")   // [pn2]
        .Output("gfeats: float32")   // [pn2,fd]
        .Output("max_idxs: int32")   // [pn2,fd]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feats_shape;
            ::tensorflow::shape_inference::ShapeHandle vlens_lens_shape;
            ::tensorflow::shape_inference::ShapeHandle vlens_bgs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&vlens_lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&vlens_bgs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(vlens_lens_shape,0),c->Dim(feats_shape,1)};
            c->set_output(0,c->MakeShape(dims));
            c->set_output(1,c->MakeShape(dims));
            return Status::OK();
        });


class NeighborMaxFeatGatherGPUOp: public OpKernel
{
public:
    explicit NeighborMaxFeatGatherGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& feats=context->input(0);      // [pn1,fd]
        const Tensor& vlens_lens=context->input(1);// [pn2]
        const Tensor& vlens_bgs=context->input(2); // [pn2]

        unsigned int pn2=vlens_lens.dim_size(0),
                    fd=feats.shape().dim_size(1),
                    pn1=feats.shape().dim_size(0);

        OP_REQUIRES(context,vlens_bgs.dim_size(0)==pn2,errors::InvalidArgument("vlens_bgs dim 0"));

        std::initializer_list<int64> gfeats_max_dim={pn2,fd};
        Tensor *gfeats_max=NULL,*max_idxs=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(gfeats_max_dim),&gfeats_max));
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(gfeats_max_dim),&max_idxs));

        auto feats_p= const_cast<float*>(feats.shaped<float,2>({pn1,fd}).data());
        auto vlens_bgs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(vlens_bgs.shaped<int,1>({pn2}).data()));
        auto vlens_lens_p=reinterpret_cast<unsigned int*>(const_cast<int*>(vlens_lens.shaped<int,1>({pn2}).data()));
        auto gfeats_max_p=gfeats_max->shaped<float,2>({pn2,fd}).data();
        auto max_idxs_p=reinterpret_cast<unsigned int*>(max_idxs->shaped<int,2>({pn2,fd}).data());

        neighborMaxFeatGatherGPU(feats_p,vlens_lens_p,vlens_bgs_p,pn2,fd,gfeats_max_p,max_idxs_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("NeighborMaxFeatGather").Device(DEVICE_GPU), NeighborMaxFeatGatherGPUOp);