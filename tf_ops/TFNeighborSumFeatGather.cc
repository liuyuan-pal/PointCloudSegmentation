//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("NeighborSumFeatGather")
        .Input("feats: float32")   // [csum,fd]
        .Input("cidxs: int32")      // [csum]
        .Input("nidxs_lens: int32") // [pn]
        .Output("gfeats: float32")  // [pn,fd]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feats_shape;
            ::tensorflow::shape_inference::ShapeHandle cidxs_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&cidxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nidxs_lens_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(nidxs_lens_shape,0),c->Dim(feats_shape,1)};
            c->set_output(0,c->MakeShape(dims));
            return Status::OK();
        });


class NeighborSumFeatGatherGPUOp: public OpKernel
{
public:
    explicit NeighborSumFeatGatherGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& feats=context->input(0);      // [csum,fd]
        const Tensor& cidxs=context->input(1);      // [csum]
        const Tensor& nidixs_lens=context->input(2);// [pn]

        unsigned int pn=nidixs_lens.dim_size(0),
                    fd=feats.shape().dim_size(1),
                    csum=feats.shape().dim_size(0);

        OP_REQUIRES(context,cidxs.dim_size(0)==csum,errors::InvalidArgument("cidxs dim 0"));

        std::initializer_list<int64> gfeats_sum_dim={pn,fd};
        Tensor *gfeats_sum=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(gfeats_sum_dim),&gfeats_sum));

        auto feats_p= const_cast<float*>(feats.shaped<float,2>({csum,fd}).data());
        auto cidxs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(cidxs.shaped<int,1>({csum}).data()));
        auto gfeats_sum_p=gfeats_sum->shaped<float,2>({pn,fd}).data();

        neighborSumFeatGatherGPU(feats_p,cidxs_p,pn,fd,csum,gfeats_sum_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("NeighborSumFeatGather").Device(DEVICE_GPU), NeighborSumFeatGatherGPUOp);