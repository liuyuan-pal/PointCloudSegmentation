//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;
using shape_inference::DimensionHandle;
REGISTER_OP("NeighborConcatNonCenterGather")
        .Input("sfeats: float32")   // [csum-pn,ifn*2]
        .Input("nidxs: int32")      // [csum]
        .Input("nidxs_lens: int32") // [pn]
        .Input("nidxs_bgs: int32")  // [pn]
        .Output("ifeats: float32")  // [pn,ifn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle sfeats_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_bgs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&sfeats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nidxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nidxs_lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&nidxs_bgs_shape));

            std::vector<DimensionHandle> output_dims(2);
            if(c->ValueKnown(c->Dim(nidxs_lens_shape,0)))
                output_dims[0]=c->Dim(nidxs_lens_shape,0);
            else
                output_dims[0]=c->UnknownDim();

            if(c->ValueKnown(c->Dim(sfeats_shape,1)))
            {
                auto val=c->Value(c->Dim(sfeats_shape,1))/2;
                output_dims[1]=c->MakeDim(val);
            }
            else output_dims[1]=c->UnknownDim();

            c->set_output(0,c->MakeShape(output_dims));
            return Status::OK();
        });


class NeighborConcatNonCenterGatherGPUOp: public OpKernel
{
    bool use_diff;
public:
    explicit NeighborConcatNonCenterGatherGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& sfeats=context->input(0);     // [csum-pn,2*ifn]
        const Tensor& nidxs=context->input(1);      // [csum]
        const Tensor& nidxs_lens=context->input(2); // [pn]
        const Tensor& nidxs_bgs=context->input(3);  // [pn]

        unsigned int pn=nidxs_lens.shape().dim_size(0),
                    ifn=sfeats.shape().dim_size(1)/2,
                    csum=nidxs.shape().dim_size(0);

        OP_REQUIRES(context,sfeats.dim_size(0)==(csum-pn),errors::InvalidArgument("sfeats dim 0"));
        OP_REQUIRES(context,nidxs_bgs.dim_size(0)==pn,errors::InvalidArgument("nidxs_bgs dim 0"));

        std::initializer_list<int64> ifeats_dim={pn,ifn};
        Tensor *ifeats=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(ifeats_dim),&ifeats));

        auto sfeats_p= const_cast<float*>(sfeats.shaped<float,2>({csum-pn,ifn*2}).data());
        auto nidxs_p= reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs.shaped<int,1>({csum}).data()));
        auto nidxs_lens_p=reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_lens.shaped<int,1>({pn}).data()));
        auto nidxs_bgs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_bgs.shaped<int,1>({pn}).data()));
        auto ifeats_p=ifeats->shaped<float,2>({pn,ifn}).data();
        concatNonCenterFeatGatherGPU(sfeats_p,nidxs_p,nidxs_lens_p,nidxs_bgs_p,pn,ifn,ifeats_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("NeighborConcatNonCenterGather").Device(DEVICE_GPU), NeighborConcatNonCenterGatherGPUOp);