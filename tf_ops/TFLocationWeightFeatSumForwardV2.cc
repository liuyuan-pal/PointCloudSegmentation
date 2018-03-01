//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("LocationWeightFeatSumV2")
        .Input("lw: float32")       // [csum,m]
        .Input("feats: float32")    // [csum,m,ofn]
        .Input("cidxs: int32")      // [csum]
        .Input("nidxs_lens: int32") // [pn]
        .Output("pfeats: float32")  // [pn,m,ofn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle lw_shape;
            ::tensorflow::shape_inference::ShapeHandle feats_shape;
            ::tensorflow::shape_inference::ShapeHandle cidxs_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&lw_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),3,&feats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&cidxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&nidxs_lens_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(nidxs_lens_shape,0),c->Dim(feats_shape,1),c->Dim(feats_shape,2)};
            c->set_output(0,c->MakeShape(dims));
            return Status::OK();
        });


class LocationWeightFeatSumGPUV2Op: public OpKernel
{
public:
    explicit LocationWeightFeatSumGPUV2Op(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& lw=context->input(0);         // [csum,m]
        const Tensor& feats=context->input(1);      // [csum,m,ofn]
        const Tensor& cidxs=context->input(2);      // [csum]
        const Tensor& nidixs_lens=context->input(3);// [pn]

        unsigned int pn=nidixs_lens.dim_size(0),
                    m=feats.shape().dim_size(1),
                    ofn=feats.shape().dim_size(2),
                    csum=lw.shape().dim_size(0);

        OP_REQUIRES(context,lw.dim_size(1)==m,errors::InvalidArgument("lw dim 1"));
        OP_REQUIRES(context,cidxs.dim_size(0)==csum,errors::InvalidArgument("cidxs dim 0"));
        OP_REQUIRES(context,feats.dim_size(0)==csum,errors::InvalidArgument("feats dim 0"));

        std::initializer_list<int64> tfeats_sum_dim={pn,m,ofn};
        Tensor *tfeats_sum=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(tfeats_sum_dim),&tfeats_sum));

        auto feats_p= const_cast<float*>(feats.shaped<float,3>({csum,m,ofn}).data());
        auto lw_p= const_cast<float*>(lw.shaped<float,2>({csum,m}).data());
        auto cidxs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(cidxs.shaped<int,1>({csum}).data()));
        auto tfeats_sum_p=tfeats_sum->shaped<float,3>({pn,m,ofn}).data();

        locWFeatSumForwardGPUV2(feats_p,lw_p,cidxs_p,pn,ofn,m,csum,tfeats_sum_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("LocationWeightFeatSumV2").Device(DEVICE_GPU), LocationWeightFeatSumGPUV2Op);