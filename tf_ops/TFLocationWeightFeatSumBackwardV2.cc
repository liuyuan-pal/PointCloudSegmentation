//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("LocationWeightFeatSumBackwardV2")
        .Input("lw: float32")       // [csum,m]
        .Input("feats: float32")    // [csum,m,ofn]
        .Input("dtfeats_sum: float32")// [pn,ofn]
        .Input("cidxs: int32")
        .Output("dlw: float32")     // [csum,m]
        .Output("dfeats: float32")  // [csum,m,ofn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle lw_shape;
            ::tensorflow::shape_inference::ShapeHandle feats_shape;
            ::tensorflow::shape_inference::ShapeHandle dtfeats_sum_shape;
            ::tensorflow::shape_inference::ShapeHandle cidxs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&lw_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),3,&feats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),3,&dtfeats_sum_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&cidxs_shape));

            c->set_output(0,lw_shape);
            c->set_output(1,feats_shape);
            return Status::OK();
        });


class LocationWeightFeatSumBackwardGPUV2Op: public OpKernel
{
public:
    explicit LocationWeightFeatSumBackwardGPUV2Op(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& lw=context->input(0);         // [csum,m]
        const Tensor& feats=context->input(1);      // [csum,m,ofn]
        const Tensor& dtfeats_sum=context->input(2);// [pn,m,ofn]
        const Tensor& cidxs=context->input(3);      // [csum]

        unsigned int pn=dtfeats_sum.shape().dim_size(0),
                m=feats.shape().dim_size(1),
                ofn=feats.shape().dim_size(2),
                csum=lw.shape().dim_size(0);

        OP_REQUIRES(context,lw.dim_size(1)==m,errors::InvalidArgument("lw dim 1"));
        OP_REQUIRES(context,dtfeats_sum.dim_size(1)==m,errors::InvalidArgument("dtfeats_sum dim 0"));
        OP_REQUIRES(context,dtfeats_sum.dim_size(2)==ofn,errors::InvalidArgument("dtfeats_sum dim 0"));
        OP_REQUIRES(context,feats.dim_size(0)==csum,errors::InvalidArgument("feats dim 0"));
        OP_REQUIRES(context,cidxs.dim_size(0)==csum,errors::InvalidArgument("cidxs dim 0"));

        Tensor *dfeats=NULL,*dlw=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(feats.shape()),&dfeats));
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(lw.shape()),&dlw));

        auto feats_p= const_cast<float*>(feats.shaped<float,3>({csum,m,ofn}).data());
        auto lw_p= const_cast<float*>(lw.shaped<float,2>({csum,m}).data());
        auto dtfeats_sum_p=const_cast<float*>(dtfeats_sum.shaped<float,3>({pn,m,ofn}).data());
        auto cidxs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(cidxs.shaped<int,1>({csum}).data()));
        auto dfeats_p=dfeats->shaped<float,3>({csum,m,ofn}).data();
        auto dlw_p=dlw->shaped<float,2>({csum,m}).data();

        locWFeatSumBackwardGPUV2(feats_p,lw_p,dtfeats_sum_p,cidxs_p,pn,ofn,m,csum,dfeats_p,dlw_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("LocationWeightFeatSumBackwardV2").Device(DEVICE_GPU), LocationWeightFeatSumBackwardGPUV2Op);