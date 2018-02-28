//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("LocationWeightSumBackward")
        .Input("lw: float32")       // [csum,m]
        .Input("dlw_sum: float32")  // [pn,m]
        .Input("nidxs_lens: int32") // [pn]
        .Input("nidxs_bgs: int32")  // [pn]
        .Output("dlw: float32")     // [csum,m]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle lw_shape;
            ::tensorflow::shape_inference::ShapeHandle dlw_sum_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_bgs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&lw_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&dlw_sum_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nidxs_lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&nidxs_bgs_shape));

            c->set_output(0,lw_shape);
            return Status::OK();
        });


class LocationWeightSumBackwardGPUOp: public OpKernel
{
public:
    explicit LocationWeightSumBackwardGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& lw=context->input(0);         // [csum,m]
        const Tensor& dlw_sum=context->input(1);     // [pn,m]
        const Tensor& nidxs_lens=context->input(2); // [pn]
        const Tensor& nidxs_bgs=context->input(3);  // [pn]

        unsigned int pn=nidxs_lens.shape().dim_size(0),
                m=lw.shape().dim_size(1),
                csum=lw.shape().dim_size(0);

        OP_REQUIRES(context,nidxs_bgs.dim_size(0)==pn,errors::InvalidArgument("nidxs_bgs dim 0"));
        OP_REQUIRES(context,dlw_sum.dim_size(0)==pn,errors::InvalidArgument("dlw_sum dim 0"));
        OP_REQUIRES(context,dlw_sum.dim_size(1)==m,errors::InvalidArgument("dlw_sum dim 1"));

        Tensor *dlw=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(lw.shape()),&dlw));

        auto dlw_sum_p= const_cast<float*>(dlw_sum.shaped<float,2>({pn,m}).data());
        auto nidxs_lens_p=reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_lens.shaped<int,1>({pn}).data()));
        auto nidxs_bgs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_bgs.shaped<int,1>({pn}).data()));
        auto dlw_p= const_cast<float*>(dlw->shaped<float,2>({csum,m}).data());

        locWSumBackwardGPU(dlw_sum_p,nidxs_lens_p,nidxs_bgs_p,csum,pn,m,dlw_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("LocationWeightSumBackward").Device(DEVICE_GPU), LocationWeightSumBackwardGPUOp);