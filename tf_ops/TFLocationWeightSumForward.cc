//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("LocationWeightSum")
        .Input("lw: float32")       // [csum,m]
        .Input("nidxs_lens: int32") // [pn]
        .Input("nidxs_bgs: int32")  // [pn]
        .Output("lw_sum: float32")  // [pn,m]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle lw_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_bgs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&lw_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nidxs_lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nidxs_bgs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(nidxs_lens_shape,0),c->Dim(lw_shape,1)};
            c->set_output(0,c->MakeShape(dims));
            return Status::OK();
        });


class LocationWeightSumGPUOp: public OpKernel
{
public:
    explicit LocationWeightSumGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& lw=context->input(0);         // [csum,m]
        const Tensor& nidxs_lens=context->input(1); // [pn]
        const Tensor& nidxs_bgs=context->input(2);  // [pn]

        unsigned int pn=nidxs_lens.shape().dim_size(0),
                m=lw.shape().dim_size(1),
                csum=lw.shape().dim_size(0);

        OP_REQUIRES(context,nidxs_bgs.dim_size(0)==pn,errors::InvalidArgument("nidxs_bgs dim 0"));

        std::initializer_list<int64> lw_sum_dim={pn,m};
        Tensor *lw_sum=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(lw_sum_dim),&lw_sum));

        auto lw_p= const_cast<float*>(lw.shaped<float,2>({csum,m}).data());
        auto nidxs_lens_p=reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_lens.shaped<int,1>({pn}).data()));
        auto nidxs_bgs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_bgs.shaped<int,1>({pn}).data()));
        auto lw_sum_p=lw_sum->shaped<float,2>({pn,m}).data();

        locWSumForwardGPU(lw_p,nidxs_lens_p,nidxs_bgs_p,pn,m,lw_sum_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("LocationWeightSum").Device(DEVICE_GPU), LocationWeightSumGPUOp);