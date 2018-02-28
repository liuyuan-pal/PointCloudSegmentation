//
// Created by pal on 18-2-8.
//
#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace std;
using namespace tensorflow;

REGISTER_OP("NeighborScatter")
        .Attr("use_diff: bool")
        .Input("ifeats: float32")   // [pn,ifn]
        .Input("nidxs: int32")      // [csum]
        .Input("nidxs_lens: int32") // [pn]
        .Input("nidxs_bgs: int32")  // [pn]
        .Output("sfeats: float32")  // [csum,ofn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle ifeats_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_bgs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&ifeats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nidxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nidxs_lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&nidxs_bgs_shape));


            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(nidxs_shape,0),c->Dim(ifeats_shape,1)};
            c->set_output(0,c->MakeShape(dims));
            return Status::OK();
        });


class NeighborScatterGPUOp: public OpKernel
{
    bool use_diff;
public:
    explicit NeighborScatterGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("use_diff",&use_diff);
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& ifeats=context->input(0);     // [pn,ifn]
        const Tensor& nidxs=context->input(1);      // [csum]
        const Tensor& nidxs_lens=context->input(2); // [pn]
        const Tensor& nidxs_bgs=context->input(3);  // [pn]

        unsigned int pn=ifeats.shape().dim_size(0),
                    ifn=ifeats.shape().dim_size(1),
                    csum=nidxs.shape().dim_size(0);

        OP_REQUIRES(context,nidxs_lens.dim_size(0)==pn,errors::InvalidArgument("nidxs_lens dim 0"));
        OP_REQUIRES(context,nidxs_bgs.dim_size(0)==pn,errors::InvalidArgument("nidxs_bgs dim 0"));

        std::initializer_list<int64> dim_size={csum,ifn};
        TensorShape sfeats_shape(dim_size);
        Tensor* sfeats=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,sfeats_shape,&sfeats));

        auto ifeats_p=const_cast<float*>(ifeats.shaped<float,2>({pn,ifn}).data());
        auto nidxs_p= reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs.shaped<int,1>({csum}).data()));
        auto nidxs_lens_p= reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_lens.shaped<int,1>({pn}).data()));
        auto nidxs_bgs_p= reinterpret_cast<unsigned int*>(const_cast<int*>(nidxs_bgs.shaped<int,1>({pn}).data()));
        auto sfeats_p=sfeats->shaped<float,2>({csum,ifn}).data();
        neighborScatterGPU(ifeats_p,nidxs_p,nidxs_lens_p,nidxs_bgs_p,pn,ifn,sfeats_p,use_diff);
    }
};

REGISTER_KERNEL_BUILDER(Name("NeighborScatter").Device(DEVICE_GPU), NeighborScatterGPUOp);