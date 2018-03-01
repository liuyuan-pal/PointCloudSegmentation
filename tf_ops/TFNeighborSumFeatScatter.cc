//
// Created by pal on 18-2-8.
//

#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("NeighborSumFeatScatter")
        .Input("gfeats_sum: float32")    // [pn,fd]
        .Input("cidxs: int32")           // [csum]
        .Output("sfeats: float32")       // [csum,fd]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle gfeats_sum_shape;
            ::tensorflow::shape_inference::ShapeHandle cidxs_shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&gfeats_sum_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&cidxs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(cidxs_shape,0),c->Dim(gfeats_sum_shape,1)};
            c->set_output(0,c->MakeShape(dims));
            return Status::OK();
        });


class NeighborSumFeatScatterGPUOp: public OpKernel
{
public:
    explicit NeighborSumFeatScatterGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& gfeats_sum=context->input(0); // [pn,fd]
        const Tensor& cidxs=context->input(1);      // [csum]

        unsigned int pn=gfeats_sum.dim_size(0),
                    fd=gfeats_sum.dim_size(1),
                    csum=cidxs.dim_size(0);

        std::initializer_list<int64> sfeats_dims={csum,fd};
        Tensor *sfeats=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(sfeats_dims),&sfeats));

        auto gfeats_sum_p= const_cast<float*>(gfeats_sum.shaped<float,2>({pn,fd}).data());
        auto cidxs_p=reinterpret_cast<unsigned int*>(const_cast<int*>(cidxs.shaped<int,1>({csum}).data()));
        auto sfeats_p=sfeats->shaped<float,2>({csum,fd}).data();

        neighborSumFeatScatterGPU(gfeats_sum_p,cidxs_p,pn,fd,csum,sfeats_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("NeighborSumFeatScatter").Device(DEVICE_GPU), NeighborSumFeatScatterGPUOp);