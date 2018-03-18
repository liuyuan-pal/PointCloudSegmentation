//
// Created by pal on 18-2-8.
//
#include "TFNeighborKernel.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace std;
using namespace tensorflow;
using shape_inference::DimensionHandle;

REGISTER_OP("EliminateCenter")
        .Input("inidxs: int32")      // [csum]
        .Input("inidxs_lens: int32") // [pn]
        .Input("inidxs_bgs: int32")  // [pn]
        .Output("onidxs: int32")      // [csum-pn]
        .Output("onidxs_lens: int32") // [pn]
        .Output("onidxs_bgs: int32")  // [pn]
        .Output("ocidxs: int32")      // [csum-pn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle nidxs_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_lens_shape;
            ::tensorflow::shape_inference::ShapeHandle nidxs_bgs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),1,&nidxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nidxs_lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nidxs_bgs_shape));

            std::vector<DimensionHandle> output_dims(1);
            if(c->ValueKnown(c->Dim(nidxs_shape,0))&&
            c->ValueKnown(c->Dim(nidxs_bgs_shape,0)))
            {
                auto val=c->Value(c->Dim(nidxs_shape,0))-c->Value(c->Dim(nidxs_bgs_shape,0));
                output_dims[0]=c->MakeDim(val);
            }
            else output_dims[0]=c->UnknownDim();

            c->set_output(0,c->MakeShape(output_dims));
            c->set_output(1,nidxs_lens_shape);
            c->set_output(2,nidxs_bgs_shape);
            c->set_output(3,c->MakeShape(output_dims));
            return Status::OK();
        });


class EliminateCenterGPUOp: public OpKernel
{
public:
    explicit EliminateCenterGPUOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& inidxs=context->input(0);      // [csum]
        const Tensor& inidxs_lens=context->input(1); // [pn]
        const Tensor& inidxs_bgs=context->input(2);  // [pn]

        unsigned int pn=inidxs_lens.shape().dim_size(0),
                     csum=inidxs.shape().dim_size(0);

        OP_REQUIRES(context,inidxs_bgs.dim_size(0)==pn,errors::InvalidArgument("nidxs_bgs dim 0"));

        std::initializer_list<int64> dim_size1={csum-pn};
        std::initializer_list<int64> dim_size2={pn};
        Tensor* onidxs=NULL,*onidxs_lens=NULL,*onidxs_bgs=NULL,*ocidxs=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dim_size1),&onidxs));
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dim_size2),&onidxs_lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dim_size2),&onidxs_bgs));
        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape(dim_size1),&ocidxs));

        auto inidxs_p= reinterpret_cast<unsigned int*>(const_cast<int*>(inidxs.shaped<int,1>({csum}).data()));
        auto inidxs_lens_p= reinterpret_cast<unsigned int*>(const_cast<int*>(inidxs_lens.shaped<int,1>({pn}).data()));
        auto inidxs_bgs_p= reinterpret_cast<unsigned int*>(const_cast<int*>(inidxs_bgs.shaped<int,1>({pn}).data()));

        auto onidxs_p=reinterpret_cast<unsigned int*>(onidxs->shaped<int,1>({csum-pn}).data());
        auto onidxs_lens_p=reinterpret_cast<unsigned int*>(onidxs_lens->shaped<int,1>({pn}).data());
        auto onidxs_bgs_p=reinterpret_cast<unsigned int*>(onidxs_bgs->shaped<int,1>({pn}).data());
        auto ocidxs_p=reinterpret_cast<unsigned int*>(ocidxs->shaped<int,1>({csum-pn}).data());
        eliminateCenterGPU(inidxs_p,inidxs_lens_p,inidxs_bgs_p,pn,onidxs_p,onidxs_lens_p,onidxs_bgs_p,ocidxs_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("EliminateCenter").Device(DEVICE_GPU), EliminateCenterGPUOp);