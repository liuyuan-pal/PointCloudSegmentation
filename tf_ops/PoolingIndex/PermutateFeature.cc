//
// Created by pal on 18-3-29.
//


#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>
#include <vector>

using namespace tensorflow;

void permutateFeature(
        float *feats,               // [pn,ps]
        int *idxs,                  // [pn]
        float *permutated_feats,    // [pn,ps]
        int pn,
        int ps
);

REGISTER_OP("PermutateFeatureBackward")
        .Input("feats: float32")                 // [pn,ps]
        .Input("idxs: int32")                    // [pn]
        .Output("permutated_feats: float32")     // [pn,ps]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feats_shape;
            ::tensorflow::shape_inference::ShapeHandle idxs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&idxs_shape));

            c->set_output(0,feats_shape);
            return Status::OK();
        });


class PermutateFeatureBackwardGPUOp: public OpKernel
{
public:
    explicit PermutateFeatureBackwardGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& feats=context->input(0);      // [pn,ps]
        const Tensor& idxs=context->input(1);       // [pn]

        unsigned int pn=feats.dim_size(0),
                     ps=feats.dim_size(1);
        OP_REQUIRES(context,idxs.dim_size(0)==pn,errors::InvalidArgument("idxs dim 0"));

        std::initializer_list<int64> dims={pn,ps};
        Tensor *permutated_feats;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&permutated_feats));
        auto feats_data= const_cast<float*>(feats.shaped<float,2>({pn,ps}).data());
        auto idxs_data= const_cast<int*>(idxs.shaped<int,1>({pn}).data());
        auto permutated_feats_data= const_cast<float*>(permutated_feats->shaped<float,2>({pn,ps}).data());
        permutateFeature(feats_data, idxs_data, permutated_feats_data, pn, ps);
    }
};

REGISTER_KERNEL_BUILDER(Name("PermutateFeatureBackward").Device(DEVICE_GPU), PermutateFeatureBackwardGPUOp);


REGISTER_OP("PermutateFeature")
        .Input("feats: float32")                 // [pn,ps]
        .Input("idxs: int32")                    // [pn]
        .Input("back_idxs: int32")               // [pn]
        .Output("permutated_feats: float32")     // [pn,ps]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feats_shape;
            ::tensorflow::shape_inference::ShapeHandle idxs_shape;
            ::tensorflow::shape_inference::ShapeHandle back_idxs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feats_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&idxs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&back_idxs_shape));

            c->set_output(0,feats_shape);
            return Status::OK();
        });


class PermutateFeatureGPUOp: public OpKernel
{
public:
    explicit PermutateFeatureGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& feats=context->input(0);      // [pn,ps]
        const Tensor& idxs=context->input(1);       // [pn]
        const Tensor& back_idxs=context->input(2);       // [pn]

        unsigned int pn=feats.dim_size(0),
                    ps=feats.dim_size(1);
        OP_REQUIRES(context,idxs.dim_size(0)==pn,errors::InvalidArgument("idxs dim 0"));
        OP_REQUIRES(context,back_idxs.dim_size(0)==pn,errors::InvalidArgument("back_idxs dim 0"));

        std::initializer_list<int64> dims={pn,ps};
        Tensor *permutated_feats;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&permutated_feats));
        auto feats_data= const_cast<float*>(feats.shaped<float,2>({pn,ps}).data());
        auto idxs_data= const_cast<int*>(idxs.shaped<int,1>({pn}).data());
        auto permutated_feats_data= const_cast<float*>(permutated_feats->shaped<float,2>({pn,ps}).data());
        permutateFeature(feats_data, idxs_data, permutated_feats_data, pn, ps);
    }
};

REGISTER_KERNEL_BUILDER(Name("PermutateFeature").Device(DEVICE_GPU), PermutateFeatureGPUOp);