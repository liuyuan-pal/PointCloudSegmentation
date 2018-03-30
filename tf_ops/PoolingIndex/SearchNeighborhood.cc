//
// Created by pal on 18-3-29
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>

using namespace tensorflow;

int searchNeighborhoodCountImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
);

void searchNeighborhoodImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [pn]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
);


REGISTER_OP("SearchNeighborhoodBruteForce")
        .Input("xyzs: float32")                 // [pn]
        .Attr("squared_nn_size: float")         // [pn]
        .Output("idxs: int32")     // [en]
        .Output("lens: int32")     // [pn]
        .Output("begs: int32")     // [pn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle xyzs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&xyzs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(xyzs_shape,0)};
            std::initializer_list<shape_inference::DimensionOrConstant> unknow_dims=
                    {c->UnknownDim()};
            c->set_output(0,c->MakeShape(unknow_dims));
            c->set_output(1,c->MakeShape(dims));
            c->set_output(2,c->MakeShape(dims));
            return Status::OK();
        });

class SearchNeighborhoodBruteForceGPUOp: public OpKernel
{
    float squared_nn_size;
public:
    explicit SearchNeighborhoodBruteForceGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("squared_nn_size",&squared_nn_size);
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& xyzs=context->input(0);       // [pn1,3]

        unsigned int pn=xyzs.dim_size(0);
        OP_REQUIRES(context,xyzs.dim_size(1)==3,errors::InvalidArgument("xyzs dim 1"));

        std::initializer_list<int64> dims={pn};
        Tensor *idxs,*lens,*begs;
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims),&lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims),&begs));
        auto xyzs_data=const_cast<float*>(xyzs.shaped<float,2>({pn,3}).data());
        auto lens_data=const_cast<int*>(lens->shaped<int,1>({pn}).data());
        auto begs_data=const_cast<int*>(begs->shaped<int,1>({pn}).data());

        int en=searchNeighborhoodCountImpl(xyzs_data,lens_data,begs_data,squared_nn_size,pn);

        std::initializer_list<int64> dims2={en};
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims2),&idxs));
        auto idxs_data=const_cast<int*>(idxs->shaped<int,1>({en}).data());
        searchNeighborhoodImpl(xyzs_data,idxs_data,begs_data,squared_nn_size,pn);
    }
};

REGISTER_KERNEL_BUILDER(Name("SearchNeighborhoodBruteForce").Device(DEVICE_GPU), SearchNeighborhoodBruteForceGPUOp);


REGISTER_OP("SearchNeighborhoodWithBins")
        .Input("xyzs: float32")                 // [pn,3]
        .Input("bin_idxs: int32")               // [pn,3]
        .Attr("squared_nn_size: float")
        .Attr("bin_len: float")
        .Output("idxs: int32")     // [en]
        .Output("lens: int32")     // [pn]
        .Output("begs: int32")     // [pn]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle xyzs_shape;
            ::tensorflow::shape_inference::ShapeHandle bin_idxs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&xyzs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&bin_idxs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(xyzs_shape,0)};
            std::initializer_list<shape_inference::DimensionOrConstant> unknow_dims=
                    {c->UnknownDim()};
            c->set_output(0,c->MakeShape(unknow_dims));
            c->set_output(1,c->MakeShape(dims));
            c->set_output(2,c->MakeShape(dims));
            return Status::OK();
        });


int searchNeighborhoodCountWithBinsImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        int *bin_idxs,              // [pn,3]
        int bin_thresh,
        float squared_nn_size,
        int pn
);

void searchNeighborhoodWithBinsImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [pn]
        int *begs,                  // [pn]
        int *bin_idxs,              // [pn,3]
        int bin_thresh,
        float squared_nn_size,
        int pn
);

class SearchNeighborhoodWithBinsGPUOp: public OpKernel
{
    float squared_nn_size;
    float bin_len;
public:
    explicit SearchNeighborhoodWithBinsGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("squared_nn_size",&squared_nn_size);
        context->GetAttr("bin_len",&bin_len);
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& xyzs=context->input(0);       // [pn1,3]
        const Tensor& bin_idxs=context->input(1);       // [pn1,3]

        unsigned int pn=xyzs.dim_size(0);
        OP_REQUIRES(context,xyzs.dim_size(1)==3,errors::InvalidArgument("xyzs dim 1"));
        OP_REQUIRES(context,bin_idxs.dim_size(0)==pn,errors::InvalidArgument("bin_idxs dim 0"));
        OP_REQUIRES(context,bin_idxs.dim_size(1)==3,errors::InvalidArgument("bin_idxs dim 1"));

        std::initializer_list<int64> dims={pn};
        Tensor *idxs,*lens,*begs;
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims),&lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims),&begs));
        auto xyzs_data=const_cast<float*>(xyzs.shaped<float,2>({pn,3}).data());
        auto bin_idxs_data=const_cast<int*>(bin_idxs.shaped<int,2>({pn,3}).data());
        auto lens_data=const_cast<int*>(lens->shaped<int,1>({pn}).data());
        auto begs_data=const_cast<int*>(begs->shaped<int,1>({pn}).data());

        int bin_thresh=ceil(sqrt(squared_nn_size)/bin_len);

        int en=searchNeighborhoodCountWithBinsImpl(xyzs_data,lens_data,begs_data,bin_idxs_data,bin_thresh,squared_nn_size,pn);

        std::initializer_list<int64> dims2={en};
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims2),&idxs));
        auto idxs_data=const_cast<int*>(idxs->shaped<int,1>({en}).data());
        searchNeighborhoodWithBinsImpl(xyzs_data,idxs_data,begs_data,bin_idxs_data,bin_thresh,squared_nn_size,pn);
    }
};

REGISTER_KERNEL_BUILDER(Name("SearchNeighborhoodWithBins").Device(DEVICE_GPU), SearchNeighborhoodWithBinsGPUOp);