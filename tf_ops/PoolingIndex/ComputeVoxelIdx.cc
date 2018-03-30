//
// Created by pal on 18-3-29
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>

using namespace tensorflow;

void computeVoxelIdxImpl(
        float* pts,
        unsigned int* voxel_idxs,
        int pt_num,
        int pt_stride,
        float voxel_len,
        float min_x,
        float min_y,
        float min_z
);

REGISTER_OP("ComputeVoxelIndex")
    .Attr("voxel_len: float")
    .Attr("block_size: float")
    .Input("pts:         float32")     // [pn,3]
    .Output("voxel_idxs: int32")       // [pn,3]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle pts_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&pts_shape));

        std::initializer_list<shape_inference::DimensionOrConstant> dims=
                {c->Dim(pts_shape,0),3};
        c->set_output(0,c->MakeShape(dims));
        return Status::OK();
});


class ComputeVoxelIndexGPUOp: public OpKernel
{
    float voxel_len;
    float block_size;
public:
    explicit ComputeVoxelIndexGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("voxel_len",&voxel_len);
        context->GetAttr("block_size",&block_size);
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& pts=context->input(0);      // [pn1,fd]

        unsigned int pn=pts.dim_size(0);
        OP_REQUIRES(context,pts.dim_size(1)==3,errors::InvalidArgument("pts dim 1"));

        std::initializer_list<int64> dims={pn,3};
        Tensor *voxel_idxs;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&voxel_idxs));

        auto pts_data= const_cast<float*>(pts.shaped<float,2>({pn,3}).data());
        auto voxel_idxs_data=reinterpret_cast<unsigned int*>(voxel_idxs->shaped<int,2>({pn,3}).data());
        computeVoxelIdxImpl(pts_data,voxel_idxs_data,pn,3,voxel_len,-block_size/2.f,-block_size/2.f,0.f);
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputeVoxelIndex").Device(DEVICE_GPU), ComputeVoxelIndexGPUOp);