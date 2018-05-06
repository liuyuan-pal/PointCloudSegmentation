//
// Created by pal on 18-3-29
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>
//#include <log4cpp/Category.hh>
//#include <log4cpp/FileAppender.hh>
//#include <log4cpp/BasicLayout.hh>
//#include <log4cpp/Priority.hh>

using namespace tensorflow;

void computeVoxelLabelImpl(
        int *point_label,
        int *vlens,
        int *vbegs,
        int *voxel_label,
        int vn,
        int cn
);

REGISTER_OP("ComputeVoxelLabel")
    .Attr("class_num: int")
    .Input("point_label: int32")
    .Input("vlens: int32")
    .Input("vbegs: int32")
    .Output("voxel_label:int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle point_label_shape;
        ::tensorflow::shape_inference::ShapeHandle vlens_shape;
        ::tensorflow::shape_inference::ShapeHandle vbegs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0),1,&point_label_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&vlens_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&vbegs_shape));

        std::initializer_list<shape_inference::DimensionOrConstant> dims={c->Dim(vlens_shape,0)};
        c->set_output(0,c->MakeShape(dims));
        return Status::OK();
});


class ComputeVoxelLabelGPUOp: public OpKernel
{
    int class_num;
public:
    explicit ComputeVoxelLabelGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("class_num",&class_num);
    }
    void Compute(OpKernelContext* context) override
    {
        const Tensor& point_label=context->input(0);
        const Tensor& vlens=context->input(1);
        const Tensor& vbegs=context->input(2);

        unsigned int pn=point_label.dim_size(0),
                     vn=vlens.dim_size(0);
        OP_REQUIRES(context,vbegs.dim_size(0)==vn,errors::InvalidArgument("vbegs dim 0"));


        std::initializer_list<int64> dims={vn};
        Tensor *voxel_label;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&voxel_label));

        auto point_label_data=const_cast<int*>(point_label.shaped<int,1>({pn}).data());
        auto vlens_data=const_cast<int*>(vlens->shaped<int,1>({vn}).data());
        auto vbegs_data=const_cast<int*>(vbegs->shaped<int,1>({vn}).data());
        auto voxel_label_data=voxel_label->shaped<int,1>({vn}).data();

        computeVoxelLabelImpl(point_label_data,vlens_data,vbegs_data,voxel_label_data,vn,class_num);
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputeVoxelLabel").Device(DEVICE_GPU), ComputeVoxelLabelGPUOp);