/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "group_point.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {

            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));
            auto points_flat = points_tensor.flat<T>();
            const T *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<T>();
            T *out = &(out_flat(0));

            GroupPointFunctor<Device, T>()(
                context->eigen_device<Device>(),
                b, n, c, m, nsample, points, idx, out);
        }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct GroupPointFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("GroupPoint").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      GroupPointGpuOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
