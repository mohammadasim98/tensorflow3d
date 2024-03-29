// kernel_example.h
#ifndef KERNEL_GROUP_H_
#define KERNEL_GROUP_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct GroupPointFunctor {
  void operator()(const Device& d, int b, int n, int c, int m, int nsample, const T *points, const int *idx, T *out);
};

template <typename Device, typename T>
struct GroupPointGradFunctor {
  void operator()(const Device& d, int b, int n, int c, int m, int nsample, const T *grad_out, const int *idx, T *grad_points);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
