#include "stylor/tensor.hpp"

namespace stylor {

Tensor::Tensor(std::vector<dnnl::memory::dim> dims)
    : dims_(std::move(dims)), engine_(dnnl::engine::kind::cpu, 0),
      desc_(dims_, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::nchw),
      memory_(desc_, engine_) {}

Tensor::Tensor(std::vector<dnnl::memory::dim> dims, const dnnl::engine &engine)
    : dims_(std::move(dims)), engine_(engine),
      desc_(dims_, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::nchw),
      memory_(desc_, engine_) {}

float *Tensor::get_data() {
  return static_cast<float *>(memory_.get_data_handle());
}

const float *Tensor::get_data() const {
  return static_cast<const float *>(memory_.get_data_handle());
}

std::vector<dnnl::memory::dim> Tensor::get_dims() const { return dims_; }

dnnl::memory &Tensor::get_memory() { return memory_; }

const dnnl::memory &Tensor::get_memory() const { return memory_; }

} // namespace stylor
