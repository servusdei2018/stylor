#ifndef STYLOR_TENSOR_HPP
#define STYLOR_TENSOR_HPP

#include <dnnl.hpp>
#include <vector>

namespace stylor {

/// @brief A wrapper around oneDNN memory objects for easy manipulation.
class Tensor {
public:
  /// @brief Construct a tensor with specific dimension sizes (e.g., {N, C, H,
  /// W}).
  /// @param dims The sizes for each dimension.
  Tensor(std::vector<dnnl::memory::dim> dims);

  /// @brief Get the raw data pointer for read/write operations.
  /// @return A pointer to the underlying float data array.
  float *get_data();

  /// @brief Get the constant raw data pointer.
  /// @return A constant pointer to the underlying float data array.
  const float *get_data() const;

  /// @brief Get the dimensions of the tensor.
  /// @return The dimensions vector.
  std::vector<dnnl::memory::dim> get_dims() const;

  /// @brief Get the underlying oneDNN memory object.
  /// @return The oneDNN memory object.
  dnnl::memory &get_memory();

  /// @brief Get the constant underlying oneDNN memory object.
  /// @return The constant oneDNN memory object.
  const dnnl::memory &get_memory() const;

private:
  std::vector<dnnl::memory::dim> dims_;
  dnnl::engine engine_;
  dnnl::memory::desc desc_;
  dnnl::memory memory_;
};

} // namespace stylor

#endif // STYLOR_TENSOR_HPP
