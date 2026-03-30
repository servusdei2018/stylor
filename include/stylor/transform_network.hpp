#ifndef STYLOR_TRANSFORM_NETWORK_HPP
#define STYLOR_TRANSFORM_NETWORK_HPP

#include "stylor/tensor.hpp"
#include <dnnl.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace stylor {

/// @brief Feed-forward Image Transform Network for Fast Neural Style Transfer.
///
/// Implements the architecture from Johnson et al. (2016) with Instance
/// Normalization.
class TransformNetwork {
public:
  /// @brief Construct the network graph.
  /// @param engine oneDNN engine.
  /// @param input_h Expected input height.
  /// @param input_w Expected input width.
  explicit TransformNetwork(const dnnl::engine &engine, int input_h = 256,
                            int input_w = 256);

  /// @brief Load weights from a safetensors file.
  void load_weights(const std::string &path);

  /// @brief Save weights to a safetensors file.
  void save_weights(const std::string &path) const;

  /// @brief Execute a forward pass.
  /// @param input Pre-processed Float32 NCHW tensor {1, 3, H, W}.
  void forward(const Tensor &input);

  /// @brief Get the generated output tensor.
  const Tensor &get_output() const;

private:
  struct ConvPrimitive {
    std::string name;
    dnnl::memory weights_mem;
    dnnl::memory bias_mem;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::convolution_forward conv;
  };

  struct GroupNormPrimitive {
    std::string name;
    dnnl::memory weights_mem; // Scale and shift stored together in oneDNN: 2x C
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::group_normalization_forward gn;
  };

  struct ReluPrimitive {
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::eltwise_forward relu;
  };

  struct ResamplingPrimitive {
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::resampling_forward resample;
  };

  struct AddPrimitive {
    dnnl::memory src0_mem; // Shortcut
    dnnl::memory src1_mem; // Residual
    dnnl::memory dst_mem;
    dnnl::sum add;
  };

  // Helper properties
  dnnl::engine engine_;
  dnnl::stream stream_;
  int input_h_;
  int input_w_;

  // Primitives and pipeline execution
  std::vector<std::function<void()>> pipeline_;

  // Memory chunks to back the learned parameter weights.
  struct ParamDescriptor {
    dnnl::memory mem;
    std::vector<int> shape;
  };
  std::unordered_map<std::string, ParamDescriptor> parameters_;

  std::unique_ptr<Tensor> output_tensor_;

  // Construction helpers
  dnnl::memory create_conv(const std::string &name, int ic, int oc, int kernel,
                           int stride, int padding, dnnl::memory src_mem);
  dnnl::memory create_norm(const std::string &name, int channels,
                           dnnl::memory src_mem);
  dnnl::memory create_relu(dnnl::memory src_mem);
  dnnl::memory create_resample(dnnl::memory src_mem, float scale);
  dnnl::memory create_add(dnnl::memory src0_mem, dnnl::memory src1_mem);
  dnnl::memory create_resblock(const std::string &name, int channels,
                               dnnl::memory src_mem);
};

} // namespace stylor

#endif // STYLOR_TRANSFORM_NETWORK_HPP
