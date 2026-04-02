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

  /// @brief Execute a backward pass to compute gradients.
  /// @param grad_output The backward gradient from the VGG loss.
  void backward(const Tensor &grad_output);

  /// @brief Get the generated output tensor.
  const Tensor &get_output() const;

  // Memory chunks to back the learned parameter weights.
  struct ParamDescriptor {
    dnnl::memory mem;
    std::vector<int> shape;
    dnnl::memory diff_mem; // Gradient for optimization
  };

  /// @brief Get the parameters of the network.
  std::unordered_map<std::string, ParamDescriptor> &get_parameters() {
    return parameters_;
  }
  const std::unordered_map<std::string, ParamDescriptor> &
  get_parameters() const {
    return parameters_;
  }

private:
  // Helper properties
  dnnl::engine engine_;
  dnnl::stream stream_;
  int input_h_;
  int input_w_;

  // Primitives and pipeline execution
  std::vector<std::function<void()>> pipeline_;
  std::vector<std::function<void()>> backward_pipeline_;

  std::unordered_map<std::string, ParamDescriptor> parameters_;

  std::unique_ptr<Tensor> output_tensor_;

  struct MemPair {
    dnnl::memory fwd;
    dnnl::memory bwd;
  };

  MemPair input_mempair_;
  MemPair output_mempair_;

  // Construction helpers
  MemPair create_conv(const std::string &name, int ic, int oc, int kernel,
                      int stride, int padding, MemPair src_mem);
  MemPair create_norm(const std::string &name, int channels, MemPair src_mem);
  MemPair create_relu(MemPair src_mem);
  MemPair create_resample(MemPair src_mem, float scale);
  MemPair create_add(MemPair src0_mem, MemPair src1_mem);
  MemPair create_resblock(const std::string &name, int channels,
                          MemPair src_mem);
  void init_weights(); // Kaiming uniform init for conv, scale=1 for norm
};

} // namespace stylor

#endif // STYLOR_TRANSFORM_NETWORK_HPP
