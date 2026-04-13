#ifndef STYLOR_VGG_HPP
#define STYLOR_VGG_HPP

#include "stylor/tensor.hpp"
#include <dnnl.hpp>
#include <string>
#include <unordered_map>

namespace stylor {

/// @brief Named feature-map extraction points within VGG-19.
///
/// Names follow the standard relu block convention.  Style loss typically uses
/// relu1_1–relu5_1; content loss uses relu4_2.
enum class VggLayer {
  relu1_1, ///< After conv1_1 + ReLU  (64 channels, full resolution)
  relu2_1, ///< After conv2_1 + ReLU  (128 channels, 1/2 resolution)
  relu3_1, ///< After conv3_1 + ReLU  (256 channels, 1/4 resolution)
  relu4_1, ///< After conv4_1 + ReLU  (512 channels, 1/8 resolution)
  relu4_2, ///< After conv4_2 + ReLU  (512 channels, 1/8 resolution) — content
  relu5_1, ///< After conv5_1 + ReLU  (512 channels, 1/16 resolution)
};

/// @brief VGG-19 feature extractor backed by oneDNN primitives.
///
/// Constructs a truncated 10-convolution VGG-19 graph (up to relu5_1) on the
/// given engine during construction. Weights must be loaded via load_weights()
/// before calling
/// forward().  After a successful forward pass, intermediate activations for
/// the six named layers are available through get_feature_map().
///
/// Thread safety: none, use one instance per thread.
class Vgg19 {
public:
  /// @brief Constructs and compiles the VGG-19 primitive graph.
  /// @param engine  oneDNN CPU engine. The caller owns the engine lifetime.
  /// @param input_h Input image height (pixels, typically 224).
  /// @param input_w Input image width  (pixels, typically 224).
  explicit Vgg19(const dnnl::engine &engine, int input_h = 224,
                 int input_w = 224);

  /// @brief Loads pre-trained weights from a `.bin` file.
  ///
  /// The file should contain the weight blobs for the full network, but we only
  /// parse the first 20 weight blobs for the 10 conv layers used (up to
  /// relu5_1).
  ///
  /// @param path Path to the `.bin` weight file.
  /// @throws std::runtime_error On I/O error or blob count/size mismatch.
  void load_weights(const std::string &path);

  /// @brief Executes a forward pass through the full VGG-19 graph.
  /// @param input Pre-processed NCHW float tensor {1, 3, H, W}.
  /// @throws std::logic_error If load_weights() has not been called yet.
  /// @throws std::invalid_argument If @p input dimensions do not match.
  void forward(const Tensor &input, dnnl::stream &stream);

  /// @brief Retrieves a cached intermediate activation.
  /// @param layer  Which layer's output to retrieve.
  /// @return Const reference to the cached feature-map Tensor.
  /// @throws std::logic_error If forward() has not been called yet.
  const Tensor &get_feature_map(VggLayer layer) const;

  /// @brief Executes a backward pass through the VGG-19 graph to compute the
  /// gradient w.r.t to the input image.
  /// @param loss_gradients Gradients from the loss functions at captured
  /// layers.
  /// @return Tensor containing the gradient with respect to the input image.
  Tensor backward(const std::unordered_map<VggLayer, Tensor> &loss_gradients,
                  dnnl::stream &stream);

  /// @brief Returns true if load_weights() has completed successfully.
  bool weights_loaded() const noexcept;

  /// @brief Returns true if at least one forward() pass has completed.
  bool forward_done() const noexcept;

private:
  // ------------------------------------------------------------------ types
  struct ConvSpec {
    int in_channels;
    int out_channels;
    int kernel;  ///< Square kernel size (always 3 for VGG-19)
    int padding; ///< Same-padding (always 1 for 3×3)
    int stride;  ///< Always 1
  };

  struct LayerPrimitive {
    dnnl::memory weights_mem;
    dnnl::memory bias_mem;
    dnnl::memory src_mem;
    dnnl::memory dst_mem; ///< Also serves as src for the subsequent ReLU
    dnnl::convolution_forward conv;
    dnnl::eltwise_forward relu;
    int capture_key = -1; ///< -1 = no capture; >=0 = VggLayer enum value

    dnnl::memory diff_src_mem;
    dnnl::memory diff_dst_mem;
    dnnl::eltwise_backward relu_bw;
    dnnl::convolution_backward_data conv_bw_data;
  };

  struct PoolPrimitive {
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::pooling_forward pool;

    dnnl::memory diff_src_mem;
    dnnl::memory diff_dst_mem;
    dnnl::memory workspace_mem;
    dnnl::pooling_backward pool_bw;
  };

  // ------------------------------------------------- construction helpers
  dnnl::memory make_weights_mem(int oc, int ic, int kh, int kw);
  dnnl::memory make_bias_mem(int oc);
  void build_block(int block, const std::vector<ConvSpec> &convs,
                   dnnl::memory &in_out_mem,
                   std::vector<int> captures); ///< -1 = no capture
  void build_pool(dnnl::memory &in_out_mem);

  // ------------------------------------------------------------------ data
  dnnl::engine engine_;

  std::vector<LayerPrimitive> conv_layers_;
  std::vector<PoolPrimitive> pool_layers_;

  // Execution order: alternating conv-groups and pool layers.
  // We store the execution order as indices into conv_layers_ / pool_layers_
  // using a tagged union stored as a pair<bool, size_t>:
  //   {false, i} → conv_layers_[i]
  //   {true,  i} → pool_layers_[i]
  std::vector<std::pair<bool, std::size_t>> exec_order_;

  std::unordered_map<int, dnnl::memory> feature_map_mems_;
  mutable std::unordered_map<int, Tensor> feature_map_cache_;

  int input_h_;
  int input_w_;
  bool weights_loaded_{false};
  bool forward_done_{false};
};

} // namespace stylor

#endif // STYLOR_VGG_HPP
