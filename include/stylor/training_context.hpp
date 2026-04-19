#ifndef STYLOR_TRAINING_CONTEXT_HPP
#define STYLOR_TRAINING_CONTEXT_HPP

#include "stylor/vgg.hpp"
#include <dnnl.hpp>
#include <unordered_map>

namespace stylor {

/// @brief Pre-built oneDNN matmul primitive for Gram matrix computation.
/// Constructed once and reused across training iterations to avoid redundant
/// JIT compilation.
struct GramPrimitives {
  dnnl::memory::desc src_md; ///< {C, HW} row-major
  dnnl::memory::desc wei_md; ///< {HW, C} column-major (transposed view)
  dnnl::memory::desc dst_md; ///< {C, C} row-major
  dnnl::memory::dim C;
  dnnl::memory::dim HW;
  dnnl::matmul prim; ///< Must be declared after memory descs (init order).

  /// @brief Constructs Gram primitives for the given channel and spatial dims.
  /// @param C  Number of channels.
  /// @param HW Product of spatial height and width (H * W).
  /// @param engine oneDNN engine used for JIT compilation.
  GramPrimitives(dnnl::memory::dim C, dnnl::memory::dim HW,
                 const dnnl::engine &engine);
};

/// @brief Pre-built oneDNN matmul primitive for the style loss backward pass
/// (dL/dF = d_gram * F).
struct StyleBackwardPrimitives {
  dnnl::memory::desc src_md; ///< {C, C}  — d_gram
  dnnl::memory::desc wei_md; ///< {C, HW} — features
  dnnl::memory::desc dst_md; ///< {C, HW} — gradient dF
  dnnl::memory::dim C;
  dnnl::memory::dim HW;
  dnnl::matmul prim; ///< Must be declared after memory descs (init order).

  /// @brief Constructs style backward primitives for the given dimensions.
  /// @param C  Number of channels.
  /// @param HW Product of spatial height and width (H * W).
  /// @param engine oneDNN engine used for JIT compilation.
  StyleBackwardPrimitives(dnnl::memory::dim C, dnnl::memory::dim HW,
                          const dnnl::engine &engine);
};

/// @brief Pre-built oneDNN primitives for the training loop.
///
/// Caches matmul primitives for Gram matrix computation and style loss
/// backward pass for each VGG-19 style layer.  Constructed once per training
/// run and reused every iteration, eliminating the repeated JIT compilation
/// that would otherwise dominate per-iteration cost.
class TrainingContext {
public:
  /// @brief Pre-builds cached primitives for all style layers.
  /// @param engine  Shared oneDNN engine.
  /// @param image_h Training image height.
  /// @param image_w Training image width.
  TrainingContext(const dnnl::engine &engine, int image_h, int image_w);

  /// @brief Returns the cached Gram matrix primitives for a style layer.
  const GramPrimitives &gram(VggLayer layer) const;

  /// @brief Returns the cached style backward primitives for a style layer.
  const StyleBackwardPrimitives &style_bw(VggLayer layer) const;

private:
  std::unordered_map<int, GramPrimitives> gram_cache_;
  std::unordered_map<int, StyleBackwardPrimitives> style_bw_cache_;
};

} // namespace stylor

#endif // STYLOR_TRAINING_CONTEXT_HPP
