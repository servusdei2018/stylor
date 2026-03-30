#ifndef STYLOR_LOSS_HPP
#define STYLOR_LOSS_HPP

#include "stylor/tensor.hpp"
#include <dnnl.hpp>
#include <optional>

namespace stylor {

/// @brief Result of a loss computation, containing the scalar loss and an
/// optional gradient tensor.
struct LossResult {
  float value;
  std::optional<Tensor> gradient;
};

/// @brief Computes the Gram matrix of a given feature map.
/// @param feature_map The input tensor of shape {1, C, H, W}.
/// @param engine The oneDNN engine.
/// @param stream The oneDNN stream for execution.
/// @return The Gram matrix of shape {1, C, C}.
Tensor compute_gram_matrix(const Tensor &feature_map,
                           const dnnl::engine &engine, dnnl::stream &stream);

/// @brief Computes the content loss (MSE) and its gradient with respect to the
/// generated features.
/// @param generated The generated feature map {1, C, H, W}.
/// @param target The target feature map {1, C, H, W}.
/// @param compute_grad Whether to compute the gradient tensor.
/// @param engine The oneDNN engine.
/// @param stream The oneDNN stream for execution.
/// @return LossResult containing the scalar loss and optionally the gradient
/// tensor.
LossResult compute_content_loss(const Tensor &generated, const Tensor &target,
                                bool compute_grad, const dnnl::engine &engine,
                                dnnl::stream &stream);

/// @brief Computes the style loss (MSE of Gram matrices) and its gradient with
/// respect to the generated features.
/// @param generated_gram The Gram matrix of the generated image {1, C, C}.
/// @param target_gram The Gram matrix of the style image {1, C, C}.
/// @param generated_features The feature map of the generated image {1, C, H,
/// W} (required for gradient projection).
/// @param compute_grad Whether to compute the gradient tensor.
/// @param engine The oneDNN engine.
/// @param stream The oneDNN stream for execution.
/// @return LossResult containing the scalar loss and optionally the gradient
/// tensor.
LossResult compute_style_loss(const Tensor &generated_gram,
                              const Tensor &target_gram,
                              const Tensor &generated_features,
                              bool compute_grad, const dnnl::engine &engine,
                              dnnl::stream &stream);

/// @brief Computes the Total Variation (TV) loss and its gradient with respect
/// to the image.
/// @param image The input image {1, C, H, W}.
/// @param compute_grad Whether to compute the gradient tensor.
/// @param engine The oneDNN engine.
/// @param stream The oneDNN stream for execution.
/// @return LossResult containing the scalar loss and optionally the gradient
/// tensor.
LossResult compute_tv_loss(const Tensor &image, bool compute_grad,
                           const dnnl::engine &engine, dnnl::stream &stream);

} // namespace stylor

#endif // STYLOR_LOSS_HPP
