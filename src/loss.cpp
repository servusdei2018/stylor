#include "stylor/loss.hpp"
#include <cstring>

namespace stylor {

Tensor compute_gram_matrix(const Tensor &feature_map,
                           const dnnl::engine &engine, dnnl::stream &stream) {
  auto dims = feature_map.get_dims();
  const auto C = dims[1];
  const auto H = dims[2];
  const auto W = dims[3];
  const auto HW = H * W;

  Tensor dst({1, 1, C, C}, engine);
  const float *f_ptr = feature_map.get_data();
  float *g_ptr = dst.get_data();

  // Manual Gram calculation: G = F * F^T
  // G[c1, c2] = sum_hw (F[c1, hw] * F[c2, hw])
  // Scaled by 1/(C*H*W)
  const float scale = 1.0f / static_cast<float>(C * H * W);

  for (std::size_t c1 = 0; c1 < static_cast<std::size_t>(C); ++c1) {
    for (std::size_t c2 = 0; c2 < static_cast<std::size_t>(C); ++c2) {
      float sum = 0.0f;
      for (std::size_t hw = 0; hw < static_cast<std::size_t>(HW); ++hw) {
        sum += f_ptr[c1 * HW + hw] * f_ptr[c2 * HW + hw];
      }
      g_ptr[c1 * C + c2] = sum * scale;
    }
  }

  return dst;
}

LossResult compute_content_loss(const Tensor &generated, const Tensor &target,
                                bool compute_grad, const dnnl::engine &engine,
                                dnnl::stream &stream) {
  auto dims = generated.get_dims();
  const std::size_t total = dims[1] * dims[2] * dims[3];
  const float normalize = 1.0f / static_cast<float>(total);

  const float *gen_ptr = generated.get_data();
  const float *tgt_ptr = target.get_data();

  float loss = 0.0f;
  std::optional<Tensor> grad;
  if (compute_grad) {
    grad.emplace(dims, engine);
  }

  float *grad_ptr = compute_grad ? (*grad).get_data() : nullptr;

  for (std::size_t i = 0; i < total; ++i) {
    float diff = gen_ptr[i] - tgt_ptr[i];
    loss += diff * diff;
    if (compute_grad) {
      grad_ptr[i] = 2.0f * diff * normalize;
    }
  }

  return {loss * normalize, std::move(grad)};
}

LossResult compute_style_loss(const Tensor &generated_gram,
                              const Tensor &target_gram,
                              const Tensor &generated_features,
                              bool compute_grad, const dnnl::engine &engine,
                              dnnl::stream &stream) {
  auto g_dims = generated_gram.get_dims();
  const auto C = g_dims[2];
  const float normalize = 1.0f / static_cast<float>(C * C);

  const float *gen_gram_ptr = generated_gram.get_data();
  const float *tgt_gram_ptr = target_gram.get_data();

  float loss = 0.0f;
  Tensor d_gram({1, 1, C, C}, engine);
  float *d_gram_ptr = d_gram.get_data();

  for (std::size_t i = 0; i < static_cast<std::size_t>(C * C); ++i) {
    float diff = gen_gram_ptr[i] - tgt_gram_ptr[i];
    loss += diff * diff;
    d_gram_ptr[i] = 2.0f * diff * normalize;
  }

  loss *= normalize;

  std::optional<Tensor> grad;
  if (compute_grad) {
    auto f_dims = generated_features.get_dims();
    grad.emplace(f_dims, engine);
    const auto HW = f_dims[2] * f_dims[3];

    const float *f_ptr = generated_features.get_data();
    float *df_ptr = (*grad).get_data();

    // Manual dL/dF = dL/dG * F
    for (std::size_t c1 = 0; c1 < static_cast<std::size_t>(C); ++c1) {
      for (std::size_t hw = 0; hw < static_cast<std::size_t>(HW); ++hw) {
        float sum = 0.0f;
        for (std::size_t c2 = 0; c2 < static_cast<std::size_t>(C); ++c2) {
          sum += d_gram_ptr[c1 * C + c2] * f_ptr[c2 * HW + hw];
        }
        // Gram forward pass had a 1/(CHW) scale. dG/dF has a 2/(CHW) scale?
        // Let's be consistent. Forward: G = (1/CHW) * F * F^T.
        // dG/dF = (1/CHW) * (F*dF^T + dF*F^T).
        // For dL/dF = dL/dG * (1/CHW) * 2 * F
        df_ptr[c1 * HW + hw] = sum * (2.0f / static_cast<float>(C * HW));
      }
    }
  }

  return {loss, std::move(grad)};
}

LossResult compute_tv_loss(const Tensor &image, bool compute_grad,
                           const dnnl::engine &engine, dnnl::stream &stream) {
  auto dims = image.get_dims();
  const auto C = dims[1];
  const auto H = dims[2];
  const auto W = dims[3];

  const float *data = image.get_data();
  float loss = 0.0f;

  std::optional<Tensor> grad;
  if (compute_grad) {
    grad.emplace(dims, engine);
    std::memset((*grad).get_data(), 0, C * H * W * 4);
  }

  for (std::size_t c = 0; c < static_cast<std::size_t>(C); ++c) {
    for (std::size_t h = 0; h < static_cast<std::size_t>(H - 1); ++h) {
      for (std::size_t w = 0; w < static_cast<std::size_t>(W - 1); ++w) {
        std::size_t i = c * H * W + h * W + w;
        std::size_t i_h = c * H * W + (h + 1) * W + w;
        std::size_t i_w = c * H * W + h * W + (w + 1);

        float dh = data[i] - data[i_h];
        float dw = data[i] - data[i_w];
        loss += dh * dh + dw * dw;

        if (compute_grad) {
          float *g = (*grad).get_data();
          g[i] += 2.0f * (dh + dw);
          g[i_h] -= 2.0f * dh;
          g[i_w] -= 2.0f * dw;
        }
      }
    }
  }

  return {loss, std::move(grad)};
}

} // namespace stylor
