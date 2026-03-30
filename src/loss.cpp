#include "stylor/loss.hpp"

namespace stylor {

Tensor compute_gram_matrix(const Tensor &feature_map,
                           const dnnl::engine &engine, dnnl::stream &stream) {
  auto dims = feature_map.get_dims();
  const auto C = dims[1];
  const auto H = dims[2];
  const auto W = dims[3];

  dnnl::memory::dims src_dims = {C, H * W};
  dnnl::memory::dims wei_dims = {H * W, C};
  dnnl::memory::dims dst_dims = {C, C};

  dnnl::memory::dims src_strides = {H * W, 1};
  dnnl::memory::dims wei_strides = {1, H * W}; // Transpose F for the weights
  dnnl::memory::dims dst_strides = {C, 1};

  auto src_md =
      dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, src_strides);
  auto wei_md =
      dnnl::memory::desc(wei_dims, dnnl::memory::data_type::f32, wei_strides);
  auto dst_md =
      dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dst_strides);

  auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md);
  auto matmul = dnnl::matmul(matmul_pd);

  Tensor dst({1, 1, C, C}, engine);

  dnnl::memory src_mem(src_md, engine,
                       feature_map.get_memory().get_data_handle());
  dnnl::memory wei_mem(wei_md, engine,
                       feature_map.get_memory().get_data_handle());
  dnnl::memory dst_mem(dst_md, engine, dst.get_memory().get_data_handle());

  matmul.execute(stream, {{DNNL_ARG_SRC, src_mem},
                          {DNNL_ARG_WEIGHTS, wei_mem},
                          {DNNL_ARG_DST, dst_mem}});

  // Scaling by 1 / (C * H * W)
  stream.wait();
  float scale = 1.0f / static_cast<float>(C * H * W);
  float *dst_ptr = dst.get_data();
  for (std::size_t i = 0; i < static_cast<std::size_t>(C * C); ++i) {
    dst_ptr[i] *= scale;
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
  float *grad_ptr = nullptr;

  if (compute_grad) {
    grad.emplace(dims, engine);
    grad_ptr = (*grad).get_data();
  }

  for (std::size_t i = 0; i < total; ++i) {
    float diff = gen_ptr[i] - tgt_ptr[i];
    loss += diff * diff;
    if (compute_grad) {
      grad_ptr[i] = 2.0f * diff * normalize;
    }
  }

  loss *= normalize;
  return {loss, std::move(grad)};
}

LossResult compute_style_loss(const Tensor &generated_gram,
                              const Tensor &target_gram,
                              const Tensor &generated_features,
                              bool compute_grad, const dnnl::engine &engine,
                              dnnl::stream &stream) {
  auto g_dims = generated_gram.get_dims();
  const auto C = g_dims[1];
  const float normalize = 1.0f / static_cast<float>(C * C);

  const float *gen_gram_ptr = generated_gram.get_data();
  const float *tgt_gram_ptr = target_gram.get_data();

  float loss = 0.0f;

  // Array to hold dLoss/dG
  Tensor d_gram({1, 1, C, C}, engine);
  float *d_gram_ptr = d_gram.get_data();

  for (std::size_t i = 0; i < static_cast<std::size_t>(C * C); ++i) {
    float diff = gen_gram_ptr[i] - tgt_gram_ptr[i];
    loss += diff * diff;
    // dL/dG = 2 * (G - T) / (C^2)
    d_gram_ptr[i] = 2.0f * diff * normalize;
  }

  loss *= normalize;

  std::optional<Tensor> grad;
  if (compute_grad) {
    auto f_dims = generated_features.get_dims();
    grad.emplace(f_dims, engine);

    // Backprop through Gram Matrix: dL/dF = dL/dG * F
    // dL/dG: [C, C]
    // F: [C, H*W]
    // Output: [C, H*W]
    const auto HW = f_dims[2] * f_dims[3];

    dnnl::memory::dims src_dims = {C, C};
    dnnl::memory::dims wei_dims = {C, HW};
    dnnl::memory::dims dst_dims = {C, HW};

    dnnl::memory::dims src_strides = {C, 1};
    dnnl::memory::dims wei_strides = {1, HW};
    dnnl::memory::dims dst_strides = {HW, 1};

    auto src_md =
        dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, src_strides);
    auto wei_md =
        dnnl::memory::desc(wei_dims, dnnl::memory::data_type::f32, wei_strides);
    auto dst_md =
        dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dst_strides);

    auto matmul_pd =
        dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md);
    auto matmul = dnnl::matmul(matmul_pd);

    dnnl::memory src_mem(src_md, engine, d_gram.get_memory().get_data_handle());
    dnnl::memory wei_mem(wei_md, engine,
                         generated_features.get_memory().get_data_handle());
    dnnl::memory dst_mem(dst_md, engine,
                         (*grad).get_memory().get_data_handle());

    matmul.execute(stream, {{DNNL_ARG_SRC, src_mem},
                            {DNNL_ARG_WEIGHTS, wei_mem},
                            {DNNL_ARG_DST, dst_mem}});

    // Gram forwardpass sums over HW, and scaled by 1/(C*H*W).
    // The backward passes the scale too if it's considered part of the forward.
    // If G = 1/(CHW) * F * F^T, then dG/dF is 2/(CHW) * F.
    // By chain rule, dL/dF = dL/dG * dG/dF = dL/dG * (2/(CHW)) * F.
    stream.wait();

    float chain_scale = 2.0f / static_cast<float>(C * HW);
    float *grad_ptr = (*grad).get_data();
    for (std::size_t i = 0; i < static_cast<std::size_t>(C * HW); ++i) {
      grad_ptr[i] *= chain_scale;
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

  const float *img_ptr = image.get_data();
  float loss = 0.0f;

  std::optional<Tensor> grad;
  float *grad_ptr = nullptr;
  if (compute_grad) {
    grad.emplace(dims, engine);
    grad_ptr = (*grad).get_data();
    for (std::size_t i = 0; i < static_cast<std::size_t>(1 * C * H * W); ++i) {
      grad_ptr[i] = 0.0f;
    }
  }

  for (std::size_t c = 0; c < static_cast<std::size_t>(C); ++c) {
    for (std::size_t h = 0; h < static_cast<std::size_t>(H); ++h) {
      for (std::size_t w = 0; w < static_cast<std::size_t>(W); ++w) {
        std::size_t idx = c * H * W + h * W + w;

        if (h < static_cast<std::size_t>(H - 1)) {
          std::size_t idx_h = c * H * W + (h + 1) * W + w;
          float diff_h = img_ptr[idx_h] - img_ptr[idx];
          loss += diff_h * diff_h;
          if (compute_grad) {
            grad_ptr[idx] += -2.0f * diff_h;
            grad_ptr[idx_h] += 2.0f * diff_h;
          }
        }

        if (w < static_cast<std::size_t>(W - 1)) {
          std::size_t idx_w = c * H * W + h * W + (w + 1);
          float diff_w = img_ptr[idx_w] - img_ptr[idx];
          loss += diff_w * diff_w;
          if (compute_grad) {
            grad_ptr[idx] += -2.0f * diff_w;
            grad_ptr[idx_w] += 2.0f * diff_w;
          }
        }
      }
    }
  }

  return {loss, std::move(grad)};
}

} // namespace stylor
