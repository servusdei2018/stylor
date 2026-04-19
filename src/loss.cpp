#include "stylor/loss.hpp"
#include "stylor/training_context.hpp"
#include <cstring>

namespace stylor {

Tensor compute_gram_matrix(const Tensor &feature_map, const GramPrimitives &gp,
                           const dnnl::engine &engine, dnnl::stream &stream) {
  auto dims = feature_map.get_dims();
  const auto C = dims[1];
  const auto H = dims[2];
  const auto W = dims[3];

  Tensor dst({1, 1, C, C}, engine);
  float *g_ptr = dst.get_data();
  void *f_raw =
      static_cast<void *>(const_cast<float *>(feature_map.get_data()));

  dnnl::memory src_mem(gp.src_md, engine, f_raw);
  dnnl::memory wei_mem(gp.wei_md, engine, f_raw);
  dnnl::memory dst_mem(gp.dst_md, engine, g_ptr);

  gp.prim.execute(stream, {{DNNL_ARG_SRC, src_mem},
                           {DNNL_ARG_WEIGHTS, wei_mem},
                           {DNNL_ARG_DST, dst_mem}});
  stream.wait();

  const float scale = 1.0f / static_cast<float>(C * H * W);
#pragma omp parallel for simd
  for (std::size_t i = 0; i < static_cast<std::size_t>(C * C); ++i) {
    g_ptr[i] *= scale;
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

#pragma omp parallel for simd reduction(+ : loss)
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
                              const StyleBackwardPrimitives &sbp,
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

#pragma omp parallel for simd reduction(+ : loss)
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

    // dL/dF via chain rule through the Gram forward pass.
    // df = d_gram * F
    dnnl::memory src_mem(sbp.src_md, engine, static_cast<void *>(d_gram_ptr));
    dnnl::memory wei_mem(sbp.wei_md, engine,
                         static_cast<void *>(const_cast<float *>(f_ptr)));
    dnnl::memory dst_mem(sbp.dst_md, engine, static_cast<void *>(df_ptr));

    sbp.prim.execute(stream, {{DNNL_ARG_SRC, src_mem},
                              {DNNL_ARG_WEIGHTS, wei_mem},
                              {DNNL_ARG_DST, dst_mem}});
    stream.wait();

    const float scale = 1.0f / static_cast<float>(C * HW);
#pragma omp parallel for simd
    for (std::size_t i = 0; i < static_cast<std::size_t>(C * HW); ++i) {
      df_ptr[i] *= scale;
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

  // Normalise by total elements so TV loss is comparable in magnitude to the
  // content loss (which is also normalised by 1/(C·H·W)).
  const float normalize = 1.0f / static_cast<float>(C * H * W);

  std::optional<Tensor> grad;
  if (compute_grad) {
    grad.emplace(dims, engine);
  }

  float *grad_ptr = compute_grad ? (*grad).get_data() : nullptr;

  if (compute_grad) {
#pragma omp parallel for simd collapse(3) reduction(+ : loss)
    for (std::size_t c = 0; c < static_cast<std::size_t>(C); ++c) {
      for (std::size_t h = 0; h < static_cast<std::size_t>(H); ++h) {
        for (std::size_t w = 0; w < static_cast<std::size_t>(W); ++w) {
          std::size_t i = c * H * W + h * W + w;
          float val = data[i];
          float l = 0.0f;
          float g = 0.0f;

          if (h < static_cast<std::size_t>(H - 1)) {
            float d = val - data[i + W];
            l += d * d;
            g += d;
          }
          if (h > 0) {
            float d = data[i - W] - val;
            g -= d;
          }
          if (w < static_cast<std::size_t>(W - 1)) {
            float d = val - data[i + 1];
            l += d * d;
            g += d;
          }
          if (w > 0) {
            float d = data[i - 1] - val;
            g -= d;
          }

          loss += l * normalize;
          grad_ptr[i] = 2.0f * g * normalize;
        }
      }
    }
  } else {
#pragma omp parallel for simd collapse(3) reduction(+ : loss)
    for (std::size_t c = 0; c < static_cast<std::size_t>(C); ++c) {
      for (std::size_t h = 0; h < static_cast<std::size_t>(H); ++h) {
        for (std::size_t w = 0; w < static_cast<std::size_t>(W); ++w) {
          std::size_t i = c * H * W + h * W + w;
          float val = data[i];
          float l = 0.0f;

          if (h < static_cast<std::size_t>(H - 1)) {
            float d = val - data[i + W];
            l += d * d;
          }
          if (w < static_cast<std::size_t>(W - 1)) {
            float d = val - data[i + 1];
            l += d * d;
          }

          loss += l * normalize;
        }
      }
    }
  }

  return {loss, std::move(grad)};
}

} // namespace stylor
