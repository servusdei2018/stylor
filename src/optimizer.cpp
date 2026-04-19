#include "stylor/optimizer.hpp"
#include <cmath>

namespace stylor {

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2,
                             float epsilon)
    : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      t_(0) {}

void AdamOptimizer::step(
    std::unordered_map<std::string, TransformNetwork::ParamDescriptor>
        &params) {
  t_++;
  const float beta1_pow = std::pow(beta1_, t_);
  const float beta2_pow = std::pow(beta2_, t_);
  const float alpha_t = lr_ * std::sqrt(1.0f - beta2_pow) / (1.0f - beta1_pow);

  for (auto &pair : params) {
    const std::string &name = pair.first;
    TransformNetwork::ParamDescriptor &desc = pair.second;

    // 4a: single try_emplace replaces find + operator[] (was 4 hash-map ops).
    // 4c: allocate interleaved [m0,v0, m1,v1, …] in one contiguous buffer.
    // 4b: use cached elem_count instead of recomputing the shape product.
    const std::size_t elem_count = desc.elem_count;
    auto [it, inserted] = state_.try_emplace(
        name, Moments{std::vector<float>(2 * elem_count, 0.0f), elem_count});
    Moments &moments = it->second;

    // Zero out param updates if the parameter tensor is missing
    if (!desc.mem || !desc.diff_mem)
      continue;

    float *w = static_cast<float *>(desc.mem.get_data_handle());
    const float *grad =
        static_cast<const float *>(desc.diff_mem.get_data_handle());
    float *mv = moments.mv.data();

    // Hoist loop-invariant bias-correction coefficients.
    const float one_minus_beta1 = 1.0f - beta1_;
    const float one_minus_beta2 = 1.0f - beta2_;

#pragma omp parallel for simd
    for (std::size_t i = 0; i < elem_count; ++i) {
      const float g = grad[i];
      const std::size_t j = 2 * i;

      // 4c: m and v for element i are adjacent → single cache-line read/write.
      const float mi = beta1_ * mv[j] + one_minus_beta1 * g;
      const float vi = beta2_ * mv[j + 1] + one_minus_beta2 * g * g;
      mv[j] = mi;
      mv[j + 1] = vi;

      // Adam update formula
      w[i] -= alpha_t * mi / (std::sqrt(vi) + epsilon_);
    }
  }
}

} // namespace stylor
