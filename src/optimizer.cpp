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

    std::size_t elem_count = 1;
    for (int d : desc.shape) {
      elem_count *= d;
    }

    if (state_.find(name) == state_.end()) {
      state_[name] = {std::vector<float>(elem_count, 0.0f),
                      std::vector<float>(elem_count, 0.0f)};
    }

    // Zero out param updates if the parameter tensor is missing
    if (!desc.mem || !desc.diff_mem)
      continue;

    float *w = static_cast<float *>(desc.mem.get_data_handle());
    const float *grad =
        static_cast<const float *>(desc.diff_mem.get_data_handle());
    float *m = state_[name].m.data();
    float *v = state_[name].v.data();

    for (std::size_t i = 0; i < elem_count; ++i) {
      float g = grad[i];
      m[i] = beta1_ * m[i] + (1.0f - beta1_) * g;
      v[i] = beta2_ * v[i] + (1.0f - beta2_) * g * g;

      // Adam update formula
      w[i] -= alpha_t * m[i] / (std::sqrt(v[i]) + epsilon_);
    }
  }
}

} // namespace stylor
