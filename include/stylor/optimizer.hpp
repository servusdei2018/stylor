#ifndef STYLOR_OPTIMIZER_HPP
#define STYLOR_OPTIMIZER_HPP

#include "stylor/transform_network.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace stylor {

/// @brief Base class for parameter optimizers.
class Optimizer {
public:
  /// @brief Virtual destructor.
  virtual ~Optimizer() = default;

  /// @brief Performs a single parameter update step.
  /// @param params A map from parameter names to their memory descriptors,
  /// including their gradients.
  virtual void
  step(std::unordered_map<std::string, TransformNetwork::ParamDescriptor>
           &params) = 0;
};

/// @brief Implements the Adam optimization algorithm.
class AdamOptimizer : public Optimizer {
public:
  /// @brief Constructs an Adam optimizer.
  /// @param learning_rate The learning rate (step size).
  /// @param beta1 The exponential decay rate for the first moment estimates.
  /// @param beta2 The exponential decay rate for the second-moment estimates.
  /// @param epsilon A small constant for numerical stability.
  AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                float beta2 = 0.999f, float epsilon = 1e-8f);

  /// @brief Performs a single Adam parameter update step.
  /// @param params A map from parameter names to their memory descriptors,
  /// including their gradients.
  void step(std::unordered_map<std::string, TransformNetwork::ParamDescriptor>
                &params) override;

private:
  float lr_;
  float beta1_;
  float beta2_;
  float epsilon_;
  int t_; // Timestep

  struct Moments {
    std::vector<float> m;
    std::vector<float> v;
  };
  std::unordered_map<std::string, Moments> state_;
};

} // namespace stylor

#endif // STYLOR_OPTIMIZER_HPP
