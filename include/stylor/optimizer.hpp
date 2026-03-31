#ifndef STYLOR_OPTIMIZER_HPP
#define STYLOR_OPTIMIZER_HPP

#include "stylor/transform_network.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace stylor {

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void
  step(std::unordered_map<std::string, TransformNetwork::ParamDescriptor>
           &params) = 0;
};

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                float beta2 = 0.999f, float epsilon = 1e-8f);

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
