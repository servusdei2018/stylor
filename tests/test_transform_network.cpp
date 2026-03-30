#include "stylor/transform_network.hpp"
#include <dnnl.hpp>
#include <gtest/gtest.h>

class TransformNetworkTest : public ::testing::Test {
protected:
  void SetUp() override { engine = dnnl::engine(dnnl::engine::kind::cpu, 0); }
  dnnl::engine engine;
};

TEST_F(TransformNetworkTest, InitializeAndForward) {
  stylor::TransformNetwork net(engine, 256, 256);

  // Setup a random dummy tensor.
  dnnl::memory::dims dims = {1, 3, 256, 256};
  stylor::Tensor input(dims, engine);
  float *data = input.get_data();
  for (int i = 0; i < 3 * 256 * 256; ++i) {
    data[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Execute a forward pass.
  EXPECT_NO_THROW(net.forward(input));

  // Validate the output shape.
  const stylor::Tensor &output = net.get_output();
  auto out_dims = output.get_dims();
  ASSERT_EQ(out_dims.size(), 4);
  EXPECT_EQ(out_dims[0], 1);
  EXPECT_EQ(out_dims[1], 3);
  EXPECT_EQ(out_dims[2], 256);
  EXPECT_EQ(out_dims[3], 256);
}

TEST_F(TransformNetworkTest, SaveLoadWeights) {
  stylor::TransformNetwork net1(engine, 64, 64);

  // Save from net1.
  std::string file_path = "test_transform_weights.safetensors";
  EXPECT_NO_THROW(net1.save_weights(file_path));

  // Load into net2.
  stylor::TransformNetwork net2(engine, 64, 64);
  EXPECT_NO_THROW(net2.load_weights(file_path));

  // Clean up the temporary file.
  std::remove(file_path.c_str());
}
