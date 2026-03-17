#include "stylor/tensor.hpp"
#include <gtest/gtest.h>

using namespace stylor;

TEST(TensorTest, Initialization) {
  std::vector<dnnl::memory::dim> dims = {1, 3, 224, 224};
  Tensor tensor(dims);

  EXPECT_EQ(tensor.get_dims(), dims);

  // Check internal memory descriptor
  auto desc = tensor.get_memory().get_desc();
  EXPECT_EQ(desc.get_size(), 1 * 3 * 224 * 224 * sizeof(float));
}

TEST(TensorTest, DataAccess) {
  std::vector<dnnl::memory::dim> dims = {1, 1, 2,
                                         2}; // 4D to match NCHW format tag
  Tensor tensor(dims);

  float *data = tensor.get_data();
  ASSERT_NE(data, nullptr);

  // Write data
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;

  // Read back via const pointer
  const float *const_data = tensor.get_data();
  EXPECT_FLOAT_EQ(const_data[0], 1.0f);
  EXPECT_FLOAT_EQ(const_data[1], 2.0f);
  EXPECT_FLOAT_EQ(const_data[2], 3.0f);
  EXPECT_FLOAT_EQ(const_data[3], 4.0f);
}
