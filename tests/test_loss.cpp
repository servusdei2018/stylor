#include "stylor/loss.hpp"
#include <gtest/gtest.h>

using namespace stylor;

namespace {

dnnl::engine cpu_engine() { return dnnl::engine(dnnl::engine::kind::cpu, 0); }

} // namespace

TEST(LossTest, GramMatrix) {
  auto engine = cpu_engine();
  dnnl::stream stream(engine);
  // shape: {1, C=2, H=1, W=2}
  Tensor F({1, 2, 1, 2}, engine);
  float *F_ptr = F.get_data();
  // Channel 1: [1, 2]
  // Channel 2: [3, 4]
  F_ptr[0] = 1.0f;
  F_ptr[1] = 2.0f;
  F_ptr[2] = 3.0f;
  F_ptr[3] = 4.0f;

  auto G = compute_gram_matrix(F, engine, stream);
  auto dims = G.get_dims();
  EXPECT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0], 1);
  EXPECT_EQ(dims[1], 1);
  EXPECT_EQ(dims[2], 2);
  EXPECT_EQ(dims[3], 2);

  float *G_ptr = G.get_data();
  // G = F * F^T
  // Expected F * F^T = [[1+4, 3+8], [3+8, 9+16]] = [[5, 11], [11, 25]]
  // Scaling by 1 / (C * H * W) = 1 / 4.
  // [[1.25, 2.75], [2.75, 6.25]]
  EXPECT_FLOAT_EQ(G_ptr[0], 1.25f);
  EXPECT_FLOAT_EQ(G_ptr[1], 2.75f);
  EXPECT_FLOAT_EQ(G_ptr[2], 2.75f);
  EXPECT_FLOAT_EQ(G_ptr[3], 6.25f);
}

TEST(LossTest, ContentLoss) {
  auto engine = cpu_engine();
  dnnl::stream stream(engine);
  Tensor gen({1, 1, 2, 2}, engine);
  Tensor tgt({1, 1, 2, 2}, engine);

  float *gen_ptr = gen.get_data();
  float *tgt_ptr = tgt.get_data();
  for (int i = 0; i < 4; ++i) {
    gen_ptr[i] = static_cast<float>(i + 1);
    tgt_ptr[i] = static_cast<float>(i);
  }

  // difference is 1 for every element. MSE = sum(1^2) / 4 = 1.0
  auto result = compute_content_loss(gen, tgt, true, engine, stream);

  EXPECT_FLOAT_EQ(result.value, 1.0f);
  ASSERT_TRUE(result.gradient.has_value());

  const float *grad_ptr = result.gradient.value().get_data();
  // grad = 2 / 4 * (gen - tgt) = 0.5 * 1 = 0.5
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(grad_ptr[i], 0.5f);
  }
}

TEST(LossTest, TotalVariationLoss) {
  auto engine = cpu_engine();
  dnnl::stream stream(engine);
  // shape: {1, 1, 2, 2}
  Tensor img({1, 1, 2, 2}, engine);
  float *img_ptr = img.get_data();
  // [[1, 2]
  //  [3, 4]]
  img_ptr[0] = 1.0f;
  img_ptr[1] = 2.0f;
  img_ptr[2] = 3.0f;
  img_ptr[3] = 4.0f;

  auto result = compute_tv_loss(img, true, engine, stream);

  // h differences: (3-1)^2 + (4-2)^2 = 4 + 4 = 8
  // w differences: (2-1)^2 + (4-3)^2 = 1 + 1 = 2
  // total = 10
  EXPECT_FLOAT_EQ(result.value, 10.0f);
  ASSERT_TRUE(result.gradient.has_value());
}
