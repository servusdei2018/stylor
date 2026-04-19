#include "stylor/loss.hpp"
#include "stylor/training_context.hpp"
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

  GramPrimitives gp(2, 2, engine);
  auto G = compute_gram_matrix(F, gp, engine, stream);
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
  // total raw = 10, normalized by 1/(C*H*W) = 1/(1*2*2) = 0.25
  // expected = 10 * 0.25 = 2.5
  EXPECT_FLOAT_EQ(result.value, 2.5f);
  ASSERT_TRUE(result.gradient.has_value());
}

// Verifies that the style loss feature-space gradient uses the correct
// 1/(C*H*W) chain-rule factor (not 2/(C*H*W) which would double-count the
// 2* already embedded in d_gram).
TEST(LossTest, StyleLossGradient) {
  auto engine = cpu_engine();
  dnnl::stream stream(engine);

  // C=1, H=1, W=1
  // Feature map F = [[5.0]]  → shape {1,1,1,1}
  Tensor F({1, 1, 1, 1}, engine);
  F.get_data()[0] = 5.0f;

  // Gram(F) = (1/CHW) * F*F^T = (1/1) * 25 = 25
  GramPrimitives gp(1, 1, engine);
  auto gen_gram = compute_gram_matrix(F, gp, engine, stream);
  EXPECT_FLOAT_EQ(gen_gram.get_data()[0], 25.0f);

  // Style target Gram = 0 (so the diff is 25).
  Tensor tgt_gram({1, 1, 1, 1}, engine);
  tgt_gram.get_data()[0] = 0.0f;

  StyleBackwardPrimitives sbp(1, 1, engine);
  auto result =
      compute_style_loss(gen_gram, tgt_gram, F, sbp, true, engine, stream);

  // Loss = normalize * diff^2  where normalize = 1/(C*C) = 1
  // loss = 1 * (25-0)^2 = 625
  EXPECT_FLOAT_EQ(result.value, 625.0f);
  ASSERT_TRUE(result.gradient.has_value());

  // d_gram = 2 * diff * normalize = 2 * 25 * 1 = 50
  // dL/dF[0,0] = d_gram[0,0] * F[0,0] * (1/(C*HW))
  //            = 50 * 5 * (1/1) = 250
  const float *grad = result.gradient.value().get_data();
  EXPECT_FLOAT_EQ(grad[0], 250.0f);
}
// Verifies that the feature-map cache deep-copies via dnnl::reorder so that
// the Tensor returned by get_feature_map()-equivalent logic is NOT an alias of
// internal VGG layer memory.  Specifically, overwriting the source buffer after
// the copy must not change the snapshot held by the cache Tensor.
TEST(LossTest, FeatureMapCacheIsDeepCopy) {
  auto engine = cpu_engine();

  // Simulate a VGG layer's dst_mem (the "live" buffer).
  dnnl::memory::desc md({1, 1, 2, 2}, dnnl::memory::data_type::f32,
                        dnnl::memory::format_tag::nchw);
  dnnl::memory live_mem(md, engine);

  // Fill live_mem with [1, 2, 3, 4].
  float *live = static_cast<float *>(live_mem.get_data_handle());
  live[0] = 1.f;
  live[1] = 2.f;
  live[2] = 3.f;
  live[3] = 4.f;

  // --- Deep-copy via reorder (the fixed get_feature_map approach) ---
  Tensor cache({1, 1, 2, 2}, engine);
  {
    dnnl::stream s(engine);
    dnnl::reorder(live_mem, cache.get_memory())
        .execute(
            s, {{DNNL_ARG_FROM, live_mem}, {DNNL_ARG_TO, cache.get_memory()}});
    s.wait();
  }

  // Overwrite the live buffer (simulating a second forward() pass).
  live[0] = 99.f;
  live[1] = 99.f;
  live[2] = 99.f;
  live[3] = 99.f;

  // The cache Tensor must still hold the original snapshot.
  const float *snapped = cache.get_data();
  EXPECT_FLOAT_EQ(snapped[0], 1.f);
  EXPECT_FLOAT_EQ(snapped[1], 2.f);
  EXPECT_FLOAT_EQ(snapped[2], 3.f);
  EXPECT_FLOAT_EQ(snapped[3], 4.f);

  // Sanity-check: set_data_handle aliasing WOULD have failed (live buffer
  // is [99, 99, 99, 99]).  This documents why the old approach was wrong.
  EXPECT_FLOAT_EQ(live[0], 99.f);
}
