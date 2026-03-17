#include "stylor/image.hpp"
#include "stylor/preprocessing.hpp"
#include <dnnl.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace stylor;

namespace {
dnnl::engine cpu_engine() { return dnnl::engine(dnnl::engine::kind::cpu, 0); }

Image make_rgb_image(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
  Image img;
  img.width = w;
  img.height = h;
  img.channels = 3;
  img.data.resize(static_cast<std::size_t>(w * h * 3));
  for (int i = 0; i < w * h; ++i) {
    img.data[i * 3 + 0] = r;
    img.data[i * 3 + 1] = g;
    img.data[i * 3 + 2] = b;
  }
  return img;
}
} // namespace

// ---------------------------------------------------------------------------

TEST(PreprocessingTest, OutputDimensions) {
  auto engine = cpu_engine();
  Image img = make_rgb_image(224, 224, 128, 128, 128);
  Tensor t = preprocess_image(img, engine);

  auto dims = t.get_dims();
  ASSERT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0], 1);   // N
  EXPECT_EQ(dims[1], 3);   // C
  EXPECT_EQ(dims[2], 224); // H
  EXPECT_EQ(dims[3], 224); // W
}

TEST(PreprocessingTest, NonSquareOutputDimensions) {
  auto engine = cpu_engine();
  Image img = make_rgb_image(320, 240, 0, 0, 0);
  Tensor t = preprocess_image(img, engine);

  auto dims = t.get_dims();
  EXPECT_EQ(dims[2], 240); // H
  EXPECT_EQ(dims[3], 320); // W
}

TEST(PreprocessingTest, MeanSubtraction) {
  // Pixel (R=123, G=116, B=103) should yield values very close to 0.
  // Note: VGG means are B=103.939, G=116.779, R=123.680 → small residual.
  auto engine = cpu_engine();
  Image img = make_rgb_image(1, 1,
                             /*R=*/124, /*G=*/117, /*B=*/104);
  Tensor t = preprocess_image(img, engine);

  const float *data = t.get_data();
  // Layout: [B_plane, G_plane, R_plane]
  EXPECT_NEAR(data[0], 104.0f - 103.939f, 1.0f); // B channel ~0
  EXPECT_NEAR(data[1], 117.0f - 116.779f, 1.0f); // G channel ~0
  EXPECT_NEAR(data[2], 124.0f - 123.680f, 1.0f); // R channel ~0
}

TEST(PreprocessingTest, ChannelOrderBGR) {
  // Pure red image (R=255, G=0, B=0).
  // After BGR swap: channel 0 (B) ≈ -103.939, channel 1 (G) ≈ -116.779,
  //                  channel 2 (R) ≈ 255 - 123.680 = 131.320
  auto engine = cpu_engine();
  Image img = make_rgb_image(1, 1, /*R=*/255, /*G=*/0, /*B=*/0);
  Tensor t = preprocess_image(img, engine);

  const float *data = t.get_data();
  EXPECT_NEAR(data[0], -103.939f, 0.5f); // B channel
  EXPECT_NEAR(data[1], -116.779f, 0.5f); // G channel
  EXPECT_NEAR(data[2], 131.320f, 0.5f);  // R channel
}

TEST(PreprocessingTest, ZeroWidthThrows) {
  auto engine = cpu_engine();
  Image bad;
  bad.width = 0;
  bad.height = 224;
  bad.channels = 3;
  EXPECT_THROW(preprocess_image(bad, engine), std::invalid_argument);
}

TEST(PreprocessingTest, WrongChannelCountThrows) {
  auto engine = cpu_engine();
  Image bad;
  bad.width = 224;
  bad.height = 224;
  bad.channels = 1; // grayscale — unsupported
  bad.data.resize(224 * 224, 128);
  EXPECT_THROW(preprocess_image(bad, engine), std::invalid_argument);
}
