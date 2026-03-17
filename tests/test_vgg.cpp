#include "stylor/image.hpp"
#include "stylor/preprocessing.hpp"
#include "stylor/vgg.hpp"
#include <cstdint>
#include <dnnl.hpp>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace fs = std::filesystem;
using namespace stylor;

namespace {

dnnl::engine cpu_engine() { return dnnl::engine(dnnl::engine::kind::cpu, 0); }

// Write a zero-valued .bin file that satisfies every weight/bias blob for
// VGG-19: 13 conv layers, each with a weight tensor and a bias vector.
//
// Blob order (VGG-19, all 3×3 convolutions):
//   Block 1:  [64×3×3×3, 64] [64×64×3×3, 64]
//   Block 2:  [128×64×3×3, 128] [128×128×3×3, 128]
//   Block 3:  [256×128×3×3, 256] [256×256×3×3, 256] × 2
//   Block 4:  [512×256×3×3, 512] [512×512×3×3, 512] × 3
//   Block 5:  [512×512×3×3, 512] × 4
static std::string write_zero_bin() {
  struct Spec {
    uint32_t oc, ic, k;
  };
  const std::vector<Spec> specs = {
      // Block 1
      {64, 3, 3},
      {64, 64, 3},
      // Block 2
      {128, 64, 3},
      {128, 128, 3},
      // Block 3
      {256, 128, 3},
      {256, 256, 3},
      {256, 256, 3},
      {256, 256, 3},
      // Block 4
      {512, 256, 3},
      {512, 512, 3},
      {512, 512, 3},
      {512, 512, 3},
      // Block 5
      {512, 512, 3},
      {512, 512, 3},
      {512, 512, 3},
      {512, 512, 3},
  };

  std::string path = (fs::temp_directory_path() / "vgg19_zero.bin").string();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);

  for (const auto &s : specs) {
    // Weight blob: oc × ic × k × k
    uint32_t w_count = s.oc * s.ic * s.k * s.k;
    out.write(reinterpret_cast<const char *>(&w_count), sizeof(w_count));
    std::vector<float> zeros(w_count, 0.0f);
    out.write(reinterpret_cast<const char *>(zeros.data()),
              w_count * sizeof(float));

    // Bias blob: oc
    uint32_t b_count = s.oc;
    out.write(reinterpret_cast<const char *>(&b_count), sizeof(b_count));
    std::vector<float> bzeros(b_count, 0.0f);
    out.write(reinterpret_cast<const char *>(bzeros.data()),
              b_count * sizeof(float));
  }
  return path;
}

// Build a black 224×224 RGB image.
Image black_image(int w = 224, int h = 224) {
  Image img;
  img.width = w;
  img.height = h;
  img.channels = 3;
  img.data.assign(static_cast<std::size_t>(w * h * 3), 0u);
  return img;
}

} // namespace

// ---------------------------------------------------------------------------
// Architecture tests
// ---------------------------------------------------------------------------

TEST(Vgg19Test, LoadWeightsDoesNotThrow) {
  auto engine = cpu_engine();
  std::string bp = write_zero_bin();
  Vgg19 net(engine);
  EXPECT_NO_THROW(net.load_weights(bp));
  EXPECT_TRUE(net.weights_loaded());
  fs::remove(bp);
}

TEST(Vgg19Test, ForwardBeforeLoadThrows) {
  auto engine = cpu_engine();
  Vgg19 net(engine);
  Image img = black_image();
  Tensor input = preprocess_image(img, engine);
  EXPECT_THROW(net.forward(input), std::logic_error);
}

TEST(Vgg19Test, GetFeatureMapBeforeForwardThrows) {
  auto engine = cpu_engine();
  std::string bp = write_zero_bin();
  Vgg19 net(engine);
  net.load_weights(bp);
  EXPECT_THROW(net.get_feature_map(VggLayer::relu1_1), std::logic_error);
  fs::remove(bp);
}

TEST(Vgg19Test, ForwardOutputShapes) {
  auto engine = cpu_engine();
  std::string bp = write_zero_bin();
  Vgg19 net(engine, 224, 224);
  net.load_weights(bp);

  Image img = black_image(224, 224);
  Tensor input = preprocess_image(img, engine);
  ASSERT_NO_THROW(net.forward(input));

  // Expected shapes: {1, channels, H, W} after respective pooling.
  struct Expected {
    VggLayer layer;
    int c, h, w;
  };
  const std::vector<Expected> expected = {
      {VggLayer::relu1_1, 64, 224, 224}, {VggLayer::relu2_1, 128, 112, 112},
      {VggLayer::relu3_1, 256, 56, 56},  {VggLayer::relu4_1, 512, 28, 28},
      {VggLayer::relu4_2, 512, 28, 28},  {VggLayer::relu5_1, 512, 14, 14},
  };

  for (const auto &e : expected) {
    const Tensor &fm = net.get_feature_map(e.layer);
    auto dims = fm.get_dims();
    ASSERT_EQ(dims.size(), 4u) << "layer=" << static_cast<int>(e.layer);
    EXPECT_EQ(dims[0], 1) << "N mismatch for layer "
                          << static_cast<int>(e.layer);
    EXPECT_EQ(dims[1], e.c)
        << "C mismatch for layer " << static_cast<int>(e.layer);
    EXPECT_EQ(dims[2], e.h)
        << "H mismatch for layer " << static_cast<int>(e.layer);
    EXPECT_EQ(dims[3], e.w)
        << "W mismatch for layer " << static_cast<int>(e.layer);
  }

  fs::remove(bp);
}

TEST(Vgg19Test, ForwardDeterminism) {
  auto engine = cpu_engine();
  std::string bp = write_zero_bin();
  Vgg19 net(engine, 224, 224);
  net.load_weights(bp);

  Image img = black_image(224, 224);
  Tensor input = preprocess_image(img, engine);

  net.forward(input);
  const Tensor &fm1 = net.get_feature_map(VggLayer::relu4_2);
  const float *ptr1 = fm1.get_data();
  auto dims = fm1.get_dims();
  const std::size_t n =
      static_cast<std::size_t>(dims[0] * dims[1] * dims[2] * dims[3]);
  std::vector<float> snap1(ptr1, ptr1 + n);

  net.forward(input);
  const Tensor &fm2 = net.get_feature_map(VggLayer::relu4_2);
  const float *ptr2 = fm2.get_data();
  std::vector<float> snap2(ptr2, ptr2 + n);

  EXPECT_EQ(snap1, snap2);
  fs::remove(bp);
}

TEST(Vgg19Test, ForwardWrongInputShapeThrows) {
  auto engine = cpu_engine();
  std::string bp = write_zero_bin();
  Vgg19 net(engine, 224, 224);
  net.load_weights(bp);

  // Input with wrong spatial size.
  Tensor bad_input({1, 3, 128, 128}, engine);
  EXPECT_THROW(net.forward(bad_input), std::invalid_argument);
  fs::remove(bp);
}
