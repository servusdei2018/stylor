#include "stylor/preprocessing.hpp"

#include <stdexcept>

namespace stylor {

namespace {

// ImageNet per-channel means used during VGG-19 training (BGR order).
constexpr float kMeanB = 103.939f;
constexpr float kMeanG = 116.779f;
constexpr float kMeanR = 123.680f;

} // namespace

Tensor preprocess_image(const Image &img, const dnnl::engine &engine) {
  if (img.width <= 0 || img.height <= 0) {
    throw std::invalid_argument(
        "preprocess_image: image has non-positive dimensions (" +
        std::to_string(img.width) + "x" + std::to_string(img.height) + ")");
  }
  if (img.channels != 3) {
    throw std::invalid_argument(
        "preprocess_image: expected 3-channel RGB image, got " +
        std::to_string(img.channels) + " channels");
  }

  const dnnl::memory::dim N = 1;
  const dnnl::memory::dim C = 3;
  const dnnl::memory::dim H = img.height;
  const dnnl::memory::dim W = img.width;

  // Allocate output: {1, 3, H, W} NCHW float tensor.
  Tensor output({N, C, H, W}, engine);
  float *dst = output.get_data();

  // Plane offsets for NCHW layout: channel c starts at c*H*W.
  const dnnl::memory::dim plane = H * W;
  float *bgr_plane[3] = {
      dst + 0 * plane, // B channel (index 0)
      dst + 1 * plane, // G channel (index 1)
      dst + 2 * plane, // R channel (index 2)
  };

  const uint8_t *src = img.data.data();

  for (int y = 0; y < img.height; ++y) {
    for (int x = 0; x < img.width; ++x) {
      const int src_idx = (y * img.width + x) * 3;
      const int dst_idx = y * img.width + x;

      // stb loads as RGB; swap to BGR and subtract ImageNet means.
      const float r = static_cast<float>(src[src_idx + 0]);
      const float g = static_cast<float>(src[src_idx + 1]);
      const float b = static_cast<float>(src[src_idx + 2]);

      bgr_plane[0][dst_idx] = b - kMeanB;
      bgr_plane[1][dst_idx] = g - kMeanG;
      bgr_plane[2][dst_idx] = r - kMeanR;
    }
  }

  return output;
}

} // namespace stylor
