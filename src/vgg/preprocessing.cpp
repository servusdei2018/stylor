#include "stylor/preprocessing.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace stylor {

namespace {

// ImageNet per-channel means in RGB order, matching torchvision VGG19
// convention. PyTorch ImageNet means: [0.485, 0.456, 0.406] * 255. The exported
// VGG19 weights (export_vgg19.py / torchvision) expect RGB input with these
// means subtracted. Plane 0 = R, plane 1 = G, plane 2 = B.
constexpr float kMeanR = 123.680f;
constexpr float kMeanG = 116.779f;
constexpr float kMeanB = 103.939f;

// Build a {1,3,1,1} f32/nchw bias memory: plane 0 = p0, plane 1 = p1, plane 2 =
// p2.
dnnl::memory make_bias_mem(const dnnl::engine &engine, float p0, float p1,
                           float p2) {
  dnnl::memory::desc md({1, 3, 1, 1}, dnnl::memory::data_type::f32,
                        dnnl::memory::format_tag::nchw);
  dnnl::memory mem(md, engine);
  float *ptr = static_cast<float *>(mem.get_data_handle());
  ptr[0] = p0;
  ptr[1] = p1;
  ptr[2] = p2;
  return mem;
}

} // namespace

// ---------------------------------------------------------------------------
// preprocess_image -- stream overload
//
// Converts a u8 RGB image (stb/nhwc interleaved) into a f32 NCHW tensor with
// ImageNet means subtracted. Plane order: 0=R, 1=G, 2=B (RGB, matching the
// torchvision VGG19 weights produced by export_vgg19.py).
//
// Two oneDNN primitive passes:
//   Pass A) dnnl::reorder u8/nhwc {N,C,H,W} -> f32/nchw {N,C,H,W}
//           Both descriptors share the same logical shape; only the memory
//           strides differ (interleaved vs planar). The JIT kernel handles
//           de-interleaving and the u8->f32 cast with no extra buffer.
//           nhwc channel order maps naturally: R->plane0, G->plane1, B->plane2.
//
//   Pass B) broadcast dnnl::binary ADD subtracts [-kMeanR, -kMeanG, -kMeanB]
//           from planes [0, 1, 2] respectively. The {1,3,1,1} bias tensor is
//           broadcast across the full spatial dimension.
// ---------------------------------------------------------------------------
Tensor preprocess_image(const Image &img, const dnnl::engine &engine,
                        dnnl::stream &stream) {
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

  // Pass A: u8/nhwc -> f32/nchw.
  // Both descriptors use the same logical {N,C,H,W} shape.
  // nhwc strides:  { C*H*W, 1, C*W, C }  (interleaved channel-last)
  // nchw strides:  { C*H*W, H*W, W, 1 }  (planar channel-first)
  // The JIT reorder kernel handles the de-interleave and u8->f32 cast in one
  // pass. img.data is aliased directly -- zero extra copy.
  dnnl::memory::desc src_md({N, C, H, W}, dnnl::memory::data_type::u8,
                            dnnl::memory::format_tag::nhwc);
  dnnl::memory src_mem(src_md, engine, const_cast<uint8_t *>(img.data.data()));

  Tensor output({N, C, H, W}, engine);

  dnnl::reorder(src_mem, output.get_memory())
      .execute(stream,
               {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, output.get_memory()}});

  // Pass B: subtract ImageNet means via in-place broadcast binary_add.
  // [-kMeanR, -kMeanG, -kMeanB] maps to planes [0, 1, 2] (RGB order).
  dnnl::memory bias_mem = make_bias_mem(engine, -kMeanR, -kMeanG, -kMeanB);

  dnnl::binary::primitive_desc binary_pd(
      engine, dnnl::algorithm::binary_add, output.get_memory().get_desc(),
      bias_mem.get_desc(), output.get_memory().get_desc());

  dnnl::binary(binary_pd).execute(stream,
                                  {{DNNL_ARG_SRC_0, output.get_memory()},
                                   {DNNL_ARG_SRC_1, bias_mem},
                                   {DNNL_ARG_DST, output.get_memory()}});

  stream.wait();
  return output;
}

// Convenience overload -- constructs a temporary stream internally.
Tensor preprocess_image(const Image &img, const dnnl::engine &engine) {
  dnnl::stream stream(engine);
  return preprocess_image(img, engine, stream);
}

// ---------------------------------------------------------------------------
// postprocess_image -- stream overload
//
// Inverse of preprocess_image. Three oneDNN primitive passes:
//   Pass A) broadcast binary ADD to restore ImageNet means (RGB plane order)
//   Pass B) eltwise_clip to clamp values to [0, 255]
//   Pass C) dnnl::reorder f32/nchw -> u8/nhwc
//           The natural mapping (plane0=R -> channel0, etc.) produces the
//           R,G,B interleaved layout that stb_image_write expects directly.
// ---------------------------------------------------------------------------
Image postprocess_image(const Tensor &tensor, const dnnl::engine &engine,
                        dnnl::stream &stream) {
  auto dims = tensor.get_dims(); // {N, C, H, W}
  if (dims.size() != 4 || dims[1] != 3) {
    throw std::invalid_argument(
        "postprocess_image: expected 4D tensor with 3 channels.");
  }

  const dnnl::memory::dim N = dims[0];
  const dnnl::memory::dim C = dims[1];
  const dnnl::memory::dim H = dims[2];
  const dnnl::memory::dim W = dims[3];

  // Pass A: restore ImageNet means (in-place broadcast binary_add, RGB order).
  dnnl::memory bias_mem = make_bias_mem(engine, kMeanR, kMeanG, kMeanB);

  dnnl::binary::primitive_desc add_pd(
      engine, dnnl::algorithm::binary_add, tensor.get_memory().get_desc(),
      bias_mem.get_desc(), tensor.get_memory().get_desc());

  dnnl::binary(add_pd).execute(stream, {{DNNL_ARG_SRC_0, tensor.get_memory()},
                                        {DNNL_ARG_SRC_1, bias_mem},
                                        {DNNL_ARG_DST, tensor.get_memory()}});

  // Pass B: clamp to [0, 255] via eltwise_clip (in-place).
  dnnl::eltwise_forward::primitive_desc clip_pd(
      engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_clip,
      tensor.get_memory().get_desc(), tensor.get_memory().get_desc(),
      /*alpha=*/0.0f, /*beta=*/255.0f);

  dnnl::eltwise_forward(clip_pd).execute(stream,
                                         {{DNNL_ARG_SRC, tensor.get_memory()},
                                          {DNNL_ARG_DST, tensor.get_memory()}});

  // Pass C: reorder f32/nchw -> u8/nhwc (re-interleave + saturating cast).
  // Both share {N,C,H,W} logical shape; nhwc tag produces interleaved output.
  // Plane 0 (R) -> byte 0, plane 1 (G) -> byte 1, plane 2 (B) -> byte 2:
  // exactly the R,G,B layout stb_image_write expects.
  Image out_img;
  out_img.width = static_cast<int>(W);
  out_img.height = static_cast<int>(H);
  out_img.channels = 3;
  out_img.data.resize(static_cast<std::size_t>(H * W * C));

  dnnl::memory::desc dst_md({N, C, H, W}, dnnl::memory::data_type::u8,
                            dnnl::memory::format_tag::nhwc);
  dnnl::memory dst_mem(dst_md, engine, out_img.data.data());

  dnnl::reorder(tensor.get_memory(), dst_mem)
      .execute(stream,
               {{DNNL_ARG_FROM, tensor.get_memory()}, {DNNL_ARG_TO, dst_mem}});

  stream.wait();
  return out_img;
}

// Convenience overload -- constructs a temporary stream internally.
Image postprocess_image(const Tensor &tensor, const dnnl::engine &engine) {
  dnnl::stream stream(engine);
  return postprocess_image(tensor, engine, stream);
}

// ---------------------------------------------------------------------------
// resize_tensor -- oneDNN bilinear resampling, stays in f32/NCHW
// ---------------------------------------------------------------------------
Tensor resize_tensor(const Tensor &tensor, int height, int width,
                     const dnnl::engine &engine, dnnl::stream &stream) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(
        "resize_tensor: target dimensions must be positive (got " +
        std::to_string(width) + "x" + std::to_string(height) + ")");
  }

  auto src_dims = tensor.get_dims(); // {N, C, H, W}
  if (src_dims[2] == height && src_dims[3] == width)
    return tensor; // no-op

  Tensor output({src_dims[0], src_dims[1],
                 static_cast<dnnl::memory::dim>(height),
                 static_cast<dnnl::memory::dim>(width)},
                engine);

  dnnl::resampling_forward::primitive_desc rs_pd(
      engine, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::resampling_linear, tensor.get_memory().get_desc(),
      output.get_memory().get_desc());

  dnnl::resampling_forward(rs_pd).execute(
      stream, {{DNNL_ARG_SRC, tensor.get_memory()},
               {DNNL_ARG_DST, output.get_memory()}});

  stream.wait();
  return output;
}

} // namespace stylor
