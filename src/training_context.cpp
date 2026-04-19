#include "stylor/training_context.hpp"
#include <stdexcept>

namespace stylor {

GramPrimitives::GramPrimitives(dnnl::memory::dim C, dnnl::memory::dim HW,
                               const dnnl::engine &engine)
    : src_md({C, HW}, dnnl::memory::data_type::f32, {HW, 1}),
      wei_md({HW, C}, dnnl::memory::data_type::f32, {1, HW}),
      dst_md({C, C}, dnnl::memory::data_type::f32, {C, 1}), C(C), HW(HW),
      prim(dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md)) {}

StyleBackwardPrimitives::StyleBackwardPrimitives(dnnl::memory::dim C,
                                                 dnnl::memory::dim HW,
                                                 const dnnl::engine &engine)
    : src_md({C, C}, dnnl::memory::data_type::f32, {C, 1}),
      wei_md({C, HW}, dnnl::memory::data_type::f32, {HW, 1}),
      dst_md({C, HW}, dnnl::memory::data_type::f32, {HW, 1}), C(C), HW(HW),
      prim(dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md)) {}

TrainingContext::TrainingContext(const dnnl::engine &engine, int image_h,
                                 int image_w) {
  // VGG-19 architecture: spatial resolution halves after each max-pool block.
  struct LayerSpec {
    VggLayer layer;
    int channels;
    int spatial_divisor;
  };
  const LayerSpec specs[] = {
      {VggLayer::relu1_1, 64, 1},   {VggLayer::relu2_1, 128, 2},
      {VggLayer::relu3_1, 256, 4},  {VggLayer::relu4_1, 512, 8},
      {VggLayer::relu5_1, 512, 16},
  };

  for (const auto &s : specs) {
    const auto C = static_cast<dnnl::memory::dim>(s.channels);
    const auto H = static_cast<dnnl::memory::dim>(image_h / s.spatial_divisor);
    const auto W = static_cast<dnnl::memory::dim>(image_w / s.spatial_divisor);
    const auto HW = H * W;
    int key = static_cast<int>(s.layer);

    gram_cache_.emplace(key, GramPrimitives(C, HW, engine));
    style_bw_cache_.emplace(key, StyleBackwardPrimitives(C, HW, engine));
  }
}

const GramPrimitives &TrainingContext::gram(VggLayer layer) const {
  auto it = gram_cache_.find(static_cast<int>(layer));
  if (it == gram_cache_.end())
    throw std::logic_error("TrainingContext::gram: layer not cached");
  return it->second;
}

const StyleBackwardPrimitives &TrainingContext::style_bw(VggLayer layer) const {
  auto it = style_bw_cache_.find(static_cast<int>(layer));
  if (it == style_bw_cache_.end())
    throw std::logic_error("TrainingContext::style_bw: layer not cached");
  return it->second;
}

} // namespace stylor
