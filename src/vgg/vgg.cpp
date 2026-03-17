#include "stylor/vgg.hpp"

#include "stylor/weight_loader.hpp"
#include <stdexcept>

namespace stylor {

// ---------------------------------------------------------------------------
// VGG-19 architecture: 13 convolutional layers across 5 blocks.
//
// Filters per block:
//   Block 1: 2 × [3→64]    pool
//   Block 2: 2 × [64→128]  pool
//   Block 3: 4 × [128→256] pool
//   Block 4: 4 × [256→512] pool
//   Block 5: 4 × [512→512] pool
//
// All conv layers: 3×3 kernel, stride 1, same-padding (pad=1).
// All pool layers: 2×2 max-pool, stride 2.
// ---------------------------------------------------------------------------

namespace {

// Cast VggLayer enum to a stable int key for the map.
inline int layer_key(VggLayer l) { return static_cast<int>(l); }

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Vgg19::Vgg19(const dnnl::engine &engine, int input_h, int input_w)
    : engine_(engine), stream_(engine_), input_h_(input_h), input_w_(input_w) {

  // Current activation memory passed block-to-block.
  // Starts as a placeholder; will be set by the first build_block call.
  dnnl::memory cur_mem;

  // --- Block 1: 2 convolutions, 3→64 channels
  // ------------------------------------
  build_block(1, {{3, 64, 3, 1, 1}, {64, 64, 3, 1, 1}}, cur_mem,
              {layer_key(VggLayer::relu1_1), -1});
  build_pool(cur_mem);

  // --- Block 2: 2 convolutions, 64→128 channels
  // ----------------------------------
  build_block(2, {{64, 128, 3, 1, 1}, {128, 128, 3, 1, 1}}, cur_mem,
              {layer_key(VggLayer::relu2_1), -1});
  build_pool(cur_mem);

  // --- Block 3: 4 convolutions, 128→256 channels
  // ---------------------------------
  build_block(3,
              {{128, 256, 3, 1, 1},
               {256, 256, 3, 1, 1},
               {256, 256, 3, 1, 1},
               {256, 256, 3, 1, 1}},
              cur_mem, {layer_key(VggLayer::relu3_1), -1, -1, -1});
  build_pool(cur_mem);

  // --- Block 4: 4 convolutions, 256→512 channels
  // ---------------------------------
  build_block(
      4,
      {{256, 512, 3, 1, 1},
       {512, 512, 3, 1, 1},
       {512, 512, 3, 1, 1},
       {512, 512, 3, 1, 1}},
      cur_mem,
      {layer_key(VggLayer::relu4_1), layer_key(VggLayer::relu4_2), -1, -1});
  build_pool(cur_mem);

  // --- Block 5: 4 convolutions, 512→512 channels
  // ---------------------------------
  build_block(5,
              {{512, 512, 3, 1, 1},
               {512, 512, 3, 1, 1},
               {512, 512, 3, 1, 1},
               {512, 512, 3, 1, 1}},
              cur_mem, {layer_key(VggLayer::relu5_1), -1, -1, -1});
  build_pool(cur_mem);
}

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

dnnl::memory Vgg19::make_weights_mem(int oc, int ic, int kh, int kw) {
  dnnl::memory::desc desc({oc, ic, kh, kw}, dnnl::memory::data_type::f32,
                          dnnl::memory::format_tag::oihw);
  return dnnl::memory(desc, engine_);
}

dnnl::memory Vgg19::make_bias_mem(int oc) {
  dnnl::memory::desc desc({oc}, dnnl::memory::data_type::f32,
                          dnnl::memory::format_tag::x);
  return dnnl::memory(desc, engine_);
}

void Vgg19::build_block(int block, const std::vector<ConvSpec> &convs,
                        dnnl::memory &in_out_mem, std::vector<int> captures) {
  // Compute current spatial dimensions from the number of pools before this
  // block (block index is 1-based).
  int pools_before = block - 1;
  int h = input_h_ >> pools_before; // divide by 2^pools
  int w = input_w_ >> pools_before;

  for (std::size_t i = 0; i < convs.size(); ++i) {
    const ConvSpec &cs = convs[i];

    // Memory descriptors.
    dnnl::memory::dims src_dims = {1, cs.in_channels, h, w};
    dnnl::memory::dims dst_dims = {1, cs.out_channels, h, w};
    dnnl::memory::dims w_dims = {cs.out_channels, cs.in_channels, cs.kernel,
                                 cs.kernel};
    dnnl::memory::dims b_dims = {cs.out_channels};
    dnnl::memory::dims strides = {cs.stride, cs.stride};
    dnnl::memory::dims padding = {cs.padding, cs.padding};

    auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::nchw);
    auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::nchw);
    auto w_md = dnnl::memory::desc(w_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::oihw);
    auto b_md = dnnl::memory::desc(b_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::x);

    // Convolution primitive descriptor (implicit bias).
    auto conv_pd = dnnl::convolution_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct, src_md, w_md, b_md, dst_md,
        strides, padding, padding);

    // ReLU primitive descriptor (in-place on dst).
    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::eltwise_relu, dst_md, dst_md,
        0.0f /*alpha: negative slope*/, 0.0f /*beta*/);

    LayerPrimitive lp;
    lp.weights_mem = dnnl::memory(conv_pd.weights_desc(), engine_);
    lp.bias_mem = dnnl::memory(conv_pd.bias_desc(), engine_);

    // For the very first layer of the network, create the src memory.
    // For subsequent layers, src is the previous layer's dst.
    if (conv_layers_.empty() && i == 0) {
      lp.src_mem = dnnl::memory(conv_pd.src_desc(), engine_);
    } else {
      lp.src_mem = in_out_mem; // chained from previous primitive
    }

    lp.dst_mem = dnnl::memory(conv_pd.dst_desc(), engine_);
    lp.conv = dnnl::convolution_forward(conv_pd);
    lp.relu = dnnl::eltwise_forward(relu_pd);

    lp.capture_key = captures[i];

    in_out_mem = lp.dst_mem; // next primitive reads from here

    const std::size_t idx = conv_layers_.size();
    conv_layers_.push_back(std::move(lp));
    exec_order_.emplace_back(false, idx);
  }
}

void Vgg19::build_pool(dnnl::memory &in_out_mem) {
  // 2×2 max-pool, stride 2 — halves spatial dimensions.
  auto src_md = in_out_mem.get_desc();

  // Infer output shape (halved spatial).
  auto src_dims = src_md.get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0], src_dims[1], src_dims[2] / 2,
                                 src_dims[3] / 2};
  auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::nchw);

  dnnl::memory::dims kernel = {2, 2};
  dnnl::memory::dims strides = {2, 2};
  dnnl::memory::dims dilation = {0, 0}; // no dilation
  dnnl::memory::dims padding = {0, 0};

  auto pool_pd = dnnl::pooling_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
      src_md, dst_md, strides, kernel, dilation, padding, padding);

  PoolPrimitive pp;
  pp.src_mem = in_out_mem;
  pp.dst_mem = dnnl::memory(pool_pd.dst_desc(), engine_);
  pp.pool = dnnl::pooling_forward(pool_pd);

  in_out_mem = pp.dst_mem;

  const std::size_t idx = pool_layers_.size();
  pool_layers_.push_back(std::move(pp));
  exec_order_.emplace_back(true, idx);
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

void Vgg19::load_weights(const std::string &path) {
  WeightLoader loader(path);

  for (auto &lp : conv_layers_) {
    auto w_dims = lp.weights_mem.get_desc().get_dims();
    const std::size_t w_count =
        static_cast<std::size_t>(w_dims[0]) * w_dims[1] * w_dims[2] * w_dims[3];
    loader.read_next(static_cast<float *>(lp.weights_mem.get_data_handle()),
                     w_count);

    const std::size_t b_count = static_cast<std::size_t>(w_dims[0]);
    loader.read_next(static_cast<float *>(lp.bias_mem.get_data_handle()),
                     b_count);
  }

  weights_loaded_ = true;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

void Vgg19::forward(const Tensor &input) {
  if (!weights_loaded_) {
    throw std::logic_error(
        "Vgg19::forward: call load_weights() before forward()");
  }

  // Validate input shape: must be {1, 3, input_h_, input_w_}.
  const auto &dims = input.get_dims();
  if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] != input_h_ ||
      dims[3] != input_w_) {
    throw std::invalid_argument("Vgg19::forward: input tensor shape mismatch. "
                                "Expected {1, 3, " +
                                std::to_string(input_h_) + ", " +
                                std::to_string(input_w_) + "}");
  }

  // Copy input data into the first conv layer's src memory.
  auto &first_src = conv_layers_.front().src_mem;
  const std::size_t input_bytes = input.get_memory().get_desc().get_size();
  std::memcpy(first_src.get_data_handle(), input.get_data(), input_bytes);

  // Execute all primitives in registered order.
  for (const auto &[is_pool, idx] : exec_order_) {
    if (!is_pool) {
      auto &lp = conv_layers_[idx];
      lp.conv.execute(stream_, {
                                   {DNNL_ARG_SRC, lp.src_mem},
                                   {DNNL_ARG_WEIGHTS, lp.weights_mem},
                                   {DNNL_ARG_BIAS, lp.bias_mem},
                                   {DNNL_ARG_DST, lp.dst_mem},
                               });
      lp.relu.execute(stream_, {
                                   {DNNL_ARG_SRC, lp.dst_mem},
                                   {DNNL_ARG_DST, lp.dst_mem},
                               });

      // Cache the output if this is a capture layer.
      if (lp.capture_key >= 0) {
        feature_map_mems_[lp.capture_key] = lp.dst_mem;
      }
    } else {
      auto &pp = pool_layers_[idx];
      pp.pool.execute(stream_, {
                                   {DNNL_ARG_SRC, pp.src_mem},
                                   {DNNL_ARG_DST, pp.dst_mem},
                               });
    }
  }

  stream_.wait();
  forward_done_ = true;
}

// ---------------------------------------------------------------------------
// Feature map retrieval
// ---------------------------------------------------------------------------

const Tensor &Vgg19::get_feature_map(VggLayer layer) const {
  if (!forward_done_) {
    throw std::logic_error(
        "Vgg19::get_feature_map: call forward() before get_feature_map()");
  }

  const int key = layer_key(layer);
  const auto it = feature_map_mems_.find(key);
  if (it == feature_map_mems_.end()) {
    throw std::out_of_range(
        "Vgg19::get_feature_map: requested layer is not a capture point "
        "(key=" +
        std::to_string(key) + ")");
  }

  // Build a Tensor view over the cached memory.  We return a reference to a
  // member so the lifetime is valid for the life of this object.
  //
  // We store the Tensor wrappers lazily in a mutable map.
  thread_local std::unordered_map<int, Tensor> cache;
  if (cache.find(key) == cache.end()) {
    const auto &mem = it->second;
    const auto dims = mem.get_desc().get_dims();
    std::vector<dnnl::memory::dim> tensor_dims(dims.begin(), dims.end());
    cache.emplace(key, Tensor(std::move(tensor_dims), engine_));
    // Point the new Tensor's memory handle at the existing buffer.
    cache.at(key).get_memory().set_data_handle(mem.get_data_handle());
  }

  return cache.at(key);
}

bool Vgg19::weights_loaded() const noexcept { return weights_loaded_; }
bool Vgg19::forward_done() const noexcept { return forward_done_; }

} // namespace stylor
