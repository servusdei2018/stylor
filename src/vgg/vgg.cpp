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
        engine_, dnnl::prop_kind::forward_training,
        dnnl::algorithm::convolution_direct, src_md, w_md, b_md, dst_md,
        strides, padding, padding);

    // Convolution backward data primitive descriptor
    auto conv_bw_data_pd = dnnl::convolution_backward_data::primitive_desc(
        engine_, dnnl::algorithm::convolution_direct, src_md, w_md, dst_md,
        strides, padding, padding, conv_pd);

    // ReLU primitive descriptor (in-place on dst).
    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_training,
        dnnl::algorithm::eltwise_relu, dst_md, dst_md,
        0.0f /*alpha: negative slope*/, 0.0f /*beta*/);

    // ReLU backward primitive descriptor
    auto relu_bw_pd = dnnl::eltwise_backward::primitive_desc(
        engine_, dnnl::algorithm::eltwise_relu_use_dst_for_bwd,
        dst_md /* diff_src */, dst_md /* diff_dst */,
        dst_md /* data_desc (using dst) */, 0.0f /* alpha */, 0.0f /* beta */,
        relu_pd);

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

    // Create memory and backward primitives
    lp.diff_src_mem = dnnl::memory(conv_bw_data_pd.diff_src_desc(), engine_);
    lp.diff_dst_mem = dnnl::memory(conv_bw_data_pd.diff_dst_desc(),
                                   engine_); // Same as relu diff_dst
    lp.relu_bw = dnnl::eltwise_backward(relu_bw_pd);
    lp.conv_bw_data = dnnl::convolution_backward_data(conv_bw_data_pd);

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
      engine_, dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max,
      src_md, dst_md, strides, kernel, dilation, padding, padding);

  auto pool_bw_pd = dnnl::pooling_backward::primitive_desc(
      engine_, dnnl::algorithm::pooling_max, src_md, dst_md, strides, kernel,
      dilation, padding, padding, pool_pd);

  PoolPrimitive pp;
  pp.src_mem = in_out_mem;
  pp.dst_mem = dnnl::memory(pool_pd.dst_desc(), engine_);
  pp.workspace_mem = dnnl::memory(pool_pd.workspace_desc(), engine_);
  pp.pool = dnnl::pooling_forward(pool_pd);

  pp.diff_src_mem = dnnl::memory(pool_bw_pd.diff_src_desc(), engine_);
  pp.diff_dst_mem = dnnl::memory(pool_bw_pd.diff_dst_desc(), engine_);
  pp.pool_bw = dnnl::pooling_backward(pool_bw_pd);

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
                                   {DNNL_ARG_WORKSPACE, pp.workspace_mem},
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

Tensor
Vgg19::backward(const std::unordered_map<VggLayer, Tensor> &loss_gradients) {
  if (!forward_done_) {
    throw std::logic_error("Vgg19::backward: call forward() before backward()");
  }

  // Helper macro to wrap dnnl::memory mapping logic securely.
  auto add_tensor_to_memory = [](dnnl::memory &mem, const Tensor &tensor) {
    float *mem_data = static_cast<float *>(mem.get_data_handle());
    const float *tensor_data = static_cast<const float *>(tensor.get_data());
    std::size_t size = mem.get_desc().get_size() / sizeof(float);
    for (std::size_t i = 0; i < size; ++i) {
      mem_data[i] += tensor_data[i];
    }
  };

  // Zero out all diff_dst and diff_src buffers before backward pass
  for (auto &lp : conv_layers_) {
    std::memset(lp.diff_dst_mem.get_data_handle(), 0,
                lp.diff_dst_mem.get_desc().get_size());
    std::memset(lp.diff_src_mem.get_data_handle(), 0,
                lp.diff_src_mem.get_desc().get_size());
  }
  for (auto &pp : pool_layers_) {
    std::memset(pp.diff_dst_mem.get_data_handle(), 0,
                pp.diff_dst_mem.get_desc().get_size());
    std::memset(pp.diff_src_mem.get_data_handle(), 0,
                pp.diff_src_mem.get_desc().get_size());
  }

  // Now, iterate through the layers in REVERSE order.
  for (auto it = exec_order_.rbegin(); it != exec_order_.rend(); ++it) {
    bool is_pool = it->first;
    std::size_t idx = it->second;

    // Check if there is an external gradient injecting into this step.
    if (!is_pool) {
      auto &lp = conv_layers_[idx];

      // If this layer was a capture point, there might be a loss gradient
      // mapped to it.
      if (lp.capture_key >= 0) {
        VggLayer vgg_enum = static_cast<VggLayer>(lp.capture_key);
        auto grad_it = loss_gradients.find(vgg_enum);
        if (grad_it != loss_gradients.end()) {
          // Add the external gradient to the current diff_dst
          add_tensor_to_memory(lp.diff_dst_mem, grad_it->second);
        }
      }

      // ReLU backward
      lp.relu_bw.execute(
          stream_, {
                       {DNNL_ARG_SRC, lp.dst_mem},
                       {DNNL_ARG_DIFF_DST, lp.diff_dst_mem},
                       {DNNL_ARG_DIFF_SRC, lp.diff_dst_mem} // in-place diff
                   });

      // Take diff_dst_mem -> compute into diff_src_mem
      lp.conv_bw_data.execute(stream_, {{DNNL_ARG_DIFF_DST, lp.diff_dst_mem},
                                        {DNNL_ARG_WEIGHTS, lp.weights_mem},
                                        {DNNL_ARG_DIFF_SRC, lp.diff_src_mem}});

      // Pass the gradient to the previous layer.
      if (it + 1 != exec_order_.rend()) {
        auto next_step = *(it + 1);
        dnnl::memory *prev_diff_dst = nullptr;
        if (next_step.first) {
          prev_diff_dst = &pool_layers_[next_step.second].diff_dst_mem;
        } else {
          prev_diff_dst = &conv_layers_[next_step.second].diff_dst_mem;
        }

        // Copy diff_src to the previous layer's diff_dst
        float *dst = static_cast<float *>(prev_diff_dst->get_data_handle());
        float *src = static_cast<float *>(lp.diff_src_mem.get_data_handle());
        std::memcpy(dst, src, prev_diff_dst->get_desc().get_size());
      }
    } else {
      auto &pp = pool_layers_[idx];

      pp.pool_bw.execute(stream_, {{DNNL_ARG_DIFF_DST, pp.diff_dst_mem},
                                   {DNNL_ARG_WORKSPACE, pp.workspace_mem},
                                   {DNNL_ARG_DIFF_SRC, pp.diff_src_mem}});

      // Pass the gradient to the previous layer
      if (it + 1 != exec_order_.rend()) {
        auto next_step = *(it + 1);
        dnnl::memory *prev_diff_dst = nullptr;
        if (next_step.first) {
          prev_diff_dst = &pool_layers_[next_step.second].diff_dst_mem;
        } else {
          prev_diff_dst = &conv_layers_[next_step.second].diff_dst_mem;
        }

        // Copy diff_src to the previous layer's diff_dst
        float *dst = static_cast<float *>(prev_diff_dst->get_data_handle());
        float *src = static_cast<float *>(pp.diff_src_mem.get_data_handle());
        std::memcpy(dst, src, prev_diff_dst->get_desc().get_size());
      }
    }
  }
  stream_.wait();

  // The gradient w.r.t the input image is now in the first conv
  // layer's diff_src_mem
  auto diff_src_desc = conv_layers_.front().diff_src_mem.get_desc();
  auto diff_src_dims = diff_src_desc.get_dims();
  std::vector<dnnl::memory::dim> dims(diff_src_dims.begin(),
                                      diff_src_dims.end());

  Tensor img_grad(std::move(dims), engine_);
  std::memcpy(img_grad.get_data(),
              conv_layers_.front().diff_src_mem.get_data_handle(),
              diff_src_desc.get_size());

  return img_grad;
}

} // namespace stylor
