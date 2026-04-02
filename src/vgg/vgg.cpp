#include "stylor/vgg.hpp"

#include "stylor/weight_loader.hpp"
#include <cstring>
#include <stdexcept>

namespace stylor {

// Cast VggLayer enum to a stable int key for the map.
inline int layer_key(VggLayer l) { return static_cast<int>(l); }

Vgg19::Vgg19(const dnnl::engine &engine, int input_h, int input_w)
    : engine_(engine), input_h_(input_h), input_w_(input_w) {

  dnnl::memory cur_mem;

  build_block(1, {{3, 64, 3, 1, 1}, {64, 64, 3, 1, 1}}, cur_mem,
              {layer_key(VggLayer::relu1_1), -1});
  build_pool(cur_mem);

  build_block(2, {{64, 128, 3, 1, 1}, {128, 128, 3, 1, 1}}, cur_mem,
              {layer_key(VggLayer::relu2_1), -1});
  build_pool(cur_mem);

  build_block(3,
              {{128, 256, 3, 1, 1},
               {256, 256, 3, 1, 1},
               {256, 256, 3, 1, 1},
               {256, 256, 3, 1, 1}},
              cur_mem, {layer_key(VggLayer::relu3_1), -1, -1, -1});
  build_pool(cur_mem);

  build_block(
      4,
      {{256, 512, 3, 1, 1},
       {512, 512, 3, 1, 1},
       {512, 512, 3, 1, 1},
       {512, 512, 3, 1, 1}},
      cur_mem,
      {layer_key(VggLayer::relu4_1), layer_key(VggLayer::relu4_2), -1, -1});
  build_pool(cur_mem);

  build_block(5,
              {{512, 512, 3, 1, 1},
               {512, 512, 3, 1, 1},
               {512, 512, 3, 1, 1},
               {512, 512, 3, 1, 1}},
              cur_mem, {layer_key(VggLayer::relu5_1), -1, -1, -1});
  build_pool(cur_mem);
}

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
  int pools_before = block - 1;
  int h = input_h_ >> pools_before;
  int w = input_w_ >> pools_before;

  for (std::size_t i = 0; i < convs.size(); ++i) {
    const ConvSpec &cs = convs[i];
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

    auto conv_pd = dnnl::convolution_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_training,
        dnnl::algorithm::convolution_direct, src_md, w_md, b_md, dst_md,
        strides, padding, padding);

    auto conv_bw_data_pd = dnnl::convolution_backward_data::primitive_desc(
        engine_, dnnl::algorithm::convolution_direct, src_md, w_md, dst_md,
        strides, padding, padding, conv_pd);

    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_training,
        dnnl::algorithm::eltwise_relu, dst_md, dst_md, 0.0f, 0.0f);

    // NOTE: eltwise_relu_use_dst_for_bwd requires DNNL_ARG_DST (the forward
    // output) at execution time — NOT DNNL_ARG_SRC. See backward() below.
    auto relu_bw_pd = dnnl::eltwise_backward::primitive_desc(
        engine_, dnnl::algorithm::eltwise_relu_use_dst_for_bwd, dst_md, dst_md,
        dst_md, 0.0f, 0.0f, relu_pd);

    LayerPrimitive lp;
    lp.weights_mem = dnnl::memory(conv_pd.weights_desc(), engine_);
    lp.bias_mem = dnnl::memory(conv_pd.bias_desc(), engine_);
    lp.src_mem = (conv_layers_.empty() && i == 0)
                     ? dnnl::memory(conv_pd.src_desc(), engine_)
                     : in_out_mem;
    lp.dst_mem = dnnl::memory(conv_pd.dst_desc(), engine_);
    lp.conv = dnnl::convolution_forward(conv_pd);
    lp.relu = dnnl::eltwise_forward(relu_pd);
    lp.capture_key = captures[i];
    lp.diff_src_mem = dnnl::memory(conv_bw_data_pd.diff_src_desc(), engine_);
    lp.diff_dst_mem = dnnl::memory(conv_bw_data_pd.diff_dst_desc(), engine_);
    lp.relu_bw = dnnl::eltwise_backward(relu_bw_pd);
    lp.conv_bw_data = dnnl::convolution_backward_data(conv_bw_data_pd);

    in_out_mem = lp.dst_mem;
    exec_order_.emplace_back(false, conv_layers_.size());
    conv_layers_.push_back(std::move(lp));
  }
}

void Vgg19::build_pool(dnnl::memory &in_out_mem) {
  auto src_md = in_out_mem.get_desc();
  auto src_dims = src_md.get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0], src_dims[1], src_dims[2] / 2,
                                 src_dims[3] / 2};
  auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::nchw);

  auto pool_pd = dnnl::pooling_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max,
      src_md, dst_md, {2, 2}, {2, 2}, {0, 0}, {0, 0}, {0, 0});
  auto pool_bw_pd = dnnl::pooling_backward::primitive_desc(
      engine_, dnnl::algorithm::pooling_max, src_md, dst_md, {2, 2}, {2, 2},
      {0, 0}, {0, 0}, {0, 0}, pool_pd);

  PoolPrimitive pp;
  pp.src_mem = in_out_mem;
  pp.dst_mem = dnnl::memory(pool_pd.dst_desc(), engine_);
  pp.workspace_mem = dnnl::memory(pool_pd.workspace_desc(), engine_);
  pp.pool = dnnl::pooling_forward(pool_pd);
  pp.diff_src_mem = dnnl::memory(pool_bw_pd.diff_src_desc(), engine_);
  pp.diff_dst_mem = dnnl::memory(pool_bw_pd.diff_dst_desc(), engine_);
  pp.pool_bw = dnnl::pooling_backward(pool_bw_pd);

  in_out_mem = pp.dst_mem;
  exec_order_.emplace_back(true, pool_layers_.size());
  pool_layers_.push_back(std::move(pp));
}

void Vgg19::load_weights(const std::string &path) {
  WeightLoader loader(path);
  dnnl::stream stream(engine_);

  for (auto &lp : conv_layers_) {
    auto w_dims = lp.weights_mem.get_desc().get_dims();
    std::size_t w_count =
        (std::size_t)w_dims[0] * w_dims[1] * w_dims[2] * w_dims[3];
    dnnl::memory load_w_mem(
        {w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw},
        engine_);
    loader.read_next(static_cast<float *>(load_w_mem.get_data_handle()),
                     w_count);
    dnnl::reorder(load_w_mem, lp.weights_mem)
        .execute(stream,
                 {{DNNL_ARG_FROM, load_w_mem}, {DNNL_ARG_TO, lp.weights_mem}});
    loader.read_next(static_cast<float *>(lp.bias_mem.get_data_handle()),
                     (std::size_t)w_dims[0]);
  }
  stream.wait();
  weights_loaded_ = true;
}

void Vgg19::forward(const Tensor &input, dnnl::stream &stream) {
  if (!weights_loaded_)
    throw std::logic_error("Vgg19::forward: call load_weights() first");

  dnnl::reorder(input.get_memory(), conv_layers_.front().src_mem)
      .execute(stream, {{DNNL_ARG_FROM, input.get_memory()},
                        {DNNL_ARG_TO, conv_layers_.front().src_mem}});

  for (const auto &[is_pool, idx] : exec_order_) {
    if (!is_pool) {
      auto &lp = conv_layers_[idx];
      lp.conv.execute(stream, {{DNNL_ARG_SRC, lp.src_mem},
                               {DNNL_ARG_WEIGHTS, lp.weights_mem},
                               {DNNL_ARG_BIAS, lp.bias_mem},
                               {DNNL_ARG_DST, lp.dst_mem}});
      lp.relu.execute(stream,
                      {{DNNL_ARG_SRC, lp.dst_mem}, {DNNL_ARG_DST, lp.dst_mem}});
      if (lp.capture_key >= 0)
        feature_map_mems_[lp.capture_key] = lp.dst_mem;
    } else {
      auto &pp = pool_layers_[idx];
      pp.pool.execute(stream, {{DNNL_ARG_SRC, pp.src_mem},
                               {DNNL_ARG_DST, pp.dst_mem},
                               {DNNL_ARG_WORKSPACE, pp.workspace_mem}});
    }
  }
  stream.wait();
  forward_done_ = true;
}

const Tensor &Vgg19::get_feature_map(VggLayer layer) const {
  if (!forward_done_)
    throw std::logic_error("Vgg19::get_feature_map: call forward() first");
  int key = layer_key(layer);
  if (feature_map_cache_.find(key) == feature_map_cache_.end()) {
    auto &mem = feature_map_mems_.at(key);
    auto dims = mem.get_desc().get_dims();
    feature_map_cache_.emplace(key,
                               Tensor({dims.begin(), dims.end()}, engine_));
    feature_map_cache_.at(key).get_memory().set_data_handle(
        mem.get_data_handle());
  }
  return feature_map_cache_.at(key);
}

bool Vgg19::weights_loaded() const noexcept { return weights_loaded_; }
bool Vgg19::forward_done() const noexcept { return forward_done_; }

Tensor
Vgg19::backward(const std::unordered_map<VggLayer, Tensor> &loss_gradients,
                dnnl::stream &stream) {
  if (!forward_done_)
    throw std::logic_error("Vgg19::backward: call forward() first");

  // Accumulate a loss gradient into a diff_dst buffer.
  auto add_to = [](dnnl::memory &mem, const Tensor &t) {
    float *d = static_cast<float *>(mem.get_data_handle());
    const float *s = t.get_data();
    for (std::size_t i = 0; i < mem.get_desc().get_size() / 4; ++i)
      d[i] += s[i];
  };

  // Zero all gradient buffers before backward sweep.
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

  for (auto it = exec_order_.rbegin(); it != exec_order_.rend(); ++it) {
    if (!it->first) {
      auto &lp = conv_layers_[it->second];

      // Inject loss gradient at capture layers.
      if (lp.capture_key >= 0 &&
          loss_gradients.count(static_cast<VggLayer>(lp.capture_key)))
        add_to(lp.diff_dst_mem,
               loss_gradients.at(static_cast<VggLayer>(lp.capture_key)));

      // ReLU backward: eltwise_relu_use_dst_for_bwd requires DNNL_ARG_DST
      // (the saved forward output), NOT DNNL_ARG_SRC.
      lp.relu_bw.execute(stream,
                         {{DNNL_ARG_DST, lp.dst_mem}, // forward output
                          {DNNL_ARG_DIFF_DST, lp.diff_dst_mem},
                          {DNNL_ARG_DIFF_SRC, lp.diff_dst_mem}}); // in-place

      // Conv backward (data only — weights already fixed during training).
      lp.conv_bw_data.execute(stream, {{DNNL_ARG_DIFF_DST, lp.diff_dst_mem},
                                       {DNNL_ARG_WEIGHTS, lp.weights_mem},
                                       {DNNL_ARG_DIFF_SRC, lp.diff_src_mem}});

      // Propagate gradient to the previous layer's diff_dst.
      if (it + 1 != exec_order_.rend()) {
        auto next = *(it + 1);
        dnnl::memory &prev = next.first
                                 ? pool_layers_[next.second].diff_dst_mem
                                 : conv_layers_[next.second].diff_dst_mem;
        dnnl::reorder(lp.diff_src_mem, prev)
            .execute(stream,
                     {{DNNL_ARG_FROM, lp.diff_src_mem}, {DNNL_ARG_TO, prev}});
      }
    } else {
      auto &pp = pool_layers_[it->second];
      pp.pool_bw.execute(stream, {{DNNL_ARG_DIFF_DST, pp.diff_dst_mem},
                                  {DNNL_ARG_WORKSPACE, pp.workspace_mem},
                                  {DNNL_ARG_DIFF_SRC, pp.diff_src_mem}});

      if (it + 1 != exec_order_.rend()) {
        auto next = *(it + 1);
        dnnl::memory &prev = next.first
                                 ? pool_layers_[next.second].diff_dst_mem
                                 : conv_layers_[next.second].diff_dst_mem;
        dnnl::reorder(pp.diff_src_mem, prev)
            .execute(stream,
                     {{DNNL_ARG_FROM, pp.diff_src_mem}, {DNNL_ARG_TO, prev}});
      }
    }
  }
  stream.wait();

  // Copy the final gradient w.r.t. the input image into a new Tensor.
  auto d_src = conv_layers_.front().diff_src_mem;
  auto raw_dims = d_src.get_desc().get_dims();
  Tensor grad({raw_dims.begin(), raw_dims.end()}, engine_);
  dnnl::reorder(d_src, grad.get_memory())
      .execute(stream,
               {{DNNL_ARG_FROM, d_src}, {DNNL_ARG_TO, grad.get_memory()}});
  stream.wait();
  return grad;
}

} // namespace stylor
