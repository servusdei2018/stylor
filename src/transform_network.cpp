#include "stylor/transform_network.hpp"
#include "stylor/safetensors_io.hpp"
#include <iostream>

namespace stylor {

TransformNetwork::TransformNetwork(const dnnl::engine &engine, int input_h,
                                   int input_w)
    : engine_(engine), stream_(engine_), input_h_(input_h), input_w_(input_w) {

  // Create initial input memory format (NCHW)
  dnnl::memory::dims input_dims = {1, 3, input_h, input_w};
  dnnl::memory::desc input_md(input_dims, dnnl::memory::data_type::f32,
                              dnnl::memory::format_tag::nchw);
  dnnl::memory fwd_mem(input_md, engine_);
  dnnl::memory bwd_mem(input_md, engine_);
  input_mempair_ = {fwd_mem, bwd_mem};

  MemPair current_mem = input_mempair_;

  // Initial convolution.
  current_mem = create_conv("conv1", 3, 32, 9, 1, 4, current_mem);
  current_mem = create_norm("norm1", 32, current_mem);
  current_mem = create_relu(current_mem);

  // First downsampling block.
  current_mem = create_conv("conv2", 32, 64, 3, 2, 1, current_mem);
  current_mem = create_norm("norm2", 64, current_mem);
  current_mem = create_relu(current_mem);

  // Second downsampling block.
  current_mem = create_conv("conv3", 64, 128, 3, 2, 1, current_mem);
  current_mem = create_norm("norm3", 128, current_mem);
  current_mem = create_relu(current_mem);

  // Residual blocks.
  for (int i = 1; i <= 5; ++i) {
    current_mem = create_resblock("res" + std::to_string(i), 128, current_mem);
  }

  // First upsampling block.
  current_mem = create_resample(current_mem, 2.0f);
  current_mem = create_conv("upconv1", 128, 64, 3, 1, 1, current_mem);
  current_mem = create_norm("upnorm1", 64, current_mem);
  current_mem = create_relu(current_mem);

  // Second upsampling block.
  current_mem = create_resample(current_mem, 2.0f);
  current_mem = create_conv("upconv2", 64, 32, 3, 1, 1, current_mem);
  current_mem = create_norm("upnorm2", 32, current_mem);
  current_mem = create_relu(current_mem);

  // Output layer.
  current_mem = create_conv("conv_out", 32, 3, 9, 1, 4, current_mem);

  output_mempair_ = current_mem;
  output_tensor_ = std::make_unique<Tensor>(
      output_mempair_.fwd.get_desc().get_dims(), engine_);
}

void TransformNetwork::load_weights(const std::string &path) {
  SafetensorsLoader loader(path);
  for (auto &pair : parameters_) {
    const std::string &name = pair.first;
    ParamDescriptor &desc = pair.second;

    if (!loader.has_tensor(name)) {
      std::cerr << "Warning: weights not found for " << name << "\n";
      continue;
    }

    std::size_t elem_count = 1;
    for (int d : desc.shape)
      elem_count *= d;

    loader.load_tensor(name, static_cast<float *>(desc.mem.get_data_handle()),
                       elem_count);
  }
}

void TransformNetwork::save_weights(const std::string &path) const {
  SafetensorsWriter writer(path);
  for (const auto &pair : parameters_) {
    const std::string &name = pair.first;
    const ParamDescriptor &desc = pair.second;

    std::size_t elem_count = 1;
    for (int d : desc.shape)
      elem_count *= d;

    writer.add_tensor(name, desc.shape,
                      static_cast<const float *>(desc.mem.get_data_handle()),
                      elem_count);
  }
  writer.write();
}

void TransformNetwork::forward(const Tensor &input) {
  // Inject input into the first layer's fwd_mem
  float *fwd_data = static_cast<float *>(input_mempair_.fwd.get_data_handle());
  const float *in_data = static_cast<const float *>(input.get_data());
  std::memcpy(fwd_data, in_data, input.get_memory().get_desc().get_size());

  for (auto &prim : pipeline_) {
    prim();
  }
  stream_.wait();

  // Extract from output layer's fwd_mem
  float *out_data = static_cast<float *>(output_tensor_->get_data());
  const float *term_fwd =
      static_cast<const float *>(output_mempair_.fwd.get_data_handle());
  std::memcpy(out_data, term_fwd,
              output_tensor_->get_memory().get_desc().get_size());
}

void TransformNetwork::backward(const Tensor &grad_output) {
  // Inject gradient
  float *out_bwd = static_cast<float *>(output_mempair_.bwd.get_data_handle());
  const float *external_grad =
      static_cast<const float *>(grad_output.get_data());
  std::memcpy(out_bwd, external_grad,
              grad_output.get_memory().get_desc().get_size());

  for (auto it = backward_pipeline_.rbegin(); it != backward_pipeline_.rend();
       ++it) {
    (*it)();
  }
  stream_.wait();
}

const Tensor &TransformNetwork::get_output() const { return *output_tensor_; }

// ---------------------------------------------------------
// Helper implementations

TransformNetwork::MemPair TransformNetwork::create_conv(const std::string &name,
                                                        int ic, int oc,
                                                        int kernel, int stride,
                                                        int padding,
                                                        MemPair src_mem) {

  dnnl::memory::dims weights_dims = {oc, ic, kernel, kernel};
  dnnl::memory::dims bias_dims = {oc};
  dnnl::memory::dims strides = {stride, stride};
  dnnl::memory::dims padding_l = {padding, padding};
  dnnl::memory::dims padding_r = {padding, padding};

  dnnl::memory::desc weights_md(weights_dims, dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::oihw);
  dnnl::memory::desc bias_md(bias_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::x);

  auto src_md = src_mem.fwd.get_desc();
  auto src_dims = src_md.get_dims();
  dnnl::memory::dim out_h = (src_dims[2] + 2 * padding - kernel) / stride + 1;
  dnnl::memory::dim out_w = (src_dims[3] + 2 * padding - kernel) / stride + 1;
  dnnl::memory::desc dst_md({src_dims[0], oc, out_h, out_w},
                            dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::any);

  dnnl::memory weights_mem(weights_md, engine_);
  dnnl::memory bias_mem(bias_md, engine_);
  dnnl::memory diff_weights_mem(weights_md, engine_);
  dnnl::memory diff_bias_mem(bias_md, engine_);

  parameters_[name + ".weight"] = {
      weights_mem, {(int)oc, (int)ic, kernel, kernel}, diff_weights_mem};
  parameters_[name + ".bias"] = {bias_mem, {(int)oc}, diff_bias_mem};

  auto conv_pd = dnnl::convolution_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_training,
      dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
      strides, padding_l, padding_r);

  auto conv_bw_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
      engine_, dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md,
      conv_pd.dst_desc(), strides, padding_l, padding_r, conv_pd);

  auto conv_bw_data_pd = dnnl::convolution_backward_data::primitive_desc(
      engine_, dnnl::algorithm::convolution_direct, src_md, weights_md,
      conv_pd.dst_desc(), strides, padding_l, padding_r, conv_pd);

  dnnl::memory dst_mem(conv_pd.dst_desc(), engine_);
  dnnl::memory diff_dst_mem(conv_pd.dst_desc(), engine_);

  dnnl::convolution_forward conv(conv_pd);
  dnnl::convolution_backward_weights conv_bw_w(conv_bw_weights_pd);
  dnnl::convolution_backward_data conv_bw_d(conv_bw_data_pd);

  pipeline_.push_back([=, this]() mutable {
    conv.execute(stream_, {{DNNL_ARG_SRC, src_mem.fwd},
                           {DNNL_ARG_WEIGHTS, weights_mem},
                           {DNNL_ARG_BIAS, bias_mem},
                           {DNNL_ARG_DST, dst_mem}});
  });

  backward_pipeline_.push_back([=, this]() mutable {
    conv_bw_w.execute(stream_, {{DNNL_ARG_SRC, src_mem.fwd},
                                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                                {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem},
                                {DNNL_ARG_DIFF_BIAS, diff_bias_mem}});
    conv_bw_d.execute(stream_, {{DNNL_ARG_DIFF_DST, diff_dst_mem},
                                {DNNL_ARG_WEIGHTS, weights_mem},
                                {DNNL_ARG_DIFF_SRC, src_mem.bwd}});
  });

  return {dst_mem, diff_dst_mem};
}

TransformNetwork::MemPair TransformNetwork::create_norm(const std::string &name,
                                                        int channels,
                                                        MemPair src_mem) {
  dnnl::memory::dims stat_dims = {channels};
  dnnl::memory::desc stat_md(stat_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::a);

  auto src_desc = src_mem.fwd.get_desc();

  auto gn_pd = dnnl::group_normalization_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_training, src_desc, src_desc, channels,
      1e-5f,
      dnnl::normalization_flags::use_scale |
          dnnl::normalization_flags::use_shift);

  auto gn_bw_pd = dnnl::group_normalization_backward::primitive_desc(
      engine_, dnnl::prop_kind::backward, src_desc, src_desc, src_desc,
      channels, 1e-5f,
      dnnl::normalization_flags::use_scale |
          dnnl::normalization_flags::use_shift,
      gn_pd);

  dnnl::memory scale_mem(gn_pd.weights_desc(), engine_);
  dnnl::memory shift_mem(gn_pd.weights_desc(), engine_);
  dnnl::memory diff_scale_mem(gn_bw_pd.diff_weights_desc(), engine_);
  dnnl::memory diff_shift_mem(gn_bw_pd.diff_weights_desc(), engine_);

  parameters_[name + ".weight"] = {scale_mem, {channels}, diff_scale_mem};
  parameters_[name + ".bias"] = {shift_mem, {channels}, diff_shift_mem};

  dnnl::memory dst_mem(gn_pd.dst_desc(), engine_);
  dnnl::memory diff_dst_mem(gn_bw_pd.diff_dst_desc(), engine_);

  dnnl::memory mean_mem(gn_pd.mean_desc(), engine_);
  dnnl::memory var_mem(gn_pd.variance_desc(), engine_);

  dnnl::group_normalization_forward gn(gn_pd);
  dnnl::group_normalization_backward gn_bw(gn_bw_pd);

  pipeline_.push_back([=, this]() mutable {
    gn.execute(stream_, {{DNNL_ARG_SRC, src_mem.fwd},
                         {DNNL_ARG_SCALE, scale_mem},
                         {DNNL_ARG_SHIFT, shift_mem},
                         {DNNL_ARG_MEAN, mean_mem},
                         {DNNL_ARG_VARIANCE, var_mem},
                         {DNNL_ARG_DST, dst_mem}});
  });

  backward_pipeline_.push_back([=, this]() mutable {
    gn_bw.execute(stream_, {{DNNL_ARG_SRC, src_mem.fwd},
                            {DNNL_ARG_MEAN, mean_mem},
                            {DNNL_ARG_VARIANCE, var_mem},
                            {DNNL_ARG_DIFF_DST, diff_dst_mem},
                            {DNNL_ARG_SCALE, scale_mem},
                            {DNNL_ARG_DIFF_SRC, src_mem.bwd},
                            {DNNL_ARG_DIFF_SCALE, diff_scale_mem},
                            {DNNL_ARG_DIFF_SHIFT, diff_shift_mem}});
  });

  return {dst_mem, diff_dst_mem};
}

TransformNetwork::MemPair TransformNetwork::create_relu(MemPair src_mem) {
  auto relu_pd = dnnl::eltwise_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_relu,
      src_mem.fwd.get_desc(), src_mem.fwd.get_desc(), 0.0f, 0.0f);

  auto relu_bw_pd = dnnl::eltwise_backward::primitive_desc(
      engine_, dnnl::algorithm::eltwise_relu_use_dst_for_bwd,
      src_mem.bwd.get_desc(), src_mem.bwd.get_desc(), src_mem.fwd.get_desc(),
      0.0f, 0.0f, relu_pd);

  dnnl::memory dst_mem(relu_pd.dst_desc(), engine_);
  dnnl::memory diff_dst_mem(relu_bw_pd.diff_dst_desc(), engine_);
  dnnl::eltwise_forward relu(relu_pd);
  dnnl::eltwise_backward relu_bw(relu_bw_pd);

  pipeline_.push_back([=, this]() mutable {
    relu.execute(stream_,
                 {{DNNL_ARG_SRC, src_mem.fwd}, {DNNL_ARG_DST, dst_mem}});
  });

  backward_pipeline_.push_back([=, this]() mutable {
    relu_bw.execute(stream_, {{DNNL_ARG_SRC, dst_mem},
                              {DNNL_ARG_DIFF_DST, diff_dst_mem},
                              {DNNL_ARG_DIFF_SRC, src_mem.bwd}});
  });

  return {dst_mem, diff_dst_mem};
}

TransformNetwork::MemPair TransformNetwork::create_resample(MemPair src_mem,
                                                            float scale) {
  auto src_dims = src_mem.fwd.get_desc().get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0], src_dims[1],
                                 (dnnl::memory::dim)(src_dims[2] * scale),
                                 (dnnl::memory::dim)(src_dims[3] * scale)};
  dnnl::memory::desc dst_md(dst_dims, dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::nchw);

  auto resamp_pd = dnnl::resampling_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_training,
      dnnl::algorithm::resampling_linear, src_mem.fwd.get_desc(), dst_md);

  auto resamp_bw_pd = dnnl::resampling_backward::primitive_desc(
      engine_, dnnl::algorithm::resampling_linear, src_mem.bwd.get_desc(),
      dst_md, resamp_pd);

  dnnl::memory dst_mem(resamp_pd.dst_desc(), engine_);
  dnnl::memory diff_dst_mem(resamp_bw_pd.diff_dst_desc(), engine_);
  dnnl::resampling_forward resamp(resamp_pd);
  dnnl::resampling_backward resamp_bw(resamp_bw_pd);

  pipeline_.push_back([=, this]() mutable {
    resamp.execute(stream_,
                   {{DNNL_ARG_SRC, src_mem.fwd}, {DNNL_ARG_DST, dst_mem}});
  });

  backward_pipeline_.push_back([=, this]() mutable {
    resamp_bw.execute(stream_, {{DNNL_ARG_DIFF_DST, diff_dst_mem},
                                {DNNL_ARG_DIFF_SRC, src_mem.bwd}});
  });

  return {dst_mem, diff_dst_mem};
}

TransformNetwork::MemPair TransformNetwork::create_add(MemPair src0_mem,
                                                       MemPair src1_mem) {
  std::vector<float> scales = {1.0f, 1.0f};
  std::vector<dnnl::memory::desc> srcs = {src0_mem.fwd.get_desc(),
                                          src1_mem.fwd.get_desc()};

  auto sum_pd = dnnl::sum::primitive_desc(engine_, scales, srcs);

  dnnl::memory dst_mem(sum_pd.dst_desc(), engine_);
  dnnl::memory diff_dst_mem(sum_pd.dst_desc(), engine_);
  dnnl::sum add(sum_pd);

  pipeline_.push_back([=, this]() mutable {
    add.execute(stream_, {{DNNL_ARG_MULTIPLE_SRC, src0_mem.fwd},
                          {DNNL_ARG_MULTIPLE_SRC + 1, src1_mem.fwd},
                          {DNNL_ARG_DST, dst_mem}});
  });

  backward_pipeline_.push_back([=, this]() mutable {
    auto copy_to = [](dnnl::memory &dst, const dnnl::memory &src) {
      float *d = static_cast<float *>(dst.get_data_handle());
      const float *s = static_cast<const float *>(src.get_data_handle());
      std::size_t size = dst.get_desc().get_size() / sizeof(float);
      std::memcpy(d, s, size * sizeof(float));
    };
    // Sum branches distribute gradient equally without accumulation
    // pre-requisites
    copy_to(src0_mem.bwd, diff_dst_mem);
    copy_to(src1_mem.bwd, diff_dst_mem);
  });

  return {dst_mem, diff_dst_mem};
}

TransformNetwork::MemPair
TransformNetwork::create_resblock(const std::string &name, int channels,
                                  MemPair src_mem) {
  // Branch the backward memory for conv1 so it doesn't overwrite the shortcut
  // gradient
  MemPair conv_input = src_mem;
  dnnl::memory conv1_bwd(src_mem.bwd.get_desc(), engine_);
  conv_input.bwd = conv1_bwd;

  MemPair out1 =
      create_conv(name + ".conv1", channels, channels, 3, 1, 1, conv_input);
  out1 = create_norm(name + ".norm1", channels, out1);
  out1 = create_relu(out1);

  MemPair out2 =
      create_conv(name + ".conv2", channels, channels, 3, 1, 1, out1);
  out2 = create_norm(name + ".norm2", channels, out2);

  MemPair res = create_add(src_mem, out2);

  // Accumulate conv1 backward branch into the main trunk
  backward_pipeline_.push_back([=, this]() mutable {
    float *main_bwd = static_cast<float *>(src_mem.bwd.get_data_handle());
    const float *branch_bwd =
        static_cast<const float *>(conv1_bwd.get_data_handle());
    std::size_t size = src_mem.bwd.get_desc().get_size() / sizeof(float);
    for (std::size_t i = 0; i < size; ++i) {
      main_bwd[i] += branch_bwd[i];
    }
  });

  return res;
}

} // namespace stylor
