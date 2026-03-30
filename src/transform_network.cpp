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
  dnnl::memory current_mem(input_md, engine_);

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

  // Final activation and scaling is handled by the caller.
  output_tensor_ =
      std::make_unique<Tensor>(current_mem.get_desc().get_dims(), engine_);
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
  // Execute the network pipeline. Input data is assumed to be in the first
  // primitive's source memory.
  for (auto &prim : pipeline_) {
    prim();
  }
  stream_.wait();
}

const Tensor &TransformNetwork::get_output() const { return *output_tensor_; }

// ---------------------------------------------------------
// Helper implementations

dnnl::memory TransformNetwork::create_conv(const std::string &name, int ic,
                                           int oc, int kernel, int stride,
                                           int padding, dnnl::memory src_mem) {

  dnnl::memory::dims weights_dims = {oc, ic, kernel, kernel};
  dnnl::memory::dims bias_dims = {oc};
  dnnl::memory::dims strides = {stride, stride};
  dnnl::memory::dims padding_l = {padding, padding};
  dnnl::memory::dims padding_r = {padding, padding};

  dnnl::memory::desc weights_md(weights_dims, dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::oihw);
  dnnl::memory::desc bias_md(bias_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::x);

  auto src_dims = src_mem.get_desc().get_dims();
  dnnl::memory::dim out_h = (src_dims[2] + 2 * padding - kernel) / stride + 1;
  dnnl::memory::dim out_w = (src_dims[3] + 2 * padding - kernel) / stride + 1;
  dnnl::memory::desc dst_md({src_dims[0], oc, out_h, out_w},
                            dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::any);

  dnnl::memory weights_mem(weights_md, engine_);
  dnnl::memory bias_mem(bias_md, engine_);

  parameters_[name + ".weight"] = {weights_mem,
                                   {(int)oc, (int)ic, kernel, kernel}};
  parameters_[name + ".bias"] = {bias_mem, {(int)oc}};

  // Forward descriptor
  auto conv_pd = dnnl::convolution_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::convolution_direct, src_mem.get_desc(), weights_md,
      bias_md, dst_md, strides, padding_l, padding_r);

  dnnl::memory dst_mem(conv_pd.dst_desc(), engine_);
  dnnl::convolution_forward conv(conv_pd);

  pipeline_.push_back([=, this]() mutable {
    conv.execute(stream_, {{DNNL_ARG_SRC, src_mem},
                           {DNNL_ARG_WEIGHTS, weights_mem},
                           {DNNL_ARG_BIAS, bias_mem},
                           {DNNL_ARG_DST, dst_mem}});
  });

  return dst_mem;
}

dnnl::memory TransformNetwork::create_norm(const std::string &name,
                                           int channels, dnnl::memory src_mem) {
  // scale and shift (2 x channels)
  dnnl::memory::dims stat_dims = {channels};
  dnnl::memory::desc stat_md(stat_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::a);

  // Instance normalization implemented via group normalization with
  // groups = channels.
  dnnl::memory::dims scale_shift_dims = {channels};
  auto src_desc = src_mem.get_desc();

  auto gn_pd = dnnl::group_normalization_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_inference, src_desc, src_desc, channels,
      1e-5f, // epsilon
      dnnl::normalization_flags::use_scale |
          dnnl::normalization_flags::use_shift);

  dnnl::memory scale_mem(gn_pd.weights_desc(), engine_);
  dnnl::memory shift_mem(gn_pd.weights_desc(), engine_);

  parameters_[name + ".weight"] = {scale_mem, {channels}};
  parameters_[name + ".bias"] = {shift_mem, {channels}};

  dnnl::memory dst_mem(gn_pd.dst_desc(), engine_);
  dnnl::group_normalization_forward gn(gn_pd);

  pipeline_.push_back([=, this]() mutable {
    gn.execute(stream_, {{DNNL_ARG_SRC, src_mem},
                         {DNNL_ARG_SCALE, scale_mem},
                         {DNNL_ARG_SHIFT, shift_mem},
                         {DNNL_ARG_DST, dst_mem}});
  });

  return dst_mem;
}

dnnl::memory TransformNetwork::create_relu(dnnl::memory src_mem) {
  auto relu_pd = dnnl::eltwise_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::eltwise_relu, src_mem.get_desc(), src_mem.get_desc(),
      0.0f, 0.0f);

  dnnl::memory dst_mem(relu_pd.dst_desc(), engine_);
  dnnl::eltwise_forward relu(relu_pd);

  pipeline_.push_back([=, this]() mutable {
    relu.execute(stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  });

  return dst_mem;
}

dnnl::memory TransformNetwork::create_resample(dnnl::memory src_mem,
                                               float scale) {
  auto src_dims = src_mem.get_desc().get_dims();
  dnnl::memory::dims dst_dims = {src_dims[0], src_dims[1],
                                 (dnnl::memory::dim)(src_dims[2] * scale),
                                 (dnnl::memory::dim)(src_dims[3] * scale)};

  dnnl::memory::desc dst_md(dst_dims, dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::nchw);

  auto resamp_pd = dnnl::resampling_forward::primitive_desc(
      engine_, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::resampling_linear, src_mem.get_desc(), dst_md);

  dnnl::memory dst_mem(resamp_pd.dst_desc(), engine_);
  dnnl::resampling_forward resamp(resamp_pd);

  pipeline_.push_back([=, this]() mutable {
    resamp.execute(stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  });

  return dst_mem;
}

dnnl::memory TransformNetwork::create_add(dnnl::memory src0_mem,
                                          dnnl::memory src1_mem) {
  std::vector<float> scales = {1.0f, 1.0f};
  std::vector<dnnl::memory::desc> srcs = {src0_mem.get_desc(),
                                          src1_mem.get_desc()};

  auto sum_pd = dnnl::sum::primitive_desc(engine_, scales, srcs);

  dnnl::memory dst_mem(sum_pd.dst_desc(), engine_);
  dnnl::sum add(sum_pd);

  pipeline_.push_back([=, this]() mutable {
    add.execute(stream_, {{DNNL_ARG_MULTIPLE_SRC, src0_mem},
                          {DNNL_ARG_MULTIPLE_SRC + 1, src1_mem},
                          {DNNL_ARG_DST, dst_mem}});
  });

  return dst_mem;
}

dnnl::memory TransformNetwork::create_resblock(const std::string &name,
                                               int channels,
                                               dnnl::memory src_mem) {
  // Conv1 -> Norm1 -> ReLU
  dnnl::memory out1 =
      create_conv(name + ".conv1", channels, channels, 3, 1, 1, src_mem);
  dnnl::memory norm1 = create_norm(name + ".norm1", channels, out1);
  dnnl::memory act1 = create_relu(norm1);

  // Conv2 -> Norm2
  dnnl::memory out2 =
      create_conv(name + ".conv2", channels, channels, 3, 1, 1, act1);
  dnnl::memory norm2 = create_norm(name + ".norm2", channels, out2);

  // Add (residual)
  return create_add(src_mem, norm2);
}

} // namespace stylor
