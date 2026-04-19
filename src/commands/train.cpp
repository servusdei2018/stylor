#include "train.hpp"
#include "stylor/image.hpp"
#include "stylor/loss.hpp"
#include "stylor/optimizer.hpp"
#include "stylor/preprocessing.hpp"
#include "stylor/training_context.hpp"
#include "stylor/transform_network.hpp"
#include "stylor/vgg.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace commands {

static void scale_tensor(stylor::Tensor &t, float scale) {
  float *ptr = t.get_data();
  auto dims = t.get_dims();
  const std::size_t n = dims[0] * dims[1] * dims[2] * dims[3];
#pragma omp parallel for simd
  for (std::size_t i = 0; i < n; ++i)
    ptr[i] *= scale;
}

static void add_tensors(stylor::Tensor &dst, const stylor::Tensor &src) {
  float *d = dst.get_data();
  const float *s = src.get_data();
  auto dims = dst.get_dims();
  const std::size_t n = dims[0] * dims[1] * dims[2] * dims[3];
#pragma omp parallel for simd
  for (std::size_t i = 0; i < n; ++i)
    d[i] += s[i];
}

/// @brief Clips all gradient buffers in @p params so their global L2 norm
/// does not exceed @p max_norm.  If @p max_norm is 0 the function is a no-op.
/// @return The pre-clip global gradient norm.
static float clip_grad_norm(
    std::unordered_map<std::string, stylor::TransformNetwork::ParamDescriptor>
        &params,
    float max_norm) {
  if (max_norm <= 0.0f)
    return 0.0f;

  double total_sq = 0.0;
  for (auto &[name, desc] : params) {
    if (!desc.diff_mem)
      continue;
    const float *g =
        static_cast<const float *>(desc.diff_mem.get_data_handle());
    size_t n = 1;
    for (int d : desc.shape)
      n *= d;
#pragma omp parallel for simd reduction(+ : total_sq)
    for (size_t i = 0; i < n; ++i)
      total_sq += static_cast<double>(g[i]) * g[i];
  }
  float total_norm = static_cast<float>(std::sqrt(total_sq));

  if (total_norm > max_norm) {
    float scale = max_norm / total_norm;
    for (auto &[name, desc] : params) {
      if (!desc.diff_mem)
        continue;
      float *g = static_cast<float *>(desc.diff_mem.get_data_handle());
      size_t n = 1;
      for (int d : desc.shape)
        n *= d;
#pragma omp parallel for simd
      for (size_t i = 0; i < n; ++i)
        g[i] *= scale;
    }
  }
  return total_norm;
}

void handle_train(const std::string &model, const std::string &style,
                  const std::string &content_path,
                  const std::string &vgg_weights, float alpha, float beta,
                  float tv_weight, int epochs, float learning_rate,
                  int image_size, int checkpoint_interval,
                  float max_grad_norm) {
  std::cout << "Training model: " << model << '\n'
            << "Style image: " << style << '\n'
            << "Content path: " << content_path << '\n'
            << "VGG weights: " << vgg_weights << '\n'
            << "Alpha: " << alpha << ", Beta: " << beta
            << ", TV weight: " << tv_weight << '\n'
            << "Image size: " << image_size << "x" << image_size
            << ", LR: " << learning_rate << ", Max grad norm: "
            << (max_grad_norm > 0.0f ? std::to_string(max_grad_norm)
                                     : "disabled")
            << '\n';

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  std::cout << "Initializing networks...\n";
  stylor::TransformNetwork net(engine, image_size, image_size);
  stylor::Vgg19 vgg(engine, image_size, image_size);

  std::cout << "Loading VGG weights from " << vgg_weights << "...\n";
  vgg.load_weights(vgg_weights);

  stylor::AdamOptimizer optimizer(learning_rate);

  std::cout << "Initializing training context...\n";
  stylor::TrainingContext ctx(engine, image_size, image_size);

  std::cout << "Preprocessing style image...\n";
  stylor::Image style_img = stylor::load_image(style);
  if (style_img.width != image_size || style_img.height != image_size) {
    std::cerr << "Warning: style image (" << style_img.width << "x"
              << style_img.height << ") does not match --image-size ("
              << image_size << "x" << image_size
              << "). Resizing automatically.\n";
    style_img = stylor::resize_image(style_img, image_size, image_size);
  }
  stylor::Tensor style_tensor =
      stylor::preprocess_image(style_img, engine, stream);

  std::cout << "Computing style targets...\n";
  vgg.forward(style_tensor, stream);
  std::unordered_map<stylor::VggLayer, stylor::Tensor> style_targets;
  stylor::VggLayer style_layers[] = {
      stylor::VggLayer::relu1_1, stylor::VggLayer::relu2_1,
      stylor::VggLayer::relu3_1, stylor::VggLayer::relu4_1,
      stylor::VggLayer::relu5_1};

  for (auto layer : style_layers) {
    style_targets.emplace(
        layer, stylor::compute_gram_matrix(vgg.get_feature_map(layer),
                                           ctx.gram(layer), engine, stream));
  }

  std::vector<std::string> content_images;
  if (fs::is_directory(content_path)) {
    for (const auto &entry : fs::recursive_directory_iterator(content_path)) {
      if (entry.is_regular_file()) {
        content_images.push_back(entry.path().string());
      }
    }
  } else {
    content_images.push_back(content_path);
  }

  if (content_images.empty()) {
    std::cerr << "No content images found in " << content_path << '\n';
    return;
  }
  std::cout << "Found " << content_images.size() << " content images.\n";

  int iter = 0;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "--- Epoch " << (epoch + 1) << "/" << epochs << " ---\n";
    for (const auto &c_path : content_images) {
      try {
        stylor::Image content_img = stylor::load_image(c_path);
        // Resize to training resolution if the image does not already match.
        if (content_img.width != image_size ||
            content_img.height != image_size) {
          std::cerr << "Warning: " << c_path << " (" << content_img.width << "x"
                    << content_img.height << ") resized to " << image_size
                    << "x" << image_size << ".\n";
          content_img =
              stylor::resize_image(content_img, image_size, image_size);
        }
        stylor::Tensor content_tensor =
            stylor::preprocess_image(content_img, engine, stream);

        // Forward content image through VGG to get content target
        vgg.forward(content_tensor, stream);
        // Deep copy to avoid overwriting content_target by the next
        // vgg.forward() call
        stylor::Tensor content_target(
            vgg.get_feature_map(stylor::VggLayer::relu4_2).get_dims(), engine);
        dnnl::reorder(
            vgg.get_feature_map(stylor::VggLayer::relu4_2).get_memory(),
            content_target.get_memory())
            .execute(
                stream,
                {{DNNL_ARG_FROM,
                  vgg.get_feature_map(stylor::VggLayer::relu4_2).get_memory()},
                 {DNNL_ARG_TO, content_target.get_memory()}});
        stream.wait();

        // Forward TransformNetwork
        net.forward(content_tensor);
        const stylor::Tensor &generated_img = net.get_output();

        // Forward VGG on generated image
        vgg.forward(generated_img, stream);

        std::unordered_map<stylor::VggLayer, stylor::Tensor> loss_gradients;
        float total_loss = 0.0f;

        // Content Loss
        auto c_loss = stylor::compute_content_loss(
            vgg.get_feature_map(stylor::VggLayer::relu4_2), content_target,
            true, engine, stream);

        float current_c_loss = alpha * c_loss.value;
        total_loss += current_c_loss;

        if (c_loss.gradient) {
          scale_tensor(*c_loss.gradient, alpha);
          loss_gradients.emplace(stylor::VggLayer::relu4_2,
                                 std::move(*c_loss.gradient));
        }

        // Style Loss
        float total_s_loss = 0.0f;
        for (auto l : style_layers) {
          auto gram = stylor::compute_gram_matrix(vgg.get_feature_map(l),
                                                  ctx.gram(l), engine, stream);
          auto s_loss = stylor::compute_style_loss(
              gram, style_targets.at(l), vgg.get_feature_map(l),
              ctx.style_bw(l), true, engine, stream);
          total_s_loss += s_loss.value;
          if (s_loss.gradient) {
            // Scale by beta: consistent with current_s_loss = beta *
            // total_s_loss.
            scale_tensor(*s_loss.gradient, beta);
            loss_gradients.emplace(l, std::move(*s_loss.gradient));
          }
        }
        float current_s_loss = beta * total_s_loss;
        total_loss += current_s_loss;

        // TV Loss
        auto tv_loss =
            stylor::compute_tv_loss(generated_img, true, engine, stream);
        float current_tv_loss = tv_weight * tv_loss.value;
        total_loss += current_tv_loss;

        if (tv_loss.gradient) {
          scale_tensor(*tv_loss.gradient, tv_weight);
        }

        // VGG Backward
        stylor::Tensor vgg_grad = vgg.backward(loss_gradients, stream);

        // Add TV derivative
        if (tv_loss.gradient) {
          add_tensors(vgg_grad, *tv_loss.gradient);
        }

        // TransformNetwork Backward & Step
        net.backward(vgg_grad);

        float grad_norm = clip_grad_norm(net.get_parameters(), max_grad_norm);
        (void)grad_norm; // available for diagnostic logging if needed

        optimizer.step(net.get_parameters());

        std::cout << "Iter " << iter << " | Loss: " << total_loss
                  << " (C: " << current_c_loss << ", S: " << current_s_loss
                  << ", TV: " << current_tv_loss << ")\n";

        if ((iter + 1) % checkpoint_interval == 0) {
          std::string cp_path =
              model + "_iter_" + std::to_string(iter + 1) + ".safetensors";
          net.save_weights(cp_path);
          std::cout << "Checkpoint saved to: " << cp_path << '\n';
        }

        iter++;

      } catch (const std::exception &e) {
        std::cerr << "Error processing " << c_path << ": " << e.what() << '\n';
      }
    }
    net.save_weights(model);
    std::cout << "Epoch " << (epoch + 1) << " complete. Saved to " << model
              << '\n';
  }
}

// Register the train command with the CLI app.
void register_train(CLI::App &app) {
  auto *train_cmd =
      app.add_subcommand("train", "Train the style transfer model");

  auto *model =
      train_cmd
          ->add_option("-m,--model", "Path to save the model (.safetensors)")
          ->required();
  auto *style = train_cmd->add_option("-s,--style", "Path to the style image")
                    ->required()
                    ->check(CLI::ExistingFile);
  auto *content =
      train_cmd
          ->add_option("-c,--content", "Path to the content image or directory")
          ->required()
          ->check(CLI::ExistingFile | CLI::ExistingDirectory);
  auto *vgg_weights =
      train_cmd->add_option("--vgg-weights", "Path to the pretrained vgg19.bin")
          ->required()
          ->check(CLI::ExistingFile);

  train_cmd->add_option("--alpha", "Content loss weight (default: 1e5)")
      ->default_val(1e5f)
      ->check(CLI::PositiveNumber);
  train_cmd->add_option("--beta", "Style loss weight (default: 1e10)")
      ->default_val(1e10f)
      ->check(CLI::PositiveNumber);
  train_cmd->add_option("--tv-weight", "Total Variation weight (default: 1e-3)")
      ->default_val(1e-3f)
      ->check(CLI::NonNegativeNumber);
  train_cmd->add_option("--epochs", "Number of training epochs (default: 2)")
      ->default_val(2)
      ->check(CLI::PositiveNumber);
  train_cmd->add_option("--lr", "Learning rate (default: 1e-3)")
      ->default_val(1e-3f)
      ->check(CLI::PositiveNumber);
  train_cmd
      ->add_option("--image-size", "Training image resolution (default: 256)")
      ->default_val(256)
      ->check(CLI::PositiveNumber);
  train_cmd
      ->add_option("--checkpoint-interval",
                   "Iterations between checkpoints (default: 2000)")
      ->default_val(2000)
      ->check(CLI::PositiveNumber);
  train_cmd
      ->add_option("--max-grad-norm",
                   "Global L2 gradient norm clipping threshold (0 = disabled, "
                   "default: 10.0)")
      ->default_val(10.0f)
      ->check(CLI::NonNegativeNumber);

  train_cmd->callback([model, style, content, vgg_weights, train_cmd]() {
    float alpha_v = train_cmd->get_option("--alpha")->as<float>();
    float beta_v = train_cmd->get_option("--beta")->as<float>();
    float tv_weight_v = train_cmd->get_option("--tv-weight")->as<float>();
    int epochs_v = train_cmd->get_option("--epochs")->as<int>();
    float lr_v = train_cmd->get_option("--lr")->as<float>();
    int image_size_v = train_cmd->get_option("--image-size")->as<int>();
    int interval_v = train_cmd->get_option("--checkpoint-interval")->as<int>();
    float max_grad_norm_v =
        train_cmd->get_option("--max-grad-norm")->as<float>();

    handle_train(model->as<std::string>(), style->as<std::string>(),
                 content->as<std::string>(), vgg_weights->as<std::string>(),
                 alpha_v, beta_v, tv_weight_v, epochs_v, lr_v, image_size_v,
                 interval_v, max_grad_norm_v);
  });
}

} // namespace commands
