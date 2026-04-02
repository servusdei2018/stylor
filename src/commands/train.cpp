#include "train.hpp"
#include "stylor/image.hpp"
#include "stylor/loss.hpp"
#include "stylor/optimizer.hpp"
#include "stylor/preprocessing.hpp"
#include "stylor/transform_network.hpp"
#include "stylor/vgg.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace commands {

static void scale_tensor(stylor::Tensor &t, float scale) {
  float *data = t.get_data();
  size_t size = 1;
  for (auto d : t.get_dims())
    size *= d;
  for (size_t i = 0; i < size; ++i) {
    data[i] *= scale;
  }
}

static void add_tensors(stylor::Tensor &dst, const stylor::Tensor &src) {
  float *dst_data = dst.get_data();
  const float *src_data = src.get_data();
  size_t size = 1;
  for (auto d : dst.get_dims())
    size *= d;
  for (size_t i = 0; i < size; ++i) {
    dst_data[i] += src_data[i];
  }
}

void handle_train(const std::string &model, const std::string &style,
                  const std::string &content_path,
                  const std::string &vgg_weights, float alpha, float beta,
                  float tv_weight, int epochs, int batch_size,
                  float learning_rate, int image_size,
                  int checkpoint_interval) {
  std::cout << "Training model: " << model << '\n'
            << "Style image: " << style << '\n'
            << "Content path: " << content_path << '\n'
            << "VGG weights: " << vgg_weights << '\n'
            << "Alpha: " << alpha << ", Beta: " << beta
            << ", TV weight: " << tv_weight << '\n'
            << "Image size: " << image_size << "x" << image_size
            << ", LR: " << learning_rate << '\n';

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  std::cout << "Initializing networks...\n";
  stylor::TransformNetwork net(engine, image_size, image_size);
  stylor::Vgg19 vgg(engine, image_size, image_size);

  std::cout << "Loading VGG weights from " << vgg_weights << "...\n";
  vgg.load_weights(vgg_weights);

  stylor::AdamOptimizer optimizer(learning_rate);

  std::cout << "Preprocessing style image...\n";
  stylor::Image style_img = stylor::load_image(style);
  stylor::Tensor style_tensor = stylor::preprocess_image(style_img, engine);

  std::cout << "Computing style targets...\n";
  vgg.forward(style_tensor, stream);
  std::unordered_map<stylor::VggLayer, stylor::Tensor> style_targets;
  stylor::VggLayer style_layers[] = {
      stylor::VggLayer::relu1_1, stylor::VggLayer::relu2_1,
      stylor::VggLayer::relu3_1, stylor::VggLayer::relu4_1,
      stylor::VggLayer::relu5_1};

  for (auto layer : style_layers) {
    style_targets.emplace(
        layer, stylor::compute_gram_matrix(vgg.get_feature_map(layer), engine,
                                           stream));
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
        // Assuming content dimensions match expected input_h, input_w of
        // network. For robustness, one would resize here. Our project is fixed
        // to exactly image_size for all inputs currently.
        stylor::Tensor content_tensor =
            stylor::preprocess_image(content_img, engine);

        // Check if dimensions match what VGG expects
        auto dims = content_tensor.get_dims();
        if (dims[2] != image_size || dims[3] != image_size) {
          std::cerr << "Skipping " << c_path
                    << ": image dimension mismatch. Expected " << image_size
                    << "x" << image_size << " but got " << dims[2] << "x"
                    << dims[3] << "\n";
          continue;
        }

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
                                                  engine, stream);
          auto s_loss = stylor::compute_style_loss(gram, style_targets.at(l),
                                                   vgg.get_feature_map(l), true,
                                                   engine, stream);
          total_s_loss += s_loss.value;
          if (s_loss.gradient) {
            scale_tensor(*s_loss.gradient, beta / 5.0f);
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
  train_cmd->add_option("--batch-size", "Batch size (default: 1)")
      ->default_val(1)
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

  train_cmd->callback([model, style, content, vgg_weights, train_cmd]() {
    // We retrieve the options dynamically inside the callback
    float alpha_v = train_cmd->get_option("--alpha")->as<float>();
    float beta_v = train_cmd->get_option("--beta")->as<float>();
    float tv_weight_v = train_cmd->get_option("--tv-weight")->as<float>();
    int epochs_v = train_cmd->get_option("--epochs")->as<int>();
    int batch_size_v = train_cmd->get_option("--batch-size")->as<int>();
    float lr_v = train_cmd->get_option("--lr")->as<float>();
    int image_size_v = train_cmd->get_option("--image-size")->as<int>();
    int interval_v = train_cmd->get_option("--checkpoint-interval")->as<int>();

    handle_train(model->as<std::string>(), style->as<std::string>(),
                 content->as<std::string>(), vgg_weights->as<std::string>(),
                 alpha_v, beta_v, tv_weight_v, epochs_v, batch_size_v, lr_v,
                 image_size_v, interval_v);
  });
}

} // namespace commands
