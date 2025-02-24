#include <CLI/CLI.hpp>
#include <iostream>
#include "stylor/version.hpp"

/// \brief Entry point for the Stylor CLI.
/// \param argc Argument count.
/// \param argv Argument vector.
/// \return Exit status.
int main(int argc, char **argv) {
  CLI::App app{"stylor " + std::string(stylor::version_string)};
  app.set_version_flag("-v,--version", stylor::version_string);

  std::string model_path;
  std::string img1, img2;

  auto *train_cmd =
      app.add_subcommand("train", "Train the style transfer model");
  train_cmd->add_option("model", model_path, "Path to save the trained model")
      ->required();
  train_cmd->add_option("style_image", img1, "Path to the style image")
      ->required();
  train_cmd->add_option("content_image", img2, "Path to the content image")
      ->required();

  auto *infer_cmd =
      app.add_subcommand("infer", "Run inference using a trained model");
  infer_cmd->add_option("model", model_path, "Path to the trained model")
      ->required();
  infer_cmd->add_option("input_image", img1, "Path to the input image")
      ->required();
  infer_cmd
      ->add_option("output_image", img2, "Path to save the stylized output")
      ->required();

  CLI11_PARSE(app, argc, argv);

  if (train_cmd->parsed()) {
    std::cout << "Training model: " << model_path << " using " << img1
              << " and " << img2 << std::endl;
  } else if (infer_cmd->parsed()) {
    std::cout << "Running inference on " << img1 << " using model "
              << model_path << ", output to " << img2 << std::endl;
  } else {
    std::cout << app.help() << std::endl;
  }

  return 0;
}
