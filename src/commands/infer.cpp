#include "infer.hpp"
#include "stylor/transform_network.hpp"
#include <iostream>

namespace commands {

/// @brief Run inference using a trained model.
/// @param model Path to the trained model.
/// @param input Path to the input image.
/// @param output Path to save the stylized output.
void handle_infer(const std::string &model, const std::string &input,
                  const std::string &output) {
  std::cout << "Running inference on " << input << " using model " << model
            << ", output to " << output << '\n';

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  stylor::TransformNetwork net(engine, 256, 256);
  net.load_weights(model);
  std::cout << "Loaded model weights successfully from: " << model << '\n';
}

/// @brief Register the infer command with the CLI app.
/// @param app The CLI app to register the command to.
void register_infer(CLI::App &app) {
  auto *infer_cmd =
      app.add_subcommand("infer", "Run inference using a trained model");

  auto *model =
      infer_cmd->add_option("model", "Path to the trained model")->required();
  auto *input =
      infer_cmd->add_option("input", "Path to the input image")->required();
  auto *output =
      infer_cmd->add_option("output", "Path to save the stylized output")
          ->required();

  infer_cmd->callback([model, input, output]() {
    handle_infer(model->as<std::string>(), input->as<std::string>(),
                 output->as<std::string>());
  });
}

} // namespace commands
