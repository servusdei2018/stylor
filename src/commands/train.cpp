#include "train.hpp"
#include <iostream>

namespace commands {

/// @brief Train the style transfer model.
/// @param model Path to save the model.
/// @param style Path to the style image.
/// @param content Path to the content image.
void handle_train(const std::string &model, const std::string &style,
                  const std::string &content) {
  std::cout << "Training model: " << model << " using " << style << " and "
            << content << '\n';
}

/// @brief Register the train command with the CLI app.
/// @param app The CLI app to register the command to.
void register_train(CLI::App &app) {
  auto *train_cmd =
      app.add_subcommand("train", "Train the style transfer model");

  auto *model =
      train_cmd->add_option("model", "Path to save the model")->required();
  auto *style =
      train_cmd->add_option("style", "Path to the style image")->required();
  auto *content =
      train_cmd->add_option("content", "Path to the content image")->required();

  train_cmd->callback([model, style, content]() {
    handle_train(model->as<std::string>(), style->as<std::string>(),
                 content->as<std::string>());
  });
}

} // namespace commands
