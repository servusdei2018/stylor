#ifndef STYLOR_TRAIN_HPP
#define STYLOR_TRAIN_HPP

#include <CLI/CLI.hpp>
#include <string>

namespace commands {

/// @brief Train the style transfer model.
/// @param model Path to save the model.
/// @param style Path to the style image.
/// @param content Path to the content image.
void handle_train(const std::string &model, const std::string &style,
                  const std::string &content);

/// @brief Register the train command with the CLI app.
/// @param app The CLI app to register the command to.
void register_train(CLI::App &app);

} // namespace commands

#endif // STYLOR_TRAIN_HPP
