#ifndef STYLOR_INFER_HPP
#define STYLOR_INFER_HPP

#include <CLI/CLI.hpp>
#include <string>

namespace commands {

/// @brief Run inference using a trained model.
/// @param model Path to the trained model.
/// @param input Path to the input image.
/// @param output Path to save the stylized output.
void handle_infer(const std::string &model, const std::string &input,
                  const std::string &output);

/// @brief Register the infer command with the CLI app.
/// @param app The CLI app to register the command to.
void register_infer(CLI::App &app);

} // namespace commands

#endif // STYLOR_INFER_HPP
