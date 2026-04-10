#ifndef STYLOR_TRAIN_HPP
#define STYLOR_TRAIN_HPP

#include <CLI/CLI.hpp>
#include <string>

namespace commands {

/// @brief Trains the style transfer model.
/// @param model           Path to save the model.
/// @param style           Path to the style image.
/// @param content_path    Path to the content image or directory.
/// @param vgg_weights     Path to the pretrained VGG-19 weights.
/// @param alpha           Content loss weight.
/// @param beta            Style loss weight.
/// @param tv_weight       Total Variation loss weight.
/// @param epochs          Number of training epochs.
/// @param learning_rate   Adam learning rate.
/// @param image_size      Training resolution (square).
/// @param checkpoint_interval Iterations between checkpoint saves.
/// @param max_grad_norm   Global L2 gradient clipping threshold (0 = off).
void handle_train(const std::string &model, const std::string &style,
                  const std::string &content_path,
                  const std::string &vgg_weights, float alpha, float beta,
                  float tv_weight, int epochs, float learning_rate,
                  int image_size, int checkpoint_interval,
                  float max_grad_norm);

/// @brief Registers the train command with the CLI app.
/// @param app The CLI app to register the command to.
void register_train(CLI::App &app);

} // namespace commands

#endif // STYLOR_TRAIN_HPP
