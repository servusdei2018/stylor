#include "commands/infer.hpp"
#include "commands/train.hpp"
#include "stylor/version.hpp"
#include <CLI/CLI.hpp>
#include <iostream>

/// @brief Entry point for the Stylor CLI.
/// @param argc Argument count.
/// @param argv Argument vector.
/// @return Exit status.
int main(int argc, char **argv) {
  CLI::App app{"stylor " + std::string(stylor::version_string)};
  app.set_version_flag("-v,--version", stylor::version_string);

  commands::register_infer(app);
  commands::register_train(app);

  CLI11_PARSE(app, argc, argv);

  return 0;
}
