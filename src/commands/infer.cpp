#include "infer.hpp"
#include "stylor/image.hpp"
#include "stylor/preprocessing.hpp"
#include "stylor/transform_network.hpp"
#include <algorithm>
#include <iostream>

namespace commands {

void handle_infer(const std::string &model, const std::string &input,
                  const std::string &output, int image_size) {
  std::cout << "Running inference on " << input << " using model " << model
            << ", output to " << output << '\n';

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  // Load the input image
  std::cout << "Loading input image...\n";
  stylor::Image input_img = stylor::load_image(input);

  // Preprocess: u8/nhwc → f32/nchw, mean subtract. Always work in float space
  // from here. The optional resize is done via oneDNN resampling (no uint8
  // round-trip).
  stylor::Tensor input_tensor =
      stylor::preprocess_image(input_img, engine, stream);

  if (image_size > 0 &&
      (input_img.width != image_size || input_img.height != image_size)) {
    std::cerr << "Warning: input image (" << input_img.width << "x"
              << input_img.height << ") resampled to " << image_size << "x"
              << image_size << " for inference.\n";
    input_tensor = stylor::resize_tensor(input_tensor, image_size, image_size,
                                         engine, stream);
  }

  auto dims = input_tensor.get_dims(); // {N, C, H, W}
  int in_h = dims[2];
  int in_w = dims[3];

  std::cout << "Initializing TransformNetwork with dims " << in_h << "x" << in_w
            << "...\n";
  stylor::TransformNetwork net(engine, in_h, in_w);

  std::cout << "Loading weights...\n";
  net.load_weights(model);

  std::cout << "Running forward pass...\n";
  net.forward(input_tensor);

  const stylor::Tensor &output_tensor = net.get_output();
  std::cout << "Post-processing output...\n";
  stylor::Image out_img =
      stylor::postprocess_image(output_tensor, engine, stream);

  std::cout << "Saving stylized output...\n";
  stylor::save_image(output, out_img);
  std::cout << "Successfully saved to " << output << '\n';
}

void register_infer(CLI::App &app) {
  auto *infer_cmd =
      app.add_subcommand("infer", "Run inference using a trained model");

  auto *model = infer_cmd->add_option("-m,--model", "Path to the trained model")
                    ->required()
                    ->check(CLI::ExistingFile);
  auto *input = infer_cmd->add_option("-i,--input", "Path to the input image")
                    ->required()
                    ->check(CLI::ExistingFile);
  auto *output =
      infer_cmd->add_option("-o,--output", "Path to save the stylized output")
          ->required();

  infer_cmd->add_option("--image-size", "Optional fixed image size override")
      ->default_val(0)
      ->check(CLI::NonNegativeNumber);

  infer_cmd->callback([model, input, output, infer_cmd]() {
    int img_size = infer_cmd->get_option("--image-size")->as<int>();
    handle_infer(model->as<std::string>(), input->as<std::string>(),
                 output->as<std::string>(), img_size);
  });
}

} // namespace commands
