#include "stylor/image.hpp"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace stylor {

Image load_image(const std::string &path) {
  Image img;
  uint8_t *raw_data =
      stbi_load(path.c_str(), &img.width, &img.height, &img.channels, 0);

  if (!raw_data) {
    throw std::runtime_error("Failed to load image: " + path + " - " +
                             stbi_failure_reason());
  }

  size_t data_size = img.width * img.height * img.channels;
  img.data.assign(raw_data, raw_data + data_size);

  stbi_image_free(raw_data);
  return img;
}

void save_image(const std::string &path, const Image &img) {
  if (img.data.empty() || img.width <= 0 || img.height <= 0 ||
      img.channels <= 0) {
    throw std::runtime_error("Cannot save empty or invalid image.");
  }

  size_t ext_pos = path.find_last_of('.');
  if (ext_pos == std::string::npos) {
    throw std::runtime_error("Path missing extension: " + path);
  }

  std::string ext = path.substr(ext_pos);
  for (char &c : ext) {
    c = std::tolower(c);
  }

  int success = 0;
  if (ext == ".png") {
    success = stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                             img.data.data(), img.width * img.channels);
  } else if (ext == ".jpg" || ext == ".jpeg") {
    success = stbi_write_jpg(path.c_str(), img.width, img.height, img.channels,
                             img.data.data(), 90);
  } else {
    throw std::runtime_error("Unsupported image file extension: " + ext);
  }

  if (!success) {
    throw std::runtime_error("Failed to save image to: " + path);
  }
}

Image resize_image(const Image &img, int width, int height) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument(
        "resize_image: target dimensions must be positive (got " +
        std::to_string(width) + "x" + std::to_string(height) + ")");

  if (img.width == width && img.height == height)
    return img; // no-op: already the right size

  Image out;
  out.width = width;
  out.height = height;
  out.channels = img.channels;
  out.data.resize(static_cast<std::size_t>(width * height * img.channels));

  stbir_resize_uint8_linear(img.data.data(), img.width, img.height, 0,
                            out.data.data(), width, height, 0,
                            static_cast<stbir_pixel_layout>(img.channels));

  return out;
}

} // namespace stylor
