#ifndef STYLOR_IMAGE_HPP
#define STYLOR_IMAGE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace stylor {

/// @brief Represents an image loaded into memory.
struct Image {
  int width;
  int height;
  int channels;
  std::vector<uint8_t> data;
};

/// @brief Load an image from a file path.
/// @param path The path to the image file (JPG, PNG, etc).
/// @return An Image object containing the pixel data.
/// @throws std::runtime_error If the file cannot be loaded.
Image load_image(const std::string &path);

/// @brief Save an image to a file path.
/// @param path The path to save the image to (supports .png and .jpg
/// extensions).
/// @param img The Image object to save.
/// @throws std::runtime_error If the image cannot be saved or the extension is
/// unsupported.
void save_image(const std::string &path, const Image &img);

} // namespace stylor

#endif // STYLOR_IMAGE_HPP
