#ifndef STYLOR_IMAGE_HPP
#define STYLOR_IMAGE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace stylor {

/// @brief Represents an image loaded into memory.
struct Image {
  /// @brief Width of the image in pixels.
  int width;
  /// @brief Height of the image in pixels.
  int height;
  /// @brief Number of color channels (e.g., 3 for RGB).
  int channels;
  /// @brief Raw pixel data in interleaved format (e.g., RGBRGB...).
  std::vector<uint8_t> data;
};

/// @brief Loads an image from a file path.
/// @param path The path to the image file (JPG, PNG, etc).
/// @return An Image object containing the pixel data.
/// @throws std::runtime_error If the file cannot be loaded.
Image load_image(const std::string &path);

/// @brief Saves an image to a file path.
/// @param path The path to save the image to (supports .png and .jpg
/// extensions).
/// @param img The Image object to save.
/// @throws std::runtime_error If the image cannot be saved or the extension is
/// unsupported.
void save_image(const std::string &path, const Image &img);

/// @brief Resizes an image to the given dimensions using bilinear resampling.
///
/// If the source image already has the requested dimensions the original Image
/// is returned without allocation. Channel count is preserved.
///
/// @param img    Source image.
/// @param width  Target width in pixels (must be > 0).
/// @param height Target height in pixels (must be > 0).
/// @return       A new Image with dimensions {width, height, img.channels}.
/// @throws std::invalid_argument If width or height are non-positive.
Image resize_image(const Image &img, int width, int height);

} // namespace stylor

#endif // STYLOR_IMAGE_HPP
