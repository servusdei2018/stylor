#include "stylor/image.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace stylor;

TEST(ImageTest, SaveAndLoad) {
  std::string temp_path = "test_image.png";
  int width = 10;
  int height = 10;
  int channels = 3;

  Image img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.data.resize(width * height * channels, 0);

  // Create a red gradient
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * channels;
      img.data[idx] = 255 * x / width; // R
      img.data[idx + 1] = 0;           // G
      img.data[idx + 2] = 0;           // B
    }
  }

  ASSERT_NO_THROW(save_image(temp_path, img));

  Image loaded_img;
  ASSERT_NO_THROW(loaded_img = load_image(temp_path));

  EXPECT_EQ(loaded_img.width, width);
  EXPECT_EQ(loaded_img.height, height);
  EXPECT_EQ(loaded_img.channels, channels);
  EXPECT_EQ(loaded_img.data.size(), img.data.size());

  // Check pixels
  EXPECT_EQ(loaded_img.data[0], 0); // (0,0) Red should be 0
  EXPECT_EQ(loaded_img.data[9 * channels],
            229); // (9,0) Red should be ~229 (255*9/10)

  std::filesystem::remove(temp_path);
}

TEST(ImageTest, InvalidLoad) {
  EXPECT_THROW(load_image("nonexistent_image.png"), std::runtime_error);
}

TEST(ImageTest, InvalidSaveExtension) {
  Image img{10, 10, 3, std::vector<uint8_t>(300, 0)};
  EXPECT_THROW(save_image("test_image.bmp", img), std::runtime_error);
}
