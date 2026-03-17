#include "stylor/weight_loader.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

namespace fs = std::filesystem;
using namespace stylor;

// ---------------------------------------------------------------------------
// Helper: write a minimal .bin file with one or more blobs.
// Each blob: [uint32_t count][count × float32]
// ---------------------------------------------------------------------------
static std::string write_bin(const std::vector<std::vector<float>> &blobs) {
  std::string path = (fs::temp_directory_path() / "test_weights.bin").string();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  for (const auto &blob : blobs) {
    uint32_t n = static_cast<uint32_t>(blob.size());
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(blob.data()),
              blob.size() * sizeof(float));
  }
  return path;
}

// ---------------------------------------------------------------------------

TEST(WeightLoaderTest, MissingFileThrows) {
  EXPECT_THROW(WeightLoader("/nonexistent/path/weights.bin"),
               std::runtime_error);
}

TEST(WeightLoaderTest, BinLoaderRoundTrip) {
  const std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
  std::string path = write_bin({expected});

  WeightLoader loader(path);
  ASSERT_TRUE(loader.has_next());

  std::vector<float> buf(expected.size());
  loader.read_next(buf.data(), buf.size());

  EXPECT_EQ(buf, expected);
  EXPECT_FALSE(loader.has_next());
  EXPECT_EQ(loader.blobs_read(), 1u);

  fs::remove(path);
}

TEST(WeightLoaderTest, BinLoaderMultipleBlobs) {
  const std::vector<float> blob0 = {0.1f, 0.2f};
  const std::vector<float> blob1 = {1.0f, 2.0f, 3.0f};
  std::string path = write_bin({blob0, blob1});

  WeightLoader loader(path);

  std::vector<float> buf0(blob0.size());
  loader.read_next(buf0.data(), buf0.size());
  EXPECT_EQ(buf0, blob0);

  std::vector<float> buf1(blob1.size());
  loader.read_next(buf1.data(), buf1.size());
  EXPECT_EQ(buf1, blob1);

  EXPECT_FALSE(loader.has_next());
  EXPECT_EQ(loader.blobs_read(), 2u);

  fs::remove(path);
}

TEST(WeightLoaderTest, BinLoaderExhaustedThrows) {
  std::string path = write_bin({{1.0f}});
  WeightLoader loader(path);

  float dummy = 0.0f;
  loader.read_next(&dummy, 1); // consumes the only blob

  EXPECT_FALSE(loader.has_next());
  EXPECT_THROW(loader.read_next(&dummy, 1), std::out_of_range);

  fs::remove(path);
}

TEST(WeightLoaderTest, CountMismatchThrows) {
  // Write a blob of 4 elements, then try to read 5.
  std::string path = write_bin({{1.0f, 2.0f, 3.0f, 4.0f}});
  WeightLoader loader(path);

  std::vector<float> buf(5);
  EXPECT_THROW(loader.read_next(buf.data(), 5), std::runtime_error);

  fs::remove(path);
}
