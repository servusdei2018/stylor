#include "stylor/weight_loader.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace stylor {

WeightLoader::WeightLoader(const std::string &path)
    : file_(path, std::ios::binary) {
  if (!file_.is_open()) {
    throw std::runtime_error("WeightLoader: cannot open file: " + path);
  }

  file_.seekg(0, std::ios::end);
  file_size_ = file_.tellg();
  file_.seekg(0, std::ios::beg);
}

bool WeightLoader::has_next() const {
  return file_.good() && (file_.tellg() < file_size_);
}

void WeightLoader::read_next(float *dst, std::size_t count) {
  if (!has_next()) {
    throw std::out_of_range(
        "WeightLoader: no more blobs (expected another blob for index " +
        std::to_string(blobs_read_) + ")");
  }

  // Read the stored element count from the 4-byte prefix.
  uint32_t stored_count = 0;
  if (!file_.read(reinterpret_cast<char *>(&stored_count),
                  sizeof(stored_count))) {
    throw std::runtime_error(
        "WeightLoader: failed to read blob header at index " +
        std::to_string(blobs_read_));
  }

  if (static_cast<std::size_t>(stored_count) != count) {
    throw std::runtime_error(
        "WeightLoader: blob " + std::to_string(blobs_read_) + " has " +
        std::to_string(stored_count) + " elements, but caller expected " +
        std::to_string(count));
  }

  const std::size_t byte_count = count * sizeof(float);
  if (!file_.read(reinterpret_cast<char *>(dst),
                  static_cast<std::streamsize>(byte_count))) {
    throw std::runtime_error(
        "WeightLoader: failed to read blob data at index " +
        std::to_string(blobs_read_) + " (" + std::to_string(count) +
        " floats)");
  }

  ++blobs_read_;
}

std::size_t WeightLoader::blobs_read() const { return blobs_read_; }

} // namespace stylor
