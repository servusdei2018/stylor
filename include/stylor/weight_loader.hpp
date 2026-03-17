#ifndef STYLOR_WEIGHT_LOADER_HPP
#define STYLOR_WEIGHT_LOADER_HPP

#include <cstddef>
#include <fstream>
#include <string>

namespace stylor {

/// @brief Reads pre-trained weight blobs from a `.bin` file.
///
/// Binary layout (little-endian):
///   For each weight tensor, in network layer order:
///     [uint32_t count] [count × float32 values]
///
/// Blobs are consumed sequentially; call has_next() before each read_next().
class WeightLoader {
public:
  /// @brief Open a weight file for sequential reading.
  /// @param path Path to the `.bin` weight file.
  /// @throws std::runtime_error If the file cannot be opened.
  explicit WeightLoader(const std::string &path);

  /// @brief Read the next blob into a caller-supplied buffer.
  ///
  /// The blob's stored element count must equal @p count exactly; a mismatch
  /// indicates that the caller's expected network shape and the file are out of
  /// sync.
  ///
  /// @param dst   Destination buffer; must be at least @p count floats.
  /// @param count Expected number of float elements in this blob.
  /// @throws std::out_of_range  If no more blobs remain in the file.
  /// @throws std::runtime_error If the stored count != @p count, or I/O fails.
  void read_next(float *dst, std::size_t count);

  /// @brief Return true if at least one more blob can be read.
  bool has_next() const;

  /// @brief Return the total number of blobs successfully read so far.
  std::size_t blobs_read() const;

private:
  mutable std::ifstream file_;
  std::streampos file_size_;
  std::size_t blobs_read_{0};
};

} // namespace stylor

#endif // STYLOR_WEIGHT_LOADER_HPP
