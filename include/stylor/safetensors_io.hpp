#ifndef STYLOR_SAFETENSORS_IO_HPP
#define STYLOR_SAFETENSORS_IO_HPP

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace stylor {

struct SafetensorInfo {
  std::string dtype;
  std::vector<int> shape;
  std::size_t data_offsets[2];
};

/// @brief Reads pre-trained weight blobs from a `.safetensors` file.
class SafetensorsLoader {
public:
  /// @brief Open a safetensors file.
  /// @param path Path to the `.safetensors` file.
  /// @throws std::runtime_error If the file cannot be opened or parsed.
  explicit SafetensorsLoader(const std::string &path);

  /// @brief Check if a tensor exists.
  bool has_tensor(const std::string &name) const;

  /// @brief Get metadata for a tensor.
  const SafetensorInfo &get_tensor_info(const std::string &name) const;

  /// @brief Load tensor data into a caller-supplied buffer.
  /// @param name  Tensor name.
  /// @param dst   Destination buffer; must be at least @p count floats.
  /// @param count Expected number of float elements in this blob.
  /// @throws std::runtime_error If the stored count != @p count, or I/O fails.
  void load_tensor(const std::string &name, float *dst, std::size_t count);

private:
  mutable std::ifstream file_;
  std::size_t header_size_;
  std::size_t buffer_start_;
  std::unordered_map<std::string, SafetensorInfo> tensors_;
};

/// @brief Writes tensors to a `.safetensors` file.
class SafetensorsWriter {
public:
  explicit SafetensorsWriter(const std::string &path);

  /// @brief Add a tensor to be written (data must remain valid until write() is
  /// called).
  void add_tensor(const std::string &name, const std::vector<int> &shape,
                  const float *data, std::size_t count);

  /// @brief Write out the file to disk.
  void write();

private:
  std::string path_;
  struct TensorData {
    std::vector<int> shape;
    const float *data;
    std::size_t count;
  };
  std::unordered_map<std::string, TensorData> tensors_;
};

} // namespace stylor

#endif // STYLOR_SAFETENSORS_IO_HPP
