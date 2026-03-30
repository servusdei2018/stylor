#include "stylor/safetensors_io.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace stylor {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// SafetensorsLoader
// ---------------------------------------------------------------------------

SafetensorsLoader::SafetensorsLoader(const std::string &path)
    : file_(path, std::ios::binary | std::ios::in) {
  if (!file_.is_open()) {
    throw std::runtime_error("Failed to open safetensors file: " + path);
  }

  // Read the 8-byte header length.
  uint64_t header_len = 0;
  if (!file_.read(reinterpret_cast<char *>(&header_len), sizeof(header_len))) {
    throw std::runtime_error("Failed to read header length.");
  }

  // Read the JSON header string.
  std::string header_str(header_len, '\0');
  if (!file_.read(&header_str[0], header_len)) {
    throw std::runtime_error("Failed to read header string.");
  }

  buffer_start_ = sizeof(header_len) + header_len;

  // Parse the header and extract tensor metadata.
  try {
    auto header_json = json::parse(header_str);
    for (auto it = header_json.begin(); it != header_json.end(); ++it) {
      if (it.key() == "__metadata__") {
        continue;
      }
      auto obj = it.value();
      SafetensorInfo info;
      info.dtype = obj.at("dtype").get<std::string>();
      if (info.dtype != "F32") {
        std::cerr << "Warning: Read tensor " << it.key() << " with dtype "
                  << info.dtype << " which is not F32. Expected F32.\n";
      }
      obj.at("shape").get_to(info.shape);
      auto offsets = obj.at("data_offsets").get<std::vector<std::size_t>>();
      info.data_offsets[0] = offsets[0];
      info.data_offsets[1] = offsets[1];
      tensors_[it.key()] = std::move(info);
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("Failed to parse safetensors header: ") + e.what());
  }
}

bool SafetensorsLoader::has_tensor(const std::string &name) const {
  return tensors_.find(name) != tensors_.end();
}

const SafetensorInfo &
SafetensorsLoader::get_tensor_info(const std::string &name) const {
  return tensors_.at(name);
}

void SafetensorsLoader::load_tensor(const std::string &name, float *dst,
                                    std::size_t count) {
  // Locate the tensor in the parsed metadata.
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("Tensor not found in safetensors file: " + name);
  }

  const auto &info = it->second;
  std::size_t size_bytes = info.data_offsets[1] - info.data_offsets[0];
  if (size_bytes != count * sizeof(float)) {
    throw std::runtime_error("Tensor count mismatch for " + name);
  }

  // Seek to the tensor's data offset and read into the destination buffer.
  file_.seekg(buffer_start_ + info.data_offsets[0], std::ios::beg);
  if (!file_.read(reinterpret_cast<char *>(dst), size_bytes)) {
    throw std::runtime_error("Failed to read tensor data for " + name);
  }
}

// ---------------------------------------------------------------------------
// SafetensorsWriter
// ---------------------------------------------------------------------------

SafetensorsWriter::SafetensorsWriter(const std::string &path) : path_(path) {}

void SafetensorsWriter::add_tensor(const std::string &name,
                                   const std::vector<int> &shape,
                                   const float *data, std::size_t count) {
  TensorData td;
  td.shape = shape;
  td.data = data;
  td.count = count;
  tensors_[name] = td;
}

void SafetensorsWriter::write() {
  // Build the JSON header with tensor metadata and compute offsets.
  json header;
  header["__metadata__"] = {{"format", "pt"}};

  std::size_t current_offset = 0;
  for (const auto &pair : tensors_) {
    const auto &name = pair.first;
    const auto &td = pair.second;

    std::size_t size_bytes = td.count * sizeof(float);
    header[name] = {
        {"dtype", "F32"},
        {"shape", td.shape},
        {"data_offsets", {current_offset, current_offset + size_bytes}}};
    current_offset += size_bytes;
  }

  // Finalize the header string and pad to 8-byte alignment.
  std::string header_str = header.dump();
  std::size_t pad = 8 - (header_str.length() % 8);
  if (pad != 8) {
    header_str.append(pad, ' ');
  }

  uint64_t header_len = header_str.length();

  // Write the file: [header_len (8B)] [header (JSON)] [data (binary)]
  std::ofstream out(path_, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open output safetensors file: " +
                             path_);
  }

  out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
  out.write(header_str.c_str(), header_str.length());

  for (const auto &pair : tensors_) {
    const auto &td = pair.second;
    out.write(reinterpret_cast<const char *>(td.data),
              td.count * sizeof(float));
  }
}

} // namespace stylor
