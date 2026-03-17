#ifndef STYLOR_PREPROCESSING_HPP
#define STYLOR_PREPROCESSING_HPP

#include "stylor/image.hpp"
#include "stylor/tensor.hpp"
#include <dnnl.hpp>

namespace stylor {

/// @brief Convert an 8-bit RGB Image into a normalised NCHW float Tensor.
///
/// Applies the ImageNet per-channel mean subtraction used during VGG-19
/// training (BGR order, Simonyan & Zisserman 2014):
///   B: 103.939   G: 116.779   R: 123.680
///
/// Channel order is swapped from RGB (stb layout) to BGR (VGG convention).
/// Output layout: {1, 3, height, width} in NCHW format.
///
/// @param img    Source image loaded via load_image().
/// @param engine oneDNN CPU engine to allocate the output Tensor on.
/// @return       Normalised NCHW float Tensor ready for VGG-19 input.
/// @throws std::invalid_argument If img has zero or negative dimensions,
///                               or does not have exactly 3 channels.
Tensor preprocess_image(const Image &img, const dnnl::engine &engine);

} // namespace stylor

#endif // STYLOR_PREPROCESSING_HPP
