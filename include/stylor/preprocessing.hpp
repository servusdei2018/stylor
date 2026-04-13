#ifndef STYLOR_PREPROCESSING_HPP
#define STYLOR_PREPROCESSING_HPP

#include "stylor/image.hpp"
#include "stylor/tensor.hpp"
#include <dnnl.hpp>

namespace stylor {

/// @brief Converts an 8-bit RGB Image into a normalised NCHW float Tensor.
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
/// @param stream oneDNN stream to submit work to (default: constructs a
///               temporary stream — prefer the stream overload in hot loops).
/// @return       Normalised NCHW float Tensor ready for VGG-19 input.
/// @throws std::invalid_argument If img has zero/negative dimensions or ≠3 ch.
Tensor preprocess_image(const Image &img, const dnnl::engine &engine,
                        dnnl::stream &stream);

/// @brief Convenience overload — constructs a temporary stream internally.
Tensor preprocess_image(const Image &img, const dnnl::engine &engine);

/// @brief Converts a normalised NCHW float Tensor back into an 8-bit RGB Image.
///
/// Applies the inverse of ImageNet per-channel mean subtraction (BGR order):
///   B: 103.939   G: 116.779   R: 123.680
///
/// Uses a oneDNN eltwise_clip primitive for clamping to [0, 255] and a
/// reorder primitive (f32/nchw → u8/nhwc) for the channel swap and saturating
/// cast.
///
/// @param tensor NCHW float Tensor output from the stylization network.
/// @param engine oneDNN CPU engine (must match the engine used for the tensor).
/// @param stream oneDNN stream to submit work to (default: constructs a
///               temporary stream — prefer the stream overload in hot loops).
/// @return       8-bit RGB Image corresponding to the styled output.
/// @throws std::invalid_argument If tensor is not a 4D, 3-channel tensor.
Image postprocess_image(const Tensor &tensor, const dnnl::engine &engine,
                        dnnl::stream &stream);

/// @brief Convenience overload — constructs a temporary stream internally.
Image postprocess_image(const Tensor &tensor, const dnnl::engine &engine);

/// @brief Resize a float NCHW Tensor using oneDNN bilinear resampling.
///
/// Operates entirely in f32/NCHW space — no round-trip through uint8.
/// Intended for use after preprocess_image when the resize must happen in
/// float domain (e.g. inference with --image-size override).
///
/// @param tensor Source NCHW f32 Tensor.
/// @param height Target height.
/// @param width  Target width.
/// @param engine oneDNN CPU engine.
/// @param stream oneDNN stream to submit work to.
/// @return       Resized NCHW f32 Tensor.
Tensor resize_tensor(const Tensor &tensor, int height, int width,
                     const dnnl::engine &engine, dnnl::stream &stream);

} // namespace stylor

#endif // STYLOR_PREPROCESSING_HPP
