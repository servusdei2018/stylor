# stylor

Stylor is a high-performance [neural style transfer](https://en.wikipedia.org/wiki/Neural_style_transfer) framework leveraging Intel [oneDNN](https://github.com/oneapi-src/oneDNN) for efficient training and inference, enabling real-time artistic transformation with minimal computational overhead.

## Getting Started

```bash
cmake -B build -S .
cd build && make -j$(sysctl -n hw.logicalcpu)
```

### Data Preparation

To train new models, you need an image dataset and the pre-trained VGG-19 weights.

#### Image Dataset
Stylor uses image datasets, such as COCO, for training. You can download the COCO val2017 dataset for a quick start:

```bash
curl http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
unzip -q val2017.zip -d test_data/
mv test_data/val2017 test_data/coco-val2017
```

#### Pre-trained VGG-19 Weights

You will need the pre-trained VGG-19 weights converted into Stylor's specific binary format. You can download the ready-to-use `vgg19.bin` file directly from our [GitHub releases](https://github.com/servusdei2018/stylor/releases).

Alternatively, you can generate your own weights file using the included python script. We recommend using `uv`:

```bash
cd scripts && uv run export_vgg19.py
```

## Usage

### Training

To train a new style transfer model, use the `train` command, providing the pre-trained VGG weights, an input style image, and your content image dataset directory:

```bash
./bin/stylor_cli train \
    --model test_data/test_model.safetensors \
    --style test_data/style.jpg \
    --content test_data/coco-val2017/ \
    --vgg-weights scripts/vgg19.bin \
    --epochs 50 \
    --image-size 256
```

### Inference

Once you have a trained model, use the `infer` command to apply the style to new content images:

```bash
./bin/stylor_cli infer \
    --model test_data/test_model.safetensors \
    --input test_data/image.jpg \
    --output test_data/stylized_50.jpg \
    --image-size 256
```


### Available Targets

You can build specific targets using `cmake --build build --target <target_name>` (or `make <target_name>` from within the `build` directory).

| Target | Description |
|--------|-------------|
| `all` | Default target. Builds all libraries and executables. |
| `stylor` | Builds the core Stylor static library. |
| `stylor_cli` | Builds the main command-line executable. |
| `test_stylor` | Builds the GoogleTest suite executable. |
| `test` | Runs all configured tests (requires building `test_stylor` first). |
| `format` | Formats all source and header files using `clang-format`. |
| `clean` | Removes all compiled outputs. |

### Tooling / IDE Support

To generate `compile_commands.json` (useful for language servers like `clangd`), configure CMake with `CMAKE_EXPORT_COMPILE_COMMANDS`:

```bash
cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Symlink to the project root so tools can find it automatically
ln -s build/compile_commands.json .
```

## License

Stylor is distributed under the MIT License. Refer to the `LICENSE` file for details.

## Citation

Please cite our software if you use this code or part of it in your work:

```bibtex
@misc{Stylor,
      title={stylor},
      author={Nathanael Bracy},
      year={2025},
      url={https://github.com/servusdei2018/stylor},
      note={edge-optimized neural style transfer using Intel oneDNN}
} 
```