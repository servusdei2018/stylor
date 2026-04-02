import struct
import torch
import torchvision.models as models

def export_vgg19_weights(output_path="vgg19.bin"):
    print("Downloading VGG-19 from PyTorch...")
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    # We only care about the feature extractor
    features = vgg19.features
    
    # Stylor's binary layout:
    #   For each weight tensor (weights then bias), in network layer order:
    #     [uint32_t count] [count × float32 values]
    
    print(f"Exporting to {output_path}...")
    with open(output_path, "wb") as f:
        # Iterate over only Conv2d modules
        for layer in features:
            if isinstance(layer, torch.nn.Conv2d):
                # Write the weights (out_channels, in_channels, kH, kW)
                weight_data = layer.weight.detach().numpy().flatten()
                f.write(struct.pack('I', len(weight_data)))
                f.write(weight_data.tobytes())
                
                # Write the bias (out_channels)
                if layer.bias is not None:
                    bias_data = layer.bias.detach().numpy().flatten()
                    f.write(struct.pack('I', len(bias_data)))
                    f.write(bias_data.tobytes())
                else:
                    # Should not happen in VGG-19 pretrained but handle just in case
                    f.write(struct.pack('I', 0))

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    export_vgg19_weights()
