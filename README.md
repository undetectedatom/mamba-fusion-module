# Integrated Mamba Block (IMB)

This repository provides a lightweight and plug-and-play implementation of the **Integrated Mamba Block (IMB)**, a dual-branch module that enhances convolutional networks by integrating **bidirectional Mamba state space modeling** with **depth-wise convolution**, and unifying them via a **flexible spatial-aware feature fusion** mechanism.

## Contents

This repository includes only the core components:

- `mamba_integration.py`: Defines `MambaIntegrationBlock`, the core IMB implementation.
- `feature_fusion.py`: Implements five different fusion strategies:
  - `FusionCrossSpatial` (default, spatial-aware)
  - `FusionSE`
  - `FusionConcat`
  - `FusionAdd`

## Integration Examples

To see how IMB can be integrated into real CNN backbones, refer to the following repositories:

- ResNet-50 Integration:  
  [https://github.com/undetectedatom/resnet50_with_mamba](https://github.com/undetectedatom/resnet50_with_mamba)

- MobileNetV2 Integration:  
  [https://github.com/undetectedatom/mobilenetv2_with_mamba](https://github.com/undetectedatom/mobilenetv2_with_mamba)

Each repository includes:
- Training and evaluation pipeline
- Configurable hyperparameters (e.g., `mamba_d_state`, `fusion_type`, `bidirectional`)
- Examples on CIFAR-100 / ImageNet-100

## How It Works

The `MambaIntegrationBlock` accepts input feature maps and processes them through:

1. A depth-wise convolution branch for local spatial encoding.
2. A bidirectional Mamba sequence modeling branch that operates on both horizontal and vertical flattened views.
3. A spatially-aware feature fusion module that integrates both branches.

The final output is combined with the residual input to enhance both local and global representations.

## Citation

Citation details will be provided here once available.

## Requirements

- PyTorch >= 1.12
- mamba-ssm (https://github.com/state-spaces/mamba)
- Other dependencies (e.g., `torchvision`, `tqdm`) are listed in the example repositories

## Contact

For questions, issues, or discussions, please raise an issue or refer to the integration repositories linked above.
