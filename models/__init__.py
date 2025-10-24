"""
Models package for domain adaptation research.

Contains modified versions of models from torchvision and custom implementations.
"""

# MaxVit
from .maxvit import MaxVit, MaxVit_T_Weights, maxvit_t

# ResNet
from .resnet import (
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
    wide_resnet50_2,
    wide_resnet101_2,
)

# ConvNeXt
from .convnext import (
    ConvNeXt,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)

__all__ = [
    # MaxVit
    "MaxVit",
    "MaxVit_T_Weights",
    "maxvit_t",
    # ResNet
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    # ConvNeXt
    "ConvNeXt",
    "ConvNeXt_Tiny_Weights",
    "ConvNeXt_Small_Weights",
    "ConvNeXt_Base_Weights",
    "ConvNeXt_Large_Weights",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]
