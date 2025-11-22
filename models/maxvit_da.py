"""
MaxVit with Domain Adaptation.

This module provides MaxVit models extended with domain adaptation capabilities
for domain-adversarial training.
"""

from typing import Any, Callable, Optional
from torch import nn, Tensor

from .maxvit_orig import MaxVit as MaxVitBase
from .maxvit_mixins import DomainAdaptationMixin


class MaxVitDA(DomainAdaptationMixin, MaxVitBase):
    """
    MaxVit with Domain Adaptation capabilities.

    This extends the base MaxVit model with:
    - Domain classifier network
    - Gradient reversal layer for adversarial domain adaptation
    - Dual output forward pass (class + domain predictions)

    The model implements domain-adversarial training where the feature extractor
    learns to produce domain-invariant features by trying to confuse the domain
    classifier through gradient reversal.

    Args:
        input_size: Size of the input image (H, W)
        stem_channels: Number of channels in the stem
        partition_size: Size of the partitions for attention
        block_channels: Number of channels in each block
        block_layers: Number of layers in each block
        head_dim: Dimension of the attention heads
        stochastic_depth_prob: Probability of stochastic depth
        norm_layer: Normalization function (default: BatchNorm2d with specific params)
        activation_layer: Activation function (default: nn.GELU)
        squeeze_ratio: Squeeze ratio in the SE Layer (default: 0.25)
        expansion_ratio: Expansion ratio in the MBConv bottleneck (default: 4)
        mlp_ratio: Expansion ratio of the MLP layer (default: 4)
        mlp_dropout: Dropout probability for the MLP layer (default: 0.0)
        attention_dropout: Dropout probability for the attention layer (default: 0.0)
        num_classes: Number of classes for main classification task (default: 1000)
        num_domains: Number of domains for domain classification (default: 11)

    Example:
        >>> from models.maxvit_da import MaxVitDA
        >>> model = MaxVitDA(
        ...     input_size=(224, 224),
        ...     stem_channels=64,
        ...     block_channels=[64, 128, 256, 512],
        ...     block_layers=[2, 2, 5, 2],
        ...     head_dim=32,
        ...     stochastic_depth_prob=0.2,
        ...     partition_size=7,
        ...     num_classes=10,
        ...     num_domains=11
        ... )
        >>> class_pred, domain_pred = model(images, lambda_val=0.5)
    """

    def __init__(
        self,
        # Base MaxVit parameters
        input_size: tuple[int, int],
        stem_channels: int,
        partition_size: int,
        block_channels: list[int],
        block_layers: list[int],
        head_dim: int,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        squeeze_ratio: float = 0.25,
        expansion_ratio: float = 4,
        mlp_ratio: int = 4,
        mlp_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        # Domain adaptation parameters
        num_domains: int = 11,
    ) -> None:
        # Initialize base MaxVit
        super().__init__(
            input_size=input_size,
            stem_channels=stem_channels,
            partition_size=partition_size,
            block_channels=block_channels,
            block_layers=block_layers,
            head_dim=head_dim,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
        )

        # Store block_channels for domain adaptation setup
        self.block_channels = block_channels

        # Add domain adaptation features
        self._setup_domain_adaptation(block_channels, num_domains)

    def forward(self, x: Tensor, lambda_val: float = 0.0) -> tuple[Tensor, Tensor]:
        """
        Forward pass with domain adaptation.

        Args:
            x: Input tensor of shape [B, C, H, W]
            lambda_val: Lambda value for gradient reversal (0.0 to 1.0)
                       - At training start: lambda_val ~ 0 (focus on classification)
                       - During training: gradually increase lambda_val
                       - Common schedule: lambda_val = 2/(1+exp(-10*progress)) - 1

        Returns:
            Tuple of (class_predictions, domain_predictions):
                - class_predictions: [B, num_classes] logits
                - domain_predictions: [B, num_domains] logits

        Example:
            >>> # Training loop
            >>> for epoch in range(num_epochs):
            ...     for batch_idx, (images, labels, domains) in enumerate(dataloader):
            ...         # Calculate lambda schedule
            ...         p = (epoch * len(dataloader) + batch_idx) / (num_epochs * len(dataloader))
            ...         lambda_val = 2 / (1 + np.exp(-10 * p)) - 1
            ...
            ...         # Forward pass
            ...         class_pred, domain_pred = model(images, lambda_val)
            ...
            ...         # Compute losses
            ...         class_loss = criterion(class_pred, labels)
            ...         domain_loss = criterion(domain_pred, domains)
            ...         total_loss = class_loss + domain_loss
        """
        return self.forward_with_domain(x, lambda_val)


def maxvit_t_da(
    *,
    weights: Optional[str] = None,
    num_classes: int = 1000,
    num_domains: int = 11,
    input_size: tuple[int, int] = (224, 224),
    **kwargs: Any
) -> MaxVitDA:
    """
    Constructs a MaxVit-T architecture with domain adaptation.

    This is the "Tiny" variant of MaxVit with domain adaptation capabilities.

    Args:
        weights: Path to pretrained weights file (optional)
                Note: Domain adaptation layers will be randomly initialized
        num_classes: Number of classes for main classification task
        num_domains: Number of domains for domain classification
        input_size: Input image size (H, W)
        **kwargs: Additional parameters passed to MaxVitDA

    Returns:
        MaxVitDA model instance

    Example:
        >>> # Create model from scratch
        >>> model = maxvit_t_da(num_classes=10, num_domains=5)
        >>>
        >>> # Load pretrained backbone (domain layers will be random)
        >>> model = maxvit_t_da(
        ...     weights='path/to/maxvit_t_weights.pth',
        ...     num_classes=10,
        ...     num_domains=5
        ... )
    """
    model = MaxVitDA(
        stem_channels=64,
        block_channels=[64, 128, 256, 512],
        block_layers=[2, 2, 5, 2],
        head_dim=32,
        stochastic_depth_prob=0.2,
        partition_size=7,
        input_size=input_size,
        num_classes=num_classes,
        num_domains=num_domains,
        **kwargs,
    )

    # Load pretrained weights if provided
    # Note: This will load the backbone weights, but domain adaptation
    # layers will remain randomly initialized (which is what you want)
    if weights is not None:
        import torch
        state_dict = torch.load(weights, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {weights} (strict=False)")

    return model


__all__ = ["MaxVitDA", "maxvit_t_da"]
