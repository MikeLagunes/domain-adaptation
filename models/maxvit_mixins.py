"""
Mixins for extending MaxVit with additional capabilities.

This module provides reusable components that can be mixed into MaxVit
models to add domain adaptation, multi-task learning, and other features.
"""

import torch
from torch import nn, Tensor
from .torchvision_utils import GradientReversalLayer


class DomainAdaptationMixin:
    """
    Mixin that adds domain adaptation capabilities to MaxVit.

    This mixin adds:
    - A domain classifier network
    - Gradient reversal layer for adversarial domain adaptation
    - Modified forward pass to output both class and domain predictions

    Usage:
        class MaxVitDA(DomainAdaptationMixin, MaxVit):
            def __init__(self, *args, num_domains=11, **kwargs):
                super().__init__(*args, **kwargs)
                self._setup_domain_adaptation(self.block_channels, num_domains)

    Based on the paper:
    "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
    https://arxiv.org/abs/1505.07818
    """

    def _setup_domain_adaptation(self, block_channels: list[int], num_domains: int = 11):
        """
        Setup domain adaptation components.

        This should be called in the __init__ of the class that uses this mixin,
        after the base model has been initialized.

        Args:
            block_channels: Channel dimensions from the base model (list of ints)
            num_domains: Number of domains to classify (default: 11)
        """
        # Domain classifier (simple MLP)
        # Takes features from the last block and predicts domain labels
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_domains, bias=False)
        )

        # Gradient reversal layer for adversarial training
        self.gradient_reversal = GradientReversalLayer()

    def forward_with_domain(self, x: Tensor, lambda_val: float = 0.0) -> tuple[Tensor, Tensor]:
        """
        Forward pass with domain adaptation.

        Args:
            x: Input tensor of shape [B, C, H, W]
            lambda_val: Lambda value for gradient reversal layer.
                       Controls the trade-off between classification and domain confusion.
                       Typically starts at 0 and increases during training (e.g., 0 -> 1)

        Returns:
            Tuple of (class_predictions, domain_predictions):
                - class_predictions: [B, num_classes] logits for main task
                - domain_predictions: [B, num_domains] logits for domain classification

        Notes:
            - During backprop, gradients from domain_predictions are reversed
            - This encourages domain-invariant features
            - lambda_val should increase gradually during training
        """
        # Extract features through stem and blocks
        # (assumes the base class has self.stem and self.blocks)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        # Domain classification branch with gradient reversal
        # The gradient reversal makes the feature extractor try to confuse the domain classifier
        reversed_features = self.gradient_reversal.apply(x, lambda_val)
        domain_pred = self.domain_classifier(reversed_features)

        # Main classification branch
        # (assumes the base class has self.classifier)
        class_pred = self.classifier(x)

        return class_pred, domain_pred
