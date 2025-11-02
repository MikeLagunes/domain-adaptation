"""
Minimal utilities from torchvision for MaxViT model.

This module contains essential utilities extracted from torchvision to support
the MaxViT implementation without requiring torchvision internal APIs.

Original sources:
- torchvision.models._api
- torchvision.models._meta
- torchvision.models._utils
- torchvision.transforms._presets
- torchvision.utils

License: BSD 3-Clause (see TORCHVISION_LICENSE)
Copyright (c) Soumith Chintala 2016
"""

import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, TypeVar, Union
from functools import partial

import torch
from torch.hub import load_state_dict_from_url

# Type variables
M = TypeVar("M", bound=torch.nn.Module)
V = TypeVar("V")
W = TypeVar("W")

# Model registry
BUILTIN_MODELS = {}


# ==================== _api.py ====================

def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    """Register a model builder function."""
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn
    return wrapper


@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method.
        meta (dict[str, Any]): Stores meta-data related to the weights of the model.
    """
    url: str
    transforms: Callable
    meta: dict[str, Any]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False

        if self.meta != other.meta:
            return False

        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                self.transforms.func == other.transforms.func
                and self.transforms.args == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms


class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights.
    Each model building method receives an optional `weights` parameter.
    """

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta


# ==================== _meta.py ====================

# ImageNet-1K categories (simplified - you may want to import from torchvision directly)
_IMAGENET_CATEGORIES = [f"class_{i}" for i in range(1000)]


# ==================== _utils.py ====================

def _ovewrite_named_param(kwargs: dict[str, Any], param: str, new_value: V) -> None:
    """Overwrite a named parameter in kwargs or add it if not present."""
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead."
            )
    else:
        kwargs[param] = new_value


def kwonly_to_pos_or_kw(fn: Callable[..., M]) -> Callable[..., M]:
    """Helper decorator to allow keyword-only arguments as positional."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> M:
        return fn(*args, **kwargs)
    return wrapper


def handle_legacy_interface(**weights: tuple[str, Union[Optional[W], Callable[[dict[str, Any]], Optional[W]]]]):
    """
    Decorates a model builder with the new interface to make it compatible with the old.
    Handles pretrained parameter deprecation.
    """
    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)

                if (
                    (weights_param not in kwargs and pretrained_param not in kwargs)
                    or isinstance(weights_arg, WeightsEnum)
                    or (isinstance(weights_arg, str) and weights_arg != "legacy")
                    or weights_arg is None
                ):
                    continue

                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]

                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated since 0.13 and may be removed in the future, "
                        f"please use '{weights_param}' instead."
                    )

                msg = (
                    f"Arguments other than a weight enum or `None` for '{weights_param}' are deprecated since 0.13 and "
                    f"may be removed in the future. "
                    f"The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.DEFAULT` "
                        f"to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


# ==================== transforms._presets.py ====================

from torchvision.transforms import InterpolationMode
from torchvision.transforms._presets import ImageClassification


# ==================== utils ====================

def _log_api_usage_once(obj: Any) -> None:
    """
    Logs API usage (simplified version).
    In torchvision this tracks usage metrics, here we make it a no-op.
    """
    pass


# ===================== Custom layers or functions =====================\

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(x)
        ctx.lambda_val = lambda_val
        # Detach and clone to avoid in-place issues
        output = x.detach().clone()
        output.requires_grad_(True)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        lambda_val = ctx.lambda_val
        
        # Debug prints (remove in production)
        # print(f"Grad shape: {grad_output.shape}, X shape: {x.shape}")
        # print(f"Lambda: {lambda_val}, Grad min/max: {grad_output.min()}/{grad_output.max()}")
        
        # Ensure shapes match
        assert grad_output.shape == x.shape, f"Shape mismatch: {grad_output.shape} vs {x.shape}"
        
        # Compute reversed gradient
        grad_input = grad_output.neg() * lambda_val
        
        return grad_input, None