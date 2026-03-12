from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, token_width: int, token_height: int) -> None:
        super().__init__()
        self.token_height = token_height
        self.token_width = token_width

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = patch_tokens.shape
        expected_tokens = self.token_height * self.token_width
        if num_tokens != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens, received {num_tokens}. "
                "Check image_size and patch_size alignment."
            )

        features = patch_tokens.reshape(batch_size, self.token_height, self.token_width, channels)
        features = features.permute(0, 3, 1, 2).contiguous()
        features = self.stem(features)
        features = self.block(features)
        return self.classifier(features)


def load_backbone(backbone_name: str, device: torch.device) -> torch.nn.Module:
    hub_dir = (Path(__file__).resolve().parents[1] / ".cache" / "torch_hub").resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))

    try:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            skip_validation=True,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load backbone '{backbone_name}'. Ensure the DINOv2 hub model is cached "
            "or internet access is available for the first download."
        ) from exc

    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.to(device)
    return backbone


def extract_patch_tokens(backbone: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = backbone.forward_features(images)
    if "x_norm_patchtokens" not in outputs:
        raise KeyError("Backbone outputs do not contain 'x_norm_patchtokens'.")
    return outputs["x_norm_patchtokens"]
