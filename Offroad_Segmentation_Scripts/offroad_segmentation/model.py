from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_MODEL_TYPE = "dinov2"
SUPPORTED_MODEL_TYPES = {"dinov2", "segformer_b0", "deeplabv3plus"}
SEGFORMER_DEFAULT_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEEPLAB_DEFAULT_ENCODER_NAME = "mobilenet_v2"
DEEPLAB_DEFAULT_ENCODER_WEIGHTS = "imagenet"


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


class DinoV2SegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        num_classes: int,
        image_size: tuple[int, int],
        patch_size: int,
        freeze_encoder: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model_type = "dinov2"
        self.backbone_name = backbone_name
        self.patch_size = patch_size
        self.token_grid = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.freeze_encoder = freeze_encoder
        self.backbone = load_dino_backbone(backbone_name, device)
        embedding_dim = infer_dino_embedding_dim(self.backbone, image_size, patch_size, device)
        self.head = SegmentationHeadConvNeXt(
            in_channels=embedding_dim,
            out_channels=num_classes,
            token_width=self.token_grid[1],
            token_height=self.token_grid[0],
        )

        if freeze_encoder:
            freeze_module(self.backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.backbone.eval()
            with torch.no_grad():
                patch_tokens = extract_patch_tokens(self.backbone, images)
        else:
            patch_tokens = extract_patch_tokens(self.backbone, images)
        return self.head(patch_tokens)

    def legacy_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "head_state_dict": self.head.state_dict(),
            "embedding_dim": int(self.head.stem[0].in_channels),
            "token_grid": list(self.token_grid),
        }


class SegFormerSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        model_name: str,
        freeze_encoder: bool,
    ) -> None:
        super().__init__()
        self.model_type = "segformer_b0"
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder

        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "SegFormer support requires the 'transformers' package."
            ) from exc

        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            self.pretrained_source = model_name
        except Exception:
            config = SegformerConfig(num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)
            self.pretrained_source = None

        if freeze_encoder:
            freeze_module(self.model.segformer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        return outputs.logits


class DeepLabV3PlusSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        encoder_name: str,
        encoder_weights: str | None,
        freeze_encoder: bool,
    ) -> None:
        super().__init__()
        self.model_type = "deeplabv3plus"
        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder
        self.freeze_batchnorm = True

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "DeepLabV3+ support requires the 'segmentation_models_pytorch' package."
            ) from exc

        resolved_weights = None if encoder_weights in (None, "", "none", "None") else encoder_weights
        try:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=resolved_weights,
                classes=num_classes,
                activation=None,
            )
            self.encoder_weights = resolved_weights
        except Exception:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                classes=num_classes,
                activation=None,
            )
            self.encoder_weights = None

        if freeze_encoder:
            freeze_module(self.model.encoder)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batchnorm:
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if self.freeze_encoder:
            self.model.encoder.eval()
        return self


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def resolve_model_type(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type", DEFAULT_MODEL_TYPE)).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of {sorted(SUPPORTED_MODEL_TYPES)}.")
    return model_type


def load_dino_backbone(backbone_name: str, device: torch.device) -> torch.nn.Module:
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
    backbone.to(device)
    return backbone


def infer_dino_embedding_dim(
    backbone: torch.nn.Module,
    image_size: tuple[int, int],
    patch_size: int,
    device: torch.device,
) -> int:
    for attribute_name in ("embed_dim", "num_features"):
        value = getattr(backbone, attribute_name, None)
        if isinstance(value, int):
            return value

    dummy = torch.zeros((1, 3, image_size[0], image_size[1]), device=device)
    with torch.no_grad():
        patch_tokens = extract_patch_tokens(backbone, dummy)
    expected_tokens = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    if patch_tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} DINOv2 tokens for image size {image_size}, got {patch_tokens.shape[1]}."
        )
    return int(patch_tokens.shape[-1])


def extract_patch_tokens(backbone: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = backbone.forward_features(images)
    if "x_norm_patchtokens" not in outputs:
        raise KeyError("Backbone outputs do not contain 'x_norm_patchtokens'.")
    return outputs["x_norm_patchtokens"]


def build_segmentation_model(config: dict[str, Any], *, num_classes: int, device: torch.device) -> nn.Module:
    model_type = resolve_model_type(config)
    freeze_encoder = bool(config.get("freeze_encoder", True))

    if model_type == "dinov2":
        model = DinoV2SegmentationModel(
            backbone_name=str(config.get("backbone_name", "dinov2_vits14")),
            num_classes=num_classes,
            image_size=tuple(int(value) for value in config["image_size"]),
            patch_size=int(config.get("patch_size", 14)),
            freeze_encoder=freeze_encoder,
            device=device,
        )
    elif model_type == "segformer_b0":
        model = SegFormerSegmentationModel(
            num_classes=num_classes,
            model_name=str(config.get("segformer_model_name", SEGFORMER_DEFAULT_MODEL_NAME)),
            freeze_encoder=freeze_encoder,
        )
    elif model_type == "deeplabv3plus":
        model = DeepLabV3PlusSegmentationModel(
            num_classes=num_classes,
            encoder_name=str(config.get("deeplab_encoder_name", DEEPLAB_DEFAULT_ENCODER_NAME)),
            encoder_weights=config.get("deeplab_encoder_weights", DEEPLAB_DEFAULT_ENCODER_WEIGHTS),
            freeze_encoder=freeze_encoder,
        )
    else:
        raise AssertionError(f"Unhandled model_type '{model_type}'.")

    return model.to(device)


def forward_model_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        if "out" in outputs:
            return outputs["out"]
    raise TypeError(f"Could not resolve logits from model output of type {type(outputs)!r}.")


def get_trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def model_descriptor(config: dict[str, Any]) -> str:
    model_type = resolve_model_type(config)
    if model_type == "dinov2":
        return str(config.get("backbone_name", "dinov2_vits14"))
    if model_type == "segformer_b0":
        return str(config.get("segformer_model_name", SEGFORMER_DEFAULT_MODEL_NAME))
    if model_type == "deeplabv3plus":
        encoder_name = str(config.get("deeplab_encoder_name", DEEPLAB_DEFAULT_ENCODER_NAME))
        return f"deeplabv3plus/{encoder_name}"
    return model_type


def load_model_weights(model: nn.Module, checkpoint: dict[str, Any] | torch.Tensor) -> None:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return

    if isinstance(model, DinoV2SegmentationModel) and isinstance(checkpoint, dict) and "head_state_dict" in checkpoint:
        model.head.load_state_dict(checkpoint["head_state_dict"])
        return

    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def checkpoint_metadata_for_model(model: nn.Module) -> dict[str, Any]:
    metadata = {"model_state_dict": model.state_dict()}
    if isinstance(model, DinoV2SegmentationModel):
        metadata.update(model.legacy_checkpoint_metadata())
    return metadata
