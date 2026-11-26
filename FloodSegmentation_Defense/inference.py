"""Дайын Attention U-Net моделін жүктеу және GeoTIFF суретіне қолдану."""

from pathlib import Path
import joblib
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F


class GatedSkip(nn.Module):
    """Декодер контексті арқылы энкодер белгілерін өлшеу."""

    def __init__(self, decoder_channels, skip_channels):
        super().__init__()
        self.decoder_channels = decoder_channels
        hidden_channels = max(skip_channels // 2, 1)
        self.decoder_projection = nn.Conv2d(decoder_channels, hidden_channels, 1)
        self.skip_projection = nn.Conv2d(skip_channels, hidden_channels, 1)
        self.score = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, features):
        decoder = features[:, : self.decoder_channels]
        skip = features[:, self.decoder_channels :]
        combined = self.decoder_projection(decoder) + self.skip_projection(skip)
        attention = self.score(F.relu(combined)).sigmoid()
        return torch.cat((decoder, skip * attention), dim=1)


def build_attention_unet():
    """Сақталған салмақтарға сәйкес архитектураны құру."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None,
    )
    skip_widths = reversed(model.encoder.out_channels[1:-1])
    for stage, skip_width in zip(model.decoder.blocks, skip_widths):
        decoder_width = stage.conv1[0].in_channels - skip_width
        stage.attention1 = GatedSkip(decoder_width, skip_width)
    return model


def load_export(path, device="cpu"):
    """Сенімді joblib пакетінен модель мен алдын ала өңдеуді жүктеу."""
    bundle = joblib.load(path)
    if bundle["format_version"] != 1:
        raise ValueError("Unsupported artifact version")
    if bundle["config"]["model"] != "attention_resnet34":
        raise ValueError("Expected an Attention U-Net with ResNet-34")
    model = build_attention_unet()
    weights = {
        name: torch.from_numpy(array.copy())
        for name, array in bundle["state_dict"].items()
    }
    model.load_state_dict(weights, strict=True)
    device = torch.device(device)
    model.to(device).eval()
    return model, bundle, device


@torch.inference_mode()
def predict_array(model, image, valid, normalization, device):
    """VV/VH dB суретінің төрт патчынан су ықтималдығын есептеу."""
    if image.shape != (2, 512, 512) or valid.shape != (512, 512):
        raise ValueError("Expected image (2,512,512) and valid mask (512,512)")
    if not np.isfinite(image[:, valid]).all():
        raise ValueError("Valid image pixels must be finite")
    mean = np.asarray(normalization["mean"], dtype="float32")[:, None, None]
    std = np.asarray(normalization["std"], dtype="float32")[:, None, None]
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid normalization standard deviation")
    normalized = (image.astype("float32") - mean) / std
    normalized[:, ~valid] = 0
    probability = np.zeros((512, 512), dtype="float32")
    for top in (0, 256):
        for left in (0, 256):
            patch = normalized[:, top : top + 256, left : left + 256]
            tensor = torch.from_numpy(patch.copy())[None].to(device)
            prediction = model(tensor).sigmoid()[0, 0].cpu().numpy()
            probability[top : top + 256, left : left + 256] = prediction
    return probability


def predict_file(model_path, image_path, output_path, device="cpu"):
    """Екі арналы GeoTIFF суретіне су маскасын жазу. Nodata — 255."""
    model, bundle, device = load_export(model_path, device)
    with rasterio.open(image_path) as source:
        image = source.read().astype("float32")
        valid = (source.read_masks() > 0).all(axis=0)
        valid &= np.isfinite(image).all(axis=0)
        profile = source.profile.copy()
    probability = predict_array(model, image, valid, bundle["normalization"], device)
    water = probability >= bundle["threshold"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(np.where(valid, water, 255).astype("uint8"), 1)
    return probability, water, valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention U-Net water segmentation")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    predict_file(arguments.model, arguments.image, arguments.output, arguments.device)
    print(arguments.output)
