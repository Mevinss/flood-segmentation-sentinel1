"""Бастапқы best.pt файлын комиссияға арналған joblib пакетіне түрлендіру."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import torch
from inference import build_attention_unet, load_export, predict_array
import rasterio


def export_checkpoint(checkpoint_path, destination):
    checkpoint_path = Path(checkpoint_path)
    destination = Path(destination)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["config"]["model"] != "attention_resnet34":
        raise ValueError("Expected attention_resnet34 checkpoint")
    if state["epoch"] + 1 != 34:
        raise ValueError("Expected the original selected epoch 34")
    original = build_attention_unet()
    original.load_state_dict(state["model"], strict=True)
    original.eval()
    with checkpoint_path.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    bundle = {
        "format_version": 1,
        "architecture": "Attention U-Net with ResNet-34",
        "state_dict": {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in state["model"].items()
        },
        "config": state["config"],
        "normalization": state["normalization"],
        "threshold": 0.473,
        "threshold_source": "validation recall matched to baseline threshold 0.5",
        "epoch": int(state["epoch"] + 1),
        "split_ids": state["split_ids"],
        "dataset": "Sen1Floods11 v1.1 HandLabeled",
        "input": {
            "channels": ["VV", "VH"],
            "units": "dB",
            "shape": [2, 512, 512],
            "patch_size": 256,
        },
        "source_checkpoint_sha256": checkpoint_hash,
        "source_drive_id": "1XmFRL-gXDsrKP5me8X1l6xhoIoMRUofk",
        "export_environment": {
            name: importlib.metadata.version(name)
            for name in ["torch", "segmentation-models-pytorch", "numpy", "joblib"]
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary_path, compress=3)
    restored, loaded, device = load_export(temporary_path)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    example = Path(__file__).parent / "data/S1Hand/Bolivia_103757_S1Hand.tif"
    with rasterio.open(example) as source:
        image = source.read().astype("float32")
        valid = (source.read_masks() > 0).all(0) & np.isfinite(image).all(0)
    expected = predict_array(original, image, valid, bundle["normalization"], device)
    actual = predict_array(restored, image, valid, loaded["normalization"], device)
    np.testing.assert_array_equal(expected, actual)
    temporary_path.replace(destination)
    verification = {
        "source_checkpoint_sha256": checkpoint_hash,
        "identical_state_dict": True,
        "identical_predictions": True,
        "max_probability_difference": float(np.max(np.abs(expected - actual))),
        "epoch": bundle["epoch"],
        "threshold": bundle["threshold"],
        "model_bytes": destination.stat().st_size,
    }
    report = Path(__file__).parent / "outputs/checkpoint_export_verification.json"
    report.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models/attention_unet_resnet34.joblib",
    )
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    export_checkpoint(args.checkpoint, args.output)
