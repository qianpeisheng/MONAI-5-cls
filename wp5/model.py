from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from monai.bundle import ConfigParser
from monai.networks.nets import UNet, BasicUNet


def build_model(arch: str = "basicunet") -> torch.nn.Module:
    if arch == "basicunet":
        return BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class WP5HeadWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, out_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def build_model_from_bundle(bundle_dir: Path, out_channels: int = 5) -> torch.nn.Module:
    config_file = bundle_dir / "configs" / "inference.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Bundle inference.json not found: {config_file}")
    parser = ConfigParser()
    parser.read_config(str(config_file))
    parser.update(pairs={"bundle_root": str(bundle_dir)})
    try:
        net = parser.get_parsed_content("network")
    except Exception:
        net = parser.get_parsed_content("network_def")
    last_conv_out = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv3d):
            last_conv_out = module.out_channels
    if last_conv_out is None:
        raise RuntimeError("Could not determine backbone output channels (no Conv3d found).")
    return WP5HeadWrapper(net, in_channels=last_conv_out, out_channels=out_channels)


def load_pretrained_non_strict(net: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    sd_raw = torch.load(ckpt_path, map_location=device)
    candidate_keys = ["state_dict", "model_state_dict", "network", "net"]
    if isinstance(sd_raw, dict):
        for k in candidate_keys:
            if k in sd_raw and isinstance(sd_raw[k], dict):
                sd = sd_raw[k]
                break
        else:
            sd = sd_raw
    else:
        sd = sd_raw

    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in d.items()}

    sd_try = sd
    for prefix in ("module.", "model."):
        if any(k.startswith(prefix) for k in sd_try.keys()):
            sd_try = strip_prefix(sd_try, prefix)

    net_keys = list(net.state_dict().keys())
    if net_keys:
        sample = net_keys[0]
        tokens = sample.split('.')
        prefix_tokens = []
        for t in tokens:
            if t.isdigit():
                break
            if t in {"weight", "bias", "running_mean", "running_var"}:
                break
            prefix_tokens.append(t)
        composite_prefix = '.'.join(prefix_tokens)
        if composite_prefix and not all(k.startswith(composite_prefix + '.') for k in sd_try.keys()):
            sd_try = {f"{composite_prefix}." + k if not k.startswith(composite_prefix + '.') else k: v for k, v in sd_try.items()}

    model_sd = net.state_dict()
    sd_filtered = {}
    for k, v in sd_try.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            sd_filtered[k] = v
    net.load_state_dict(sd_filtered, strict=False)


def reinitialize_weights(model: torch.nn.Module) -> None:
    for m in model.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass

