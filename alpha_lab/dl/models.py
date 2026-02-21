# mypy: ignore-errors

from __future__ import annotations

from typing import Any


def require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "PyTorch is required for dl.model_type=tcn|transformer. "
            "Install with: pip install -e .[dl]"
        ) from exc
    return torch, nn


def build_model(
    model_type: str,
    *,
    input_dim: int,
    lookback: int,
    params: dict[str, object],
) -> Any:
    torch, nn = require_torch()

    hidden_dim = int(params.get("hidden_dim", 32))
    num_layers = int(params.get("num_layers", 2))
    dropout = float(params.get("dropout", 0.1))

    class TCNRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = input_dim
            kernel_size = int(params.get("kernel_size", 3))
            for _ in range(max(1, num_layers)):
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1,
                )
                layers.append(conv)
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_channels = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> Any:
            # x: [batch, lookback, features]
            x = x.transpose(1, 2)
            y = self.backbone(x)
            y = y[:, :, : x.size(-1)]
            return self.head(y[:, :, -1]).squeeze(-1)

    class TransformerRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            nhead = int(params.get("nhead", 4))
            if hidden_dim % nhead != 0:
                nhead = 1
            self.proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
                dim_feedforward=max(hidden_dim * 2, 64),
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
            self.head = nn.Linear(hidden_dim, 1)
            self.lookback = lookback

        def _positional_encoding(self, length: int, dim: int, device: Any) -> Any:
            position = torch.arange(length, device=device).unsqueeze(1)
            scale = -torch.log(torch.tensor(10000.0, device=device)) / dim
            div_term = torch.exp(torch.arange(0, dim, 2, device=device) * scale)
            pe = torch.zeros(length, dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe

        def forward(self, x: Any) -> Any:
            h = self.proj(x)
            pe = self._positional_encoding(h.size(1), h.size(2), h.device)
            h = h + pe.unsqueeze(0)
            y = self.encoder(h)
            return self.head(y[:, -1, :]).squeeze(-1)

    normalized = model_type.lower()
    if normalized == "tcn":
        return TCNRegressor()
    if normalized == "transformer":
        return TransformerRegressor()
    raise ValueError(f"unsupported dl model_type: {model_type}")
