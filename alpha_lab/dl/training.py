from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.config import DLSpec
from alpha_lab.dl.dataset import build_sequence_dataset, resolve_feature_frame
from alpha_lab.dl.models import build_model, require_torch


@dataclass(slots=True)
class DLTrainResult:
    checkpoint_path: Path
    metadata_path: Path
    train_size: int
    val_size: int
    best_val_loss: float


def _split_train_val(n_samples: int, val_split: float) -> tuple[np.ndarray, np.ndarray]:
    val_size = int(round(n_samples * val_split))
    val_size = min(max(val_size, 1), max(n_samples - 1, 1))
    train_size = max(n_samples - val_size, 1)
    indices = np.arange(n_samples)
    return indices[:train_size], indices[train_size:]


def _clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "clone"):
            cloned[key] = value.detach().cpu().clone()
        elif hasattr(value, "copy"):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def train_model(
    spec: DLSpec,
    frame: pd.DataFrame,
    factor_frame: pd.DataFrame,
    artifact_dir: str | Path,
) -> DLTrainResult:
    if spec.model_type not in {"tcn", "transformer"}:
        raise ValueError("dl.model_type must be tcn or transformer in train mode")

    feature_frame = resolve_feature_frame(spec, frame, factor_frame)
    dataset = build_sequence_dataset(spec, frame, feature_frame, include_target=True)
    if dataset.features.shape[0] < 8:
        raise ValueError("not enough DL samples for training")
    if dataset.targets is None:
        raise ValueError("DL training targets are missing")

    torch, _ = require_torch()
    torch.manual_seed(spec.train.seed)
    np.random.seed(spec.train.seed)

    train_idx, val_idx = _split_train_val(dataset.features.shape[0], spec.train.val_split)
    train_x = torch.tensor(dataset.features[train_idx], dtype=torch.float32)
    train_y = torch.tensor(dataset.targets[train_idx], dtype=torch.float32)
    val_x = torch.tensor(dataset.features[val_idx], dtype=torch.float32)
    val_y = torch.tensor(dataset.targets[val_idx], dtype=torch.float32)

    model = build_model(
        spec.model_type,
        input_dim=dataset.features.shape[2],
        lookback=dataset.features.shape[1],
        params=spec.params,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.train.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs = max(1, int(spec.train.epochs))
    batch_size = max(1, int(spec.train.batch_size))

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_x.size(0))
        for start in range(0, train_x.size(0), batch_size):
            batch_idx = permutation[start : start + batch_size]
            pred = model(train_x[batch_idx])
            loss = loss_fn(pred, train_y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = _clone_state_dict(model.state_dict())

    if best_state is None:
        best_state = _clone_state_dict(model.state_dict())

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(spec.checkpoint_path)
        if spec.checkpoint_path
        else artifact_path / f"{spec.model_type}_checkpoint.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_type": spec.model_type,
        "state_dict": best_state,
        "feature_names": dataset.feature_names,
        "lookback": dataset.features.shape[1],
        "params": spec.params,
    }
    torch.save(payload, checkpoint_path)

    metadata = {
        "model_type": spec.model_type,
        "train_size": int(train_x.size(0)),
        "val_size": int(val_x.size(0)),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "feature_names": dataset.feature_names,
    }
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return DLTrainResult(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        train_size=int(train_x.size(0)),
        val_size=int(val_x.size(0)),
        best_val_loss=best_val,
    )


def infer_model(spec: DLSpec, frame: pd.DataFrame, factor_frame: pd.DataFrame) -> pd.Series:
    if spec.model_type not in {"tcn", "transformer"}:
        raise ValueError("dl.model_type must be tcn or transformer in infer mode")
    if not spec.checkpoint_path:
        raise ValueError("dl.checkpoint_path is required in infer mode")

    torch, _ = require_torch()
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DL checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    ckpt_feature_names = payload.get("feature_names", [])
    if ckpt_feature_names and not (spec.features or spec.feature_factors):
        spec_for_infer = replace(spec, features=list(ckpt_feature_names))
    else:
        spec_for_infer = spec

    feature_frame = resolve_feature_frame(spec_for_infer, frame, factor_frame)
    dataset = build_sequence_dataset(spec_for_infer, frame, feature_frame, include_target=False)

    if dataset.features.shape[0] == 0:
        return pd.Series(np.nan, index=frame.index, name=spec.output_name)

    model = build_model(
        spec.model_type,
        input_dim=dataset.features.shape[2],
        lookback=dataset.features.shape[1],
        params=payload.get("params", spec.params),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(dataset.features, dtype=torch.float32)).cpu().numpy()

    out = pd.Series(np.nan, index=frame.index, dtype=float, name=spec.output_name)
    out.loc[dataset.index] = predictions.astype(float)
    return out
