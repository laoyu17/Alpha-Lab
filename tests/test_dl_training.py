from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd

from alpha_lab.config import DLSpec, DLTrainSpec
from alpha_lab.dl import training
from alpha_lab.dl.dataset import DLDataset


class FakeTensor:
    def __init__(self, data: Any):
        self.data = np.asarray(data, dtype=float)

    def size(self, dim: int | None = None):
        if dim is None:
            return self.data.shape
        return self.data.shape[dim]

    def __getitem__(self, idx: Any) -> FakeTensor:
        if isinstance(idx, FakeTensor):
            idx = idx.data.astype(int)
        return FakeTensor(self.data[idx])

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def backward(self) -> None:
        return None

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def clone(self) -> FakeTensor:
        return FakeTensor(self.data.copy())


class FakeOptimizer:
    def __init__(self, params: list[Any], lr: float):  # noqa: ARG002
        self.model = params[0]

    def zero_grad(self) -> None:
        return None

    def step(self) -> None:
        self.model.step += 1


class FakeMSELoss:
    _val_map = {1: 0.5, 2: 0.1, 3: 0.3}

    def __call__(self, pred: FakeTensor, target: FakeTensor) -> FakeTensor:  # noqa: ARG002
        step = int(round(pred.item()))
        value = self._val_map.get(step, 1.0)
        return FakeTensor([value])


class FakeModel:
    def __init__(self) -> None:
        self.step = 0

    def parameters(self) -> list[object]:
        return [self]

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def __call__(self, x: FakeTensor) -> FakeTensor:
        return FakeTensor(np.full((x.size(0),), float(self.step)))

    def state_dict(self) -> dict[str, FakeTensor]:
        return {"step": FakeTensor([float(self.step)])}


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:  # noqa: ANN001
        return False


class FakeTorch:
    float32 = "float32"

    def __init__(self) -> None:
        self.saved_payload: dict[str, Any] | None = None
        self.optim = SimpleNamespace(Adam=FakeOptimizer)
        self.nn = SimpleNamespace(MSELoss=FakeMSELoss)

    def manual_seed(self, seed: int) -> None:  # noqa: ARG002
        return None

    def tensor(self, arr: Any, dtype: str | None = None) -> FakeTensor:  # noqa: ARG002
        return FakeTensor(arr)

    def randperm(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=int)

    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()

    def save(self, payload: dict[str, Any], path: Path) -> None:
        self.saved_payload = payload
        Path(path).write_bytes(b"fake-checkpoint")


def _mock_dataset(frame: pd.DataFrame) -> DLDataset:
    sample_count = 10
    features = np.ones((sample_count, 4, 2), dtype=np.float32)
    targets = np.zeros(sample_count, dtype=np.float32)
    tuples = [frame.index[i] for i in range(sample_count)]
    index = pd.MultiIndex.from_tuples(tuples, names=["datetime", "symbol"])
    return DLDataset(
        features=features,
        targets=targets,
        index=index,
        feature_names=["f1", "f2"],
    )


def test_train_model_saves_best_validation_checkpoint(monkeypatch, tmp_path: Path) -> None:
    fake_torch = FakeTorch()

    monkeypatch.setattr(training, "require_torch", lambda: (fake_torch, None))
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        training,
        "resolve_feature_frame",
        lambda spec, frame, factor_frame: pd.DataFrame(index=frame.index),  # noqa: ARG005
    )
    monkeypatch.setattr(
        training,
        "build_sequence_dataset",
        lambda spec, frame, feature_frame, include_target: _mock_dataset(frame),  # noqa: ARG005
    )

    dates = pd.bdate_range("2024-01-02", periods=5)
    symbols = ["AAA", "BBB"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    frame = pd.DataFrame({"close": 100.0}, index=idx)
    factor_frame = pd.DataFrame(index=idx)

    spec = DLSpec(
        enabled=True,
        mode="train",
        model_type="tcn",
        params={},
        train=DLTrainSpec(epochs=3, batch_size=64, val_split=0.2, seed=7),
    )

    result = training.train_model(spec, frame, factor_frame, artifact_dir=tmp_path)

    assert fake_torch.saved_payload is not None
    saved_step = int(fake_torch.saved_payload["state_dict"]["step"].item())
    assert saved_step == 2
    assert result.best_val_loss == 0.1

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["best_epoch"] == 2
