from __future__ import annotations

import importlib
from typing import cast

from alpha_lab.dl.base import DLPlugin
from alpha_lab.dl.plugins import TemporalMLPStubPlugin

_BUILTIN_PLUGINS: dict[str, DLPlugin] = {
    "temporal_mlp_stub": TemporalMLPStubPlugin(),
}


def get_dl_plugin(name: str) -> DLPlugin:
    if name in _BUILTIN_PLUGINS:
        return _BUILTIN_PLUGINS[name]

    if "." not in name:
        raise ValueError(
            f"Unknown DL plugin: {name}. "
            "Use builtin name or import path like package.module.ClassName"
        )

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"DL plugin class not found: {name}")
    plugin = cls()
    return cast(DLPlugin, plugin)
