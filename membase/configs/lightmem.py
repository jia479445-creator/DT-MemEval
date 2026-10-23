from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import MemBaseConfig


class LightMemConfig(MemBaseConfig):
    """MemBase configuration for the source LightMem implementation."""

    lightmem_config: dict[str, Any] = Field(default_factory=dict)

    def get_llm_models(self) -> list[str]:
        manager = self.lightmem_config.get("memory_manager", {})
        manager_config = manager.get("configs", {}) if isinstance(manager, dict) else {}
        model = manager_config.get("model")
        return [model] if isinstance(model, str) and model else []
