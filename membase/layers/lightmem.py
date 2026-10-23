from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, ClassVar

from .base import MemBaseLayer
from ..configs.lightmem import LightMemConfig
from ..model_types.dataset import Message
from ..model_types.memory import MemoryEntry
from ..utils import PatchSpec, make_attr_patch, token_monitor


def _ensure_lightmem_source_path() -> Path:
    """Resolve the bundled src-layout package independently of cwd/venv setup."""
    source_dir = Path(__file__).resolve().parent / "LightMem" / "src"
    package_dir = source_dir / "lightmem"
    if not (package_dir / "memory" / "lightmem.py").is_file():
        raise ModuleNotFoundError(
            f"Bundled LightMem source is missing at {package_dir}. "
            "Sync the complete membase/layers/LightMem/src/lightmem directory "
            "to the server project before running LightMem."
        )
    # Prioritize this checkout so its prompt/provenance fixes are actually used.
    source_path = str(source_dir)
    if not sys.path or sys.path[0] != source_path:
        if source_path in sys.path:
            sys.path.remove(source_path)
        sys.path.insert(0, source_path)
        importlib.invalidate_caches()
    return source_dir


def _message_dict(message: Message) -> dict[str, str]:
    """Convert one normalized MemBase message to LightMem's input format."""
    return {
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "speaker_name": message.name,
        "speaker_id": message.name,
        "time_stamp": message.timestamp,
    }


def _resolve_lightmem_credentials(lightmem_config: dict[str, Any]) -> None:
    """Resolve credentials in a runtime copy, without saving secrets to memory config.

    Priority: explicit manager key, configured credential file, environment.
    Relative credential paths are rooted at the MemBase checkout, not the cwd.
    """
    manager = lightmem_config.get("memory_manager", {})
    if manager.get("model_name", "openai") != "openai":
        return
    settings = manager.setdefault("configs", {})
    credential_path = settings.pop("api_config_path", None)
    if settings.get("base_url") and not settings.get("openai_base_url"):
        settings["openai_base_url"] = settings.pop("base_url")
    if settings.get("api_key"):
        return
    if credential_path:
        path = Path(credential_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            raise ValueError(f"LightMem credential file not found: {path}")
        try:
            credentials = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Cannot read LightMem credential file as JSON") from None
        if not isinstance(credentials, dict):
            raise ValueError("LightMem credential file must contain a JSON object")
        if "api_keys" in credentials or "base_urls" in credentials:
            keys, urls = credentials.get("api_keys"), credentials.get("base_urls")
            if not isinstance(keys, list) or not isinstance(urls, list) or not keys or len(keys) != len(urls):
                raise ValueError("LightMem credentials require equal nonempty api_keys/base_urls arrays")
            api_key, base_url = keys[0], urls[0]
        else:
            api_key = credentials.get("api_key") or credentials.get("llm_api_key")
            base_url = (credentials.get("openai_base_url") or credentials.get("base_url")
                        or credentials.get("llm_base_url"))
            source_model = credentials.get("llm_model")
            if source_model and source_model != settings.get("model"):
                raise ValueError("LightMem memory model differs from the model in its credential source")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("LightMem credential file has no nonempty API key")
        if not isinstance(base_url, str) or not base_url.startswith(("https://", "http://")):
            raise ValueError("LightMem credential file has no HTTP(S) API endpoint")
        settings["api_key"] = api_key.strip()
        settings["openai_base_url"] = base_url.rstrip("/")
        return
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
        return
    raise ValueError(
        "LightMem memory manager has no API credentials. Set "
        "lightmem_config.memory_manager.configs.api_config_path to an existing "
        "credential JSON, or provide api_key/openai_base_url, or set OPENAI_API_KEY. "
        "QA/judge API configs are not automatically used for memory construction."
    )


class LightMemLayer(MemBaseLayer):
    """Adapter around the source repository's ``LightMemory`` class."""

    layer_type: ClassVar[str] = "LightMem"

    def __init__(self, config: LightMemConfig) -> None:
        self.config = config
        self._init_layer(config)

    @staticmethod
    def _collection_name(user_id: str) -> str:
        digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:20]
        return f"membase_{digest}"

    def _build_lightmem_config(self, config: LightMemConfig) -> dict[str, Any]:
        lightmem_config = copy.deepcopy(config.lightmem_config)
        _resolve_lightmem_credentials(lightmem_config)
        retriever = lightmem_config.setdefault("embedding_retriever", {})
        retriever_configs = retriever.setdefault("configs", {})
        retriever_configs["collection_name"] = self._collection_name(config.user_id)
        retriever_configs["path"] = os.path.join(config.save_dir, "qdrant")
        retriever_configs["on_disk"] = True
        return lightmem_config

    def _init_layer(self, config: LightMemConfig) -> None:
        # Check credentials before importing/loading GPU models.
        runtime_config = self._build_lightmem_config(config)
        _ensure_lightmem_source_path()
        from lightmem.memory.lightmem import LightMemory

        self.memory_layer = LightMemory.from_config(runtime_config)

    def add_message(self, message: Message, **kwargs: Any) -> None:
        self.memory_layer.add_memory(
            messages=[_message_dict(message)],
            force_segment=True,
            force_extract=True,
        )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        if messages:
            self.memory_layer.add_memory(
                messages=[_message_dict(message) for message in messages],
                force_segment=True,
                force_extract=True,
            )

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        memories = self.memory_layer.retrieve(query=query, limit=k, **kwargs)
        return [
            MemoryEntry(content=memory, formatted_content=memory, metadata={"source": "LightMem"})
            for memory in memories
        ]

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError("LightMem deletion is not exposed by the source adapter.")

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        raise NotImplementedError("LightMem updates are managed by its offline update pipeline.")

    def flush(self) -> None:
        return None

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(os.path.join(self.config.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"layer_type": self.layer_type, **self.config.model_dump(mode="python")}, f, ensure_ascii=False, indent=2)
        stats = getattr(self.memory_layer, "get_token_statistics", None)
        if callable(stats):
            with open(os.path.join(self.config.save_dir, "lightmem_token_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats(), f, ensure_ascii=False, indent=2)

    def load_memory(self, user_id: str | None = None) -> bool:
        user_id = user_id or self.config.user_id
        config_path = os.path.join(self.config.save_dir, "config.json")
        if not os.path.exists(config_path):
            return False
        with open(config_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("user_id") != user_id:
            raise ValueError(f"The stored user id ({saved.get('user_id')}) does not match {user_id}.")
        return os.path.isdir(os.path.join(self.config.save_dir, "qdrant"))

    def get_patch_specs(self) -> list[PatchSpec]:
        manager = getattr(self.memory_layer, "manager", None)
        model = self.config.lightmem_config.get("memory_manager", {}).get("configs", {}).get("model")
        if manager is None or not hasattr(manager, "generate_response") or not model:
            return []

        def output_dict(result: Any) -> dict[str, Any]:
            value = result[0] if isinstance(result, tuple) else result
            if isinstance(value, dict):
                value = value.get("content", "")
            return {"messages": value if isinstance(value, (str, list)) else str(value)}

        getter, setter = make_attr_patch(manager, "generate_response")
        return [PatchSpec(
            name=f"{manager.__class__.__name__}.generate_response",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: (model, {}),
                extract_input_dict=lambda *args, **kwargs: {
                    "messages": kwargs.get("messages", args[0] if args else ""),
                    "metadata": {"op_type": "generation, update"},
                },
                extract_output_dict=output_dict,
            ),
        )]

    def cleanup(self) -> None:
        return None
