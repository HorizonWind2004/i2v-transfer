from __future__ import annotations

from dataclasses import dataclass
import ast
import importlib.util
import sys
import inspect
from pathlib import Path
from typing import Any, Type


@dataclass
class InferConfig:
    dit_ckpt: str | None = None
    dit_ckpt_type: str = "pt"
    dit_arch_config: dict[str, Any] | None = None
    dit_precision: str = "bf16"

    vae_arch_config: dict[str, Any] | None = None
    vae_precision: str = "bf16"

    text_encoder_arch_config: dict[str, Any] | None = None
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048

    scheduler_arch_config: dict[str, Any] | None = None

    training_mode: bool = False
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False
    use_fsdp_inference: bool = False
    cpu_offload: bool = False
    pin_cpu_memory: bool = False


def load_infer_config_class_from_pyfile(file_path: str) -> Type[Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = "_i2v_joyedit_config_" + str(abs(hash(path.resolve())))
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    # Checkpoint Python configs import the standalone `infer_runtime` and `modules` namespaces.
    # Rewrite only the imports inside this config, without registering global aliases.
    tree = ast.parse(path.read_text(), filename=str(path))
    prefixes = ("infer_runtime", "modules")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in prefixes:
                node.module = "i2vt.vendor.joyedit." + node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in prefixes:
                    if not alias.asname:
                        raise ValueError("Use from-imports or explicit aliases for JoyEdit config imports")
                    alias.name = "i2vt.vendor.joyedit." + alias.name
    sys.modules[module_name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is InferConfig:
            continue
        if issubclass(obj, InferConfig):
            return obj

    raise ValueError(f"No class inheriting from 'InferConfig' was found in '{file_path}'.")
