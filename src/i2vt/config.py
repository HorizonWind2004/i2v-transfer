"""Typed, dependency-light configuration loading and serialization.

A YAML file sets only the keys that differ from the defaults below. Loading fills in the rest, and
the resolved configuration a run writes holds every value.
"""

from __future__ import annotations

import os
import re
import types
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


def _convert(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Any:
        return value
    if origin in (Union, types.UnionType):
        for candidate in args:
            try:
                return _convert(value, candidate, path)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"{path}: invalid value {value!r}")
    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a mapping")
        hints = get_type_hints(annotation)
        unknown = value.keys() - hints.keys()
        if unknown:
            raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
        try:
            return annotation(**{k: _convert(v, hints[k], f"{path}.{k}") for k, v in value.items()})
        except TypeError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a list")
        return [_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a mapping")
        return {k: _convert(v, args[1], f"{path}.{k}") for k, v in value.items()}
    if annotation is float and type(value) in (float, int):
        return float(value)
    if type(value) is not annotation:
        raise ValueError(f"{path}: expected {annotation.__name__}, got {type(value).__name__}")
    return value


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment(v) for v in value]
    if isinstance(value, str):
        value = os.path.expandvars(value)
        if re.search(r"\$\{?[A-Za-z_]", value):
            raise ValueError(f"Unresolved environment variable in {value!r}")
    return value


class ConfigMixin:
    """Serialize our configuration objects; independent of HF model serialization."""

    @classmethod
    def from_dict(cls, data: dict, *, expand: bool = False):
        return _convert(expand_environment(data) if expand else data, cls, cls.__name__)

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle), expand=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


# Edit types with their own training loss curve and teacher CFG. The per-type loss vector is
# all-reduced across ranks, so its length and order are fixed here; the last slot collects the rest.
EDIT_TYPES = (
    "background_change",
    "creative_edit",
    "global_style",
    "local_add",
    "local_change",
    "local_remove",
    "subtitle_edit",
    "action_edit",
    "relight",
    "expression_edit",
    "motion_edit",
    "other",
)

MODES = ("train", "eval", "train_eval")
MODE_ALIASES = {"train_only": "train", "openve_only": "eval", "train_then_openve": "train_eval"}
OBJECTIVES = ("online", "offline")
OBJECTIVE_ALIASES = {"opd": "online", "sft": "offline"}
TEACHER_BACKENDS = ("joyedit", "offline")
TEACHER_ALIASES = {"offline_target": "offline"}


def resolve_loss_steps(spec: str | None, num_steps: int) -> list[int]:
    """Parse a loss_steps setting into the rollout step indices that receive gradients.

    The trajectory always runs all num_steps steps. Excluded steps still advance the scheduler and
    only skip the loss, so "first:5" and "last:5" see the same states and differ only in where they
    are supervised.
    """
    spec = (spec or "all").strip().lower()
    if spec in {"", "all"}:
        return list(range(num_steps))
    kind, _, raw = spec.partition(":")
    if kind not in {"first", "last"} or not raw.isdigit():
        raise ValueError(f"Unknown loss_steps: {spec!r} (expected 'all', 'first:N' or 'last:N')")
    n = int(raw)
    if not 1 <= n <= num_steps:
        raise ValueError(f"loss_steps {spec!r} out of range for {num_steps} rollout steps")
    return list(range(n)) if kind == "first" else list(range(num_steps - n, num_steps))


# Training configuration.


@dataclass(kw_only=True)
class RolloutConfig(ConfigMixin):
    """The student's own denoising trajectory during online training."""

    num_inference_steps: int = 10
    chunk_size: int = 1
    """Rollout steps whose teacher predictions are collected before their backward passes."""
    train_steps: int = 0
    """Truncate the trajectory to its first train_steps steps; 0 runs all of them."""
    timestep_mode: str = "scheduler"
    timestep_min: int = 1
    timestep_max: int = 0
    guidance_scale: float = 1.0
    """CFG of the prediction that advances the trajectory."""
    student_guidance_scale: float = 1.0
    """CFG of the prediction that enters the loss."""
    kwargs: dict[str, Any] = field(default_factory=dict)
    loss_steps: str = "all"
    """Rollout steps that enter the loss: "all", "first:N" (high noise) or "last:N" (low noise).

    Unlike train_steps, the trajectory still runs all num_inference_steps steps.
    """


@dataclass(kw_only=True)
class ValidationConfig(ConfigMixin):
    """Preview edits of training images, decoded from the student's rollout."""

    steps: int = 25
    first_step: bool = True
    samples: int = 4
    samples_per_rank: int = 1
    edit_types: list[str] = field(default_factory=list)
    samples_per_type: int = 0
    start_index: int = 0
    seed: int = 20260528
    num_inference_steps: int = 10
    prompts: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class LoggingConfig(ConfigMixin):
    project: str = "i2vt"
    mode: str = "offline"
    api_key_file: str | None = None


@dataclass(kw_only=True)
class OfflineConfig(ConfigMixin):
    """How offline training draws its noise levels.

    Offline training supervises the student on noisy versions of the stored teacher edit, with no
    rollout. Sigma comes from a continuous density pushed through the sampler's shift, so training
    covers every noise level inference visits.
    """

    sigma_sampling: str = "uniform"
    """uniform or logit_normal, both pushed through the sampler's own shift.

    The sampler builds its grid as linspace(1, 1/N) pushed through that shift, so a uniform draw
    follows the density inference walks. logit_normal composed with a shift of 5 puts most draws at
    high noise.
    """

    logit_mean: float = 0.0
    logit_std: float = 1.0
    draws_per_sample: int = 1
    """Noise levels drawn per sample; gradient accumulation sets the number of samples."""

    sigma_min: float = 1e-3
    """The teacher's (x_t - x0)/sigma is undefined at 0 and explodes near it."""


@dataclass(kw_only=True)
class TrainingConfig(ConfigMixin):
    learning_rate: float
    max_steps: int
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int
    save_steps: int = 50
    log_steps: int = 1
    seed: int = 2026
    check_vae_compat: bool = True
    offload_optimizer: bool = False
    """Keep the AdamW state and the fp32 master copy of the trainable weights in CPU memory.

    The update is the same fp32 AdamW; only where it runs changes. It saves 8 bytes of GPU memory per trainable
    parameter (the two Adam moments), about 10 GB for I2VT-14B, and costs a few seconds per optimizer step.
    """
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    loss_type: str = "v"
    """v (v-MSE, used by every released model), v_cos, x (x0-MSE) or x_cos."""
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    objective: str = "online"
    """Where the training target comes from.

    "online": the student rolls out and the teacher gives the velocity at each latent the student
    reaches. "offline": the noisy latent is built from the stored teacher edit and the target
    velocity is exact, with no rollout. Both share the loss types, accumulation, logging and
    checkpointing.
    """

    deterministic: bool = False
    """Make two runs of one configuration produce identical losses.

    Seeds every generator the loop draws from, pins cuDNN and cuBLAS to deterministic kernels and
    gives the dataloader workers a derived seed. Reproducibility also requires the same rank count,
    accumulation and chunk size: DistributedSampler splits the epoch by world size, and the order of
    gradient reductions follows it.
    """


# Experiment configuration.


@dataclass(kw_only=True)
class RunConfig(ConfigMixin):
    name: str
    mode: str = "train"
    """train, eval (evaluate a checkpoint) or train_eval (train, then evaluate the final checkpoint)."""
    output_dir: str = "outputs"
    cache_dir: str | None = None
    """Latent and prompt caches; defaults to cache/<name>."""
    inference_dir: str = "outputs/inference"


@dataclass(kw_only=True)
class RuntimeConfig(ConfigMixin):
    student_gpus: list[int]
    """GPUs for training. Without teacher_gpus each one holds a student rank and its teacher."""
    teacher_gpus: list[int] | None = None
    """GPUs that serve the teacher over RPC, one server each, disjoint from student_gpus.

    Set this when a card cannot hold the student and the teacher together (A100 40/80 GB for the
    14B and CoF models). Left unset, every student GPU also holds its own teacher (B300).
    """
    eval_gpus: list[int] | None = None
    """GPUs for evaluation, one inference worker each; defaults to student_gpus."""


@dataclass(kw_only=True)
class DataConfig(ConfigMixin):
    train_dataset: str = "hf:i2vt_17k"
    base_path: str = "/"
    """Root for relative image paths in a CSV training set."""
    height: int = 512
    width: int = 512
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    resize_mode: str = "area"
    """area keeps the source aspect ratio at about height*width pixels, as evaluation does;
    squash resizes every frame to width x height."""


@dataclass(kw_only=True)
class StudentConfig(ConfigMixin):
    backend: str
    model_path: str
    lora_path: str | None = None
    lora_rank: int = 128
    lora_alpha: float = 64.0
    lora_target: str = "all"
    weight_dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    cache_size: int = 8
    max_sequence_length: int = 512
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    lora_path_2: str | None = None
    """Initial LoRA of the low-noise expert of a two-expert model."""
    tune_mode: str = "lora"
    """lora trains an adapter over the frozen base; full trains the transformer itself and ignores
    the lora_* fields."""


@dataclass(kw_only=True)
class TeacherConfig(ConfigMixin):
    """The image editor that supervises the student. It runs on each student GPU."""

    backend: str = "joyedit"
    model_path: str | None = None
    guidance_scale: float = 1.0
    cache_size: int = 16
    model_kwargs: dict[str, Any] = field(default_factory=lambda: {"config_path": None})
    guidance_scale_by_type: dict[str, float] | None = None
    """Teacher CFG per edit type. Types not listed use guidance_scale."""
    host: str = "127.0.0.1"
    """RPC only: where the teacher servers listen and the student ranks connect."""
    ports: list[int] | None = None
    """RPC only: one TCP port per teacher GPU (default 29600, 29601, ...)."""
    client_workers: int = 0
    """RPC only: requests in flight per student rank (0 = one per teacher)."""
    rpc_timeout: float | None = None
    """RPC only: seconds before a teacher call fails (None = wait)."""
    guidance_scale_by_task: dict[str, float] | None = None
    """Teacher CFG per sub-task (metadata.task, e.g. smile). Overrides guidance_scale_by_type.

    Expression and action edits share the edit type action_edit, so only the sub-task tells them apart.
    """


@dataclass(kw_only=True)
class InferenceConfig(ConfigMixin):
    """Video editing settings. Unset values take the student backend's defaults."""

    num_inference_steps: int | None = None
    guidance_scale: float = 1.0
    seed: int = 42
    fps: int = 16
    num_frames: int | None = None
    source_frames: int | None = None
    memory_mode: str = "full"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ScoringConfig(ConfigMixin):
    enabled: bool = True
    model: str | None = None
    """Gemini judge; defaults to the benchmark's judge (see data/registry.py)."""
    api_key_file: str | None = None
    """File with one Gemini API key per line; otherwise GEMINI_API_KEY or GOOGLE_API_KEY."""


@dataclass(kw_only=True)
class EvalConfig(ConfigMixin):
    bench: str = "openve"
    """Benchmark name: openve or openve_pp."""
    bench_dir: str | None = None
    bench_csv: str | None = None
    """Local benchmark files; downloaded from the Hub when unset."""
    checkpoint_path: str | None = None
    """LoRA to evaluate; defaults to the final checkpoint of this run."""
    filter_types: list[str] = field(default_factory=list)
    """Edit types to evaluate; defaults to the benchmark's."""
    limit: int | None = None
    skip_existing: bool = True
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    skip_checkpoint: bool = False
    """Evaluate the base editor with no I2VT LoRA (eval mode only).

    The backend's own weights, including VideoCoF's student.lora_path, still load.
    """


@dataclass(kw_only=True)
class ExperimentConfig(ConfigMixin):
    run: RunConfig
    runtime: RuntimeConfig
    data: DataConfig = field(default_factory=DataConfig)
    student: StudentConfig
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    train: TrainingConfig
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        self.run.mode = MODE_ALIASES.get(self.run.mode, self.run.mode)
        self.train.objective = OBJECTIVE_ALIASES.get(self.train.objective, self.train.objective)
        self.teacher.backend = TEACHER_ALIASES.get(self.teacher.backend, self.teacher.backend)
        if self.run.cache_dir is None:
            self.run.cache_dir = f"cache/{self.run.name}"
        if self.runtime.eval_gpus is None:
            self.runtime.eval_gpus = list(self.runtime.student_gpus)
        from i2vt.data.registry import benchmark

        bench = benchmark(self.eval.bench)
        if not self.eval.filter_types:
            self.eval.filter_types = list(bench.filter_types)
        if self.eval.scoring.model is None:
            self.eval.scoring.model = bench.judge
        if self.student.backend not in {"bernini", "videocof"}:
            raise ValueError(f"student.backend: expected bernini or videocof, got {self.student.backend!r}")
        from i2vt.models import validate_backend_config

        # Fills the backend's defaults into the kwargs and inference fields, then checks them.
        validate_backend_config(self)
        self.validate()

    def validate(self):
        if self.run.mode not in MODES:
            raise ValueError(f"run.mode: expected one of {MODES}, got {self.run.mode!r}")
        if self.eval.skip_checkpoint and self.run.mode != "eval":
            raise ValueError("eval.skip_checkpoint: only meaningful with run.mode eval")
        if self.teacher.backend not in TEACHER_BACKENDS:
            raise ValueError(f"teacher.backend: expected one of {TEACHER_BACKENDS}, got {self.teacher.backend!r}")
        if self.teacher.backend == "joyedit" and not self.teacher.model_path:
            raise ValueError("teacher.model_path: required for the joyedit teacher")
        if self.train.objective not in OBJECTIVES:
            raise ValueError(f"train.objective: expected online or offline, got {self.train.objective!r}")
        if (self.train.objective == "offline") != (self.teacher.backend == "offline"):
            # Offline training against a live teacher would regress onto its answer for a latent
            # built from the stored edit, and the offline teacher knows nothing of a rollout.
            raise ValueError("train.objective offline requires teacher.backend offline, and vice versa")
        offline = self.train.offline
        if offline.sigma_sampling not in {"logit_normal", "uniform"}:
            raise ValueError(f"train.offline.sigma_sampling: unsupported {offline.sigma_sampling!r}")
        if not 0.0 < offline.sigma_min < 1.0:
            raise ValueError("train.offline.sigma_min: must lie in (0, 1)")
        if offline.draws_per_sample <= 0:
            raise ValueError("train.offline.draws_per_sample: must be positive")
        if self.train.objective == "offline" and (self.train.rollout.loss_steps or "all").strip().lower() != "all":
            raise ValueError("train.rollout.loss_steps: offline training supervises every draw, so it must be 'all'")
        if self.data.resize_mode not in {"area", "squash"}:
            raise ValueError(f"data.resize_mode: expected 'area' or 'squash', got {self.data.resize_mode!r}")
        by_task = self.teacher.guidance_scale_by_task
        if by_task and any(float(v) <= 0 for v in by_task.values()):
            raise ValueError("teacher.guidance_scale_by_task: every scale must be positive")
        by_type = self.teacher.guidance_scale_by_type
        if by_type:
            unknown = sorted(set(by_type) - set(EDIT_TYPES))
            if unknown:
                raise ValueError(f"teacher.guidance_scale_by_type: unknown edit types {unknown}")
            if any(float(v) <= 0 for v in by_type.values()):
                raise ValueError("teacher.guidance_scale_by_type: every scale must be positive")
        if self.train.loss_type not in {"v", "v_cos", "x", "x_cos"}:
            raise ValueError("train.loss_type: unsupported loss")
        if self.train.learning_rate <= 0 or self.train.max_steps <= 0:
            raise ValueError("Learning rate and max_steps must be positive")
        if self.train.save_steps <= 0 or self.train.gradient_accumulation_steps <= 0:
            raise ValueError("Save interval and gradient accumulation must be positive")
        if self.train.rollout.num_inference_steps <= 0 or self.train.rollout.chunk_size <= 0:
            raise ValueError("Rollout steps and chunk size must be positive")
        if not 0 <= self.train.rollout.train_steps <= self.train.rollout.num_inference_steps:
            raise ValueError("Invalid rollout train_steps")
        if self.train.rollout.timestep_mode not in {"scheduler", "uniform_random", "uniform_grid"}:
            raise ValueError("Unknown timestep_mode")
        resolve_loss_steps(self.train.rollout.loss_steps, self.train.rollout.num_inference_steps)
        for devices in (self.runtime.student_gpus, self.runtime.eval_gpus):
            if not devices or len(devices) != len(set(devices)) or min(devices) < 0:
                raise ValueError("GPU lists must be nonempty, unique, and nonnegative")
        if self.runtime.teacher_gpus is not None:
            teachers = self.runtime.teacher_gpus
            if not teachers or len(teachers) != len(set(teachers)) or min(teachers) < 0:
                raise ValueError("runtime.teacher_gpus must be nonempty, unique, and nonnegative")
            if set(teachers) & set(self.runtime.student_gpus):
                raise ValueError("runtime.teacher_gpus and runtime.student_gpus overlap")
            if self.teacher.backend != "joyedit":
                raise ValueError("runtime.teacher_gpus (RPC) supports the joyedit teacher only")
            if self.teacher.ports is None:
                self.teacher.ports = [29600 + i for i in range(len(teachers))]
            if len(self.teacher.ports) != len(teachers):
                raise ValueError("teacher.ports: one port per teacher GPU is required")
            if len(set(self.teacher.ports)) != len(self.teacher.ports) or any(
                not 0 < p < 65536 for p in self.teacher.ports
            ):
                raise ValueError("teacher.ports: invalid or repeated port")
        if self.run.name in {"", ".", ".."} or Path(self.run.name).name != self.run.name:
            raise ValueError("run.name must be a single path component")
        if self.student.weight_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("student.weight_dtype: unsupported dtype")
        if self.student.lora_target not in {"all", "no_self_attn"}:
            raise ValueError("student.lora_target: unsupported target")
        if self.student.tune_mode not in {"lora", "full"}:
            raise ValueError("student.tune_mode: expected 'lora' or 'full'")
        if self.student.tune_mode == "full" and self.student.lora_path:
            raise ValueError("student.lora_path is a LoRA checkpoint; unsupported with tune_mode full")
        if self.train.log_steps <= 0 or self.eval.inference.num_inference_steps <= 0:
            raise ValueError("Logging interval and inference steps must be positive")
        if self.eval.limit is not None and self.eval.limit <= 0:
            raise ValueError("eval.limit must be positive or null")
        if self.data.num_workers < 0 or min(self.data.height, self.data.width) <= 0:
            raise ValueError("Invalid data dimensions or worker count")
        if self.eval.inference.memory_mode not in {"full", "model_cpu_offload", "sequential_cpu_offload"}:
            raise ValueError("eval.inference.memory_mode: unsupported mode")
        if min(self.eval.inference.fps, self.eval.inference.num_frames, self.eval.inference.source_frames) <= 0:
            raise ValueError("Frame counts and fps must be positive")

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_dir) / self.run.name

    @property
    def inference_path(self) -> Path:
        """<inference_dir>/<name>/<bench>; a step count other than the backend's is added to <bench>."""
        from i2vt.models import default_inference_steps

        steps = self.eval.inference.num_inference_steps
        bench = self.eval.bench if steps == default_inference_steps(self) else f"{self.eval.bench}_{steps}steps"
        return Path(self.run.inference_dir) / self.run.name / bench

    @property
    def checkpoint_path(self) -> Path | None:
        if self.eval.skip_checkpoint:
            return None
        if self.eval.checkpoint_path:
            return Path(self.eval.checkpoint_path)
        # A full fine-tune saves the transformer under another name than a LoRA.
        kind = "full" if self.student.tune_mode == "full" else "lora"
        return self.output_path / "checkpoints" / f"{self.student.backend}_{kind}_step_{self.train.max_steps}.safetensors"
