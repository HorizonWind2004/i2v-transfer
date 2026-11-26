from __future__ import annotations

import json
import os
import re
import time
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from i2vt.config import EDIT_TYPES, resolve_loss_steps
from i2vt.data.dataset import ImageEditDataset, build_image_edit_dataset
from i2vt.loss import distillation_loss
from i2vt.models import create_student
from i2vt.models.modeling_utils import StudentBackend
from i2vt.pipelines import training_preview
from i2vt.pipelines.pipeline_utils import training_size
from i2vt.utils import assert_compatible_vae


def broadcast_params(params, src: int, group):
    for p in params:
        dist.broadcast(p.data, src=src, group=group)


def average_gradients(params, group, world_size: int):
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        p.grad.div_(world_size)


def average_scalar(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    y = x.detach().float()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
    return y / world_size


def average_vector(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    y = x.detach().float()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
    return y / world_size


# Per-type loss is all-reduced across ranks, so every rank uses the same fixed table.
TASK_NAMES = EDIT_TYPES
TASK_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}


def sum_vector(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    """Per-type totals need sums, not means, because each type has a different sample count."""
    y = x.detach().float()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
    return y


TIMER_NAMES = (
    "step",
    "encode_prompt",
    "initial_latents",
    "student_velocity",
    "teacher_send",
    "teacher_wait",
    "scheduler_step",
    "backward",
    "all_reduce",
    "optimizer",
)


def ensure_wandb_auth(config):
    if config.train.logging.mode != "online":
        return
    key = os.environ.get("WANDB_API_KEY")
    key_file = config.train.logging.api_key_file
    if not key and key_file:
        path = Path(key_file).expanduser()
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            os.environ["WANDB_API_KEY"] = key
    if key and not re.fullmatch(r"[A-Za-z0-9_]+", key):
        raise ValueError("Invalid W&B API key format. Expected only letters, digits, and underscores.")
    if key:
        import wandb

        wandb.login(key=key, relogin=False, verify=True)


def maybe_init_wandb(config, out_dir: Path, run_config: dict):
    if config.train.logging.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without wandb logging", flush=True)
        return None

    ensure_wandb_auth(config)
    wandb_dir = Path(os.environ.get("WANDB_DIR", out_dir)).expanduser()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=config.train.logging.project,
        name=config.run.name,
        mode=config.train.logging.mode,
        dir=str(wandb_dir),
        config=run_config,
    )


def finish_wandb_run(wandb_run) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except OSError as exc:
        print(f"wandb finish warning: {exc}", flush=True)


def run_validation_previews(
    config,
    student: StudentBackend,
    dataset: ImageEditDataset,
    out_dir: Path,
    step: int,
    wandb_run,
    rank: int,
    student_offset: int,
    student_world: int,
    student_group,
    coordinator: bool,
):
    per_rank = max(0, int(config.train.validation.samples_per_rank))
    samples_per_type = max(0, int(config.train.validation.samples_per_type))
    validation_edit_types = [item.strip() for item in config.train.validation.edit_types if item.strip()]
    stratified = bool(validation_edit_types and samples_per_type > 0)
    if config.train.validation.steps <= 0 or (
        config.train.validation.samples <= 0 and per_rank <= 0 and not stratified
    ):
        return
    local_rank = rank - student_offset

    explicit_prompts: list[str] | None = None
    vp_raw = config.train.validation.prompts
    if vp_raw:
        explicit_prompts = [p.strip() for p in vp_raw if p.strip()]

    stratified_indices: list[int] | None = None
    if stratified:
        buckets = {edit_type: [] for edit_type in validation_edit_types}
        cursor = int(config.train.validation.start_index)
        for offset in range(len(dataset)):
            idx = (cursor + offset) % len(dataset)
            edit_type = dataset[idx].get("edit_type")
            if edit_type in buckets and len(buckets[edit_type]) < samples_per_type:
                buckets[edit_type].append(idx)
            if all(len(items) >= samples_per_type for items in buckets.values()):
                break
        missing = {
            edit_type: samples_per_type - len(items)
            for edit_type, items in buckets.items()
            if len(items) < samples_per_type
        }
        if missing:
            raise RuntimeError(f"validation missing samples per edit_type: {missing}")
        stratified_indices = [idx for edit_type in validation_edit_types for idx in buckets[edit_type]]

    sample_count = (
        len(stratified_indices)
        if stratified_indices is not None
        else min(per_rank if per_rank > 0 else config.train.validation.samples, len(dataset))
    )
    if explicit_prompts is not None:
        sample_count = min(sample_count, len(explicit_prompts))
    if sample_count <= 0:
        return
    validation_dir = out_dir / "validation" / f"step_{step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(sample_count):
        if per_rank > 0:
            data_idx = (config.train.validation.start_index + local_rank * sample_count + sample_idx) % len(dataset)
            stem = f"rank_{rank:02d}_sample_{sample_idx:02d}_idx_{data_idx}"
        else:
            if not coordinator:
                continue
            data_idx = (
                stratified_indices[sample_idx]
                if stratified_indices is not None
                else (config.train.validation.start_index + sample_idx) % len(dataset)
            )
            stem = f"sample_{sample_idx:02d}_idx_{data_idx}"
        data = dataset[data_idx]
        prompt = (
            explicit_prompts[sample_idx % len(explicit_prompts)] if explicit_prompts else data["video_edit_prompt"]
        )
        source, prediction, compare = training_preview(
            student,
            data["source"],
            prompt,
            seed=config.train.validation.seed + step * 1000 + local_rank * sample_count + sample_idx,
            rollout_steps=config.train.validation.num_inference_steps or config.train.rollout.num_inference_steps,
        )
        source_path = validation_dir / f"{stem}_source.jpg"
        pred_path = validation_dir / f"{stem}_pred.jpg"
        compare_path = validation_dir / f"{stem}_compare.jpg"
        prompt_path = validation_dir / f"{stem}_prompt.txt"
        source.save(source_path, quality=95)
        prediction.save(pred_path, quality=95)
        compare.save(compare_path, quality=95)
        prompt_path.write_text(prompt, encoding="utf-8")
    if per_rank > 0:
        dist.barrier(group=student_group)
    if coordinator and wandb_run is not None:
        import wandb

        wandb_images = []
        for compare_path in sorted(validation_dir.glob("*_compare.jpg")):
            prompt_path = compare_path.with_name(compare_path.name.replace("_compare.jpg", "_prompt.txt"))
            caption = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else compare_path.stem
            wandb_images.append(wandb.Image(str(compare_path), caption=f"step={step}: {caption}"))
        if wandb_images:
            wandb.log({"validation/rollout_compare": wandb_images}, step=step)
    if coordinator:
        total = sample_count * student_world if per_rank > 0 else sample_count
        print(f"validation step={step} saved {total} previews to {validation_dir}", flush=True)


def _train(config, rank, world, device, student_group, student, teacher):
    student_offset = 0
    student_world = world - student_offset
    out_dir = Path(config.run.output_dir) / config.run.name
    coordinator = rank == student_offset
    global_batch_size = student_world
    grad_accum_steps = max(1, int(config.train.gradient_accumulation_steps))
    effective_global_batch_size = global_batch_size * grad_accum_steps
    if coordinator:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

    student = student if student is not None else create_student(config, device)
    if config.train.check_vae_compat:
        teacher_sig = teacher.vae_signature()
        assert_compatible_vae(student.vae_signature(), teacher_sig)

    params = student.params()
    broadcast_params(params, src=student_offset, group=student_group)
    optimizer_cls = OffloadedAdamW if config.train.offload_optimizer else torch.optim.AdamW
    optimizer = optimizer_cls(params, lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    dataset = build_image_edit_dataset(config.data.base_path, config.data.train_dataset)
    sampler = DistributedSampler(dataset, num_replicas=student_world, rank=rank - student_offset, shuffle=True)
    loader_kwargs = dict(
        batch_size=1, sampler=sampler, collate_fn=lambda xs: xs[0], num_workers=config.data.num_workers
    )
    if config.train.deterministic:
        # Each worker process reseeds itself, so without these the shuffle order
        # and any augmentation draw differ between runs.
        generator = torch.Generator()
        generator.manual_seed(config.train.seed + rank)
        loader_kwargs["generator"] = generator
        loader_kwargs["worker_init_fn"] = lambda worker_id: torch.manual_seed(
            config.train.seed + rank * 1000 + worker_id
        )
    if config.data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(config.data.prefetch_factor))
        loader_kwargs["pin_memory"] = bool(config.data.pin_memory)
        loader_kwargs["persistent_workers"] = bool(config.data.persistent_workers)
    loader = DataLoader(dataset, **loader_kwargs)
    wandb_run = None
    if coordinator:
        run_config = {
            **config.to_dict(),
            "dataset_size": len(dataset),
            "local_batch_size": 1,
            "student_world_size": student_world,
            "global_batch_size": global_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "effective_global_batch_size": effective_global_batch_size,
        }
        wandb_run = maybe_init_wandb(config, out_dir, run_config)
        print(
            "training_config "
            f"dataset={config.data.train_dataset} dataset_size={len(dataset)} "
            f"local_batch_size=1 global_batch_size={global_batch_size} "
            f"gradient_accumulation_steps={grad_accum_steps} "
            f"effective_global_batch_size={effective_global_batch_size} "
            f"ranks={student_world} student_backend={config.student.backend} "
            f"rollout_steps={config.train.rollout.num_inference_steps} rollout_cfg={config.train.rollout.guidance_scale} "
            f"rollout_timestep_mode={config.train.rollout.timestep_mode} "
            f"student_cfg={config.train.rollout.student_guidance_scale} teacher_cfg={config.teacher.guidance_scale} "
            f"reasoning_latents={config.student.model_kwargs.get('reasoning_latent_count', 0)}",
            flush=True,
        )

    step = 0
    epoch = 0
    sampler.set_epoch(epoch)
    loader_iter = iter(loader)
    while step < config.train.max_steps:
        step_start = time.perf_counter()
        timers = {name: 0.0 for name in TIMER_NAMES}
        optimizer.zero_grad(set_to_none=True)
        accum_losses = []
        accum_timestep_losses = []
        task_sum = torch.zeros(len(TASK_NAMES), device=device)
        task_cnt = torch.zeros(len(TASK_NAMES), device=device)

        for _accum_idx in range(grad_accum_steps):
            try:
                data = next(loader_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                loader_iter = iter(loader)
                data = next(loader_iter)
            losses = []
            selected_timesteps = []
            chunk_size = config.train.rollout.chunk_size or config.train.rollout.num_inference_steps
            # Offline training changes only where the steps come from; the teacher call, loss,
            # accumulation and logging below are shared. Every offline draw is supervised.
            if config.train.objective == "offline":
                train_idx = set(range(max(1, int(config.train.offline.draws_per_sample))))
                trajectory = iter(student.offline_steps(data, config.train.rollout, config.train.offline))
            else:
                # The loss is normalised by the steps that backpropagate, so a trajectory truncated by
                # train_steps keeps the effective learning rate.
                rollout_steps = (
                    config.train.rollout.train_steps or config.train.rollout.num_inference_steps
                )
                train_idx = set(resolve_loss_steps(config.train.rollout.loss_steps, rollout_steps))
                trajectory = iter(student.rollout(data, config.train.rollout))
            # Per-sample size: the teacher must see the pixels the student sees.
            train_w, train_h = training_size(
                data["source"].size, height=config.data.height, width=config.data.width,
                mode=config.data.resize_mode,
            )
            conditioning = {**data, "height": train_h, "width": train_w}
            step_idx = 0
            while True:
                pending_chunk = []
                # Steps outside loss_steps never enter pending_chunk, so only an empty slice of the
                # trajectory ends it (with last:5 the first chunks hold no supervised step).
                consumed = 0
                for sample in islice(trajectory, chunk_size):
                    consumed += 1
                    for name, duration in sample.timings.items():
                        timers[name] += duration
                    # A step without loss: the rollout already advanced the trajectory, so no teacher call is needed.
                    if step_idx not in train_idx:
                        step_idx += 1
                        continue
                    # Keep the tensor: .item() here would synchronise with the GPU once per
                    # rollout step. The values are read at log time.
                    step_idx += 1
                    selected_timesteps.append(sample.timestep.detach())
                    t0 = time.perf_counter()
                    future = teacher.submit(sample, conditioning)
                    timers["teacher_send"] += time.perf_counter() - t0
                    pending_chunk.append((future, sample))
                if not consumed:
                    break
                if not pending_chunk:
                    continue
                for future, sample in pending_chunk:
                    t0 = time.perf_counter()
                    target = future.result().to(device=device)
                    timers["teacher_wait"] += time.perf_counter() - t0
                    loss_i = distillation_loss(sample, target, config.train.loss_type, teacher.prediction_type)
                    losses.append(loss_i.detach())
                    t0 = time.perf_counter()
                    (loss_i / (len(train_idx) * grad_accum_steps)).backward()
                    timers["backward"] += time.perf_counter() - t0
                pending_chunk.clear()
            if not losses:
                raise RuntimeError(f"Student {config.train.objective} produced no distillation steps")

            accum_losses.append(torch.stack(losses).mean())
            accum_timestep_losses.append(torch.stack([x.detach().float() for x in losses]))
            # The loader yields one sample, so edit_type is a bare string; wrap it before bucketing.
            edit_types = data.get("edit_type") or ()
            if isinstance(edit_types, (str, bytes)):
                edit_types = (edit_types,)
            for edit_type in edit_types:
                idx = TASK_INDEX.get(str(edit_type), len(TASK_NAMES) - 1)
                task_sum[idx] += accum_losses[-1].detach()
                task_cnt[idx] += 1

        loss = torch.stack(accum_losses).mean()
        local_timestep_losses = torch.stack(accum_timestep_losses).mean(dim=0)
        t0 = time.perf_counter()
        average_gradients(params, student_group, student_world)
        timers["all_reduce"] += time.perf_counter() - t0
        grad_norm = torch.nn.utils.clip_grad_norm_(params, config.train.max_grad_norm)
        t0 = time.perf_counter()
        optimizer.step()
        timers["optimizer"] += time.perf_counter() - t0
        timers["step"] = time.perf_counter() - step_start
        step += 1
        should_log = step % config.train.log_steps == 0
        if should_log:
            mean_loss = average_scalar(loss, student_group, student_world)
            timestep_losses = average_vector(local_timestep_losses, student_group, student_world)
            timer_tensor = torch.tensor([timers[name] for name in TIMER_NAMES], device=device, dtype=torch.float32)
            mean_timers = average_vector(timer_tensor, student_group, student_world)
            task_sums = sum_vector(task_sum, student_group, student_world)
            task_counts = sum_vector(task_cnt, student_group, student_world)
            if coordinator:
                loss_items = {f"train/distill_loss_t{i}": float(v.item()) for i, v in enumerate(timestep_losses)}
                task_items = {
                    f"train/loss_by_task/{name}": float((task_sums[i] / task_counts[i]).item())
                    for i, name in enumerate(TASK_NAMES)
                    if float(task_counts[i].item()) > 0
                }
                timing_items = {f"time/{name}_sec": float(v.item()) for name, v in zip(TIMER_NAMES, mean_timers)}
                log_payload = {
                    "train/distill_loss": float(mean_loss.item()),
                    "train/grad_norm": float(grad_norm.detach().float().item()),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/global_batch_size": global_batch_size,
                    "train/gradient_accumulation_steps": grad_accum_steps,
                    "train/effective_global_batch_size": effective_global_batch_size,
                    "train/rollout_steps": config.train.rollout.num_inference_steps,
                    "train/rollout_train_steps": config.train.rollout.train_steps
                    or config.train.rollout.num_inference_steps,
                    "train/rollout_timestep_mode": config.train.rollout.timestep_mode,
                    "train/reasoning_latent_count": config.student.model_kwargs.get("reasoning_latent_count", 0),
                    "train/samples_per_sec": effective_global_batch_size / max(float(mean_timers[0].item()), 1e-6),
                    **loss_items,
                    **task_items,
                    **timing_items,
                }
                timing_summary = " ".join(
                    f"{name}={float(value.item()):.3f}s" for name, value in zip(TIMER_NAMES, mean_timers)
                )
                print(
                    f"step={step} distill_loss={mean_loss.item():.6f} "
                    f"grad_norm={float(grad_norm.detach().float().item()):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"global_batch_size={global_batch_size} "
                    f"grad_accum={grad_accum_steps} "
                    f"effective_global_batch_size={effective_global_batch_size} "
                    f"samples_per_sec={effective_global_batch_size / max(float(mean_timers[0].item()), 1e-6):.4f} "
                    f"rollout_steps={config.train.rollout.num_inference_steps} "
                    f"rollout_train_steps={config.train.rollout.train_steps or config.train.rollout.num_inference_steps} "
                    f"rollout_timestep_mode={config.train.rollout.timestep_mode} "
                    f"timesteps={[round(x, 3) for x in torch.stack(selected_timesteps).tolist()]} "
                    f"timestep_losses={[round(float(v.item()), 6) for v in timestep_losses]} "
                    f"task_losses=[{' '.join(f'{k.rsplit(chr(47), 1)[-1]}={v:.5f}' for k, v in task_items.items())}] "
                    f"timing=[{timing_summary}]",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log(log_payload, step=step)
        if coordinator and step % config.train.save_steps == 0:
            student.save(out_dir / "checkpoints", step)
        should_validate = config.train.validation.steps > 0 and step % config.train.validation.steps == 0
        if config.train.validation.first_step and config.train.validation.steps > 0 and step == 1:
            should_validate = True
        if should_validate:
            run_validation_previews(
                config,
                student,
                dataset,
                out_dir,
                step,
                wandb_run,
                rank,
                student_offset,
                student_world,
                student_group,
                coordinator,
            )

    dist.barrier(group=student_group)
    if coordinator:
        student.save(out_dir / "checkpoints", step)
        finish_wandb_run(wandb_run)


class OffloadedAdamW:
    """AdamW whose moments and fp32 master weights live in CPU memory.

    The GPU keeps the trainable weights and their gradients. At each step the gradients go to the
    CPU, the same fp32 AdamW update runs there, and the new weights are copied back, so the result
    equals torch.optim.AdamW on the GPU and the GPU never holds the two moments.
    """

    def __init__(self, params, lr, weight_decay):
        self.params = list(params)
        self.master = [torch.nn.Parameter(p.detach().to("cpu", dtype=torch.float32, copy=True)) for p in self.params]
        self.optimizer = torch.optim.AdamW(self.master, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.params, self.master):
            m.grad = None if p.grad is None else p.grad.detach().to("cpu", dtype=torch.float32)
        self.optimizer.step()
        for p, m in zip(self.params, self.master):
            p.copy_(m.to(device=p.device, dtype=p.dtype))
            m.grad = None


class Trainer:
    """Train the student of one resolved experiment against the given teacher."""

    def __init__(self, config, student=None, *, teacher):
        self.config = config
        self.student = student
        self.teacher = teacher

    def train(self, rank, world_size, device, process_group):
        return _train(self.config, rank, world_size, device, process_group, self.student, self.teacher)
