"""Launch a training or evaluation run from one YAML configuration."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from i2vt.config import MODE_ALIASES, MODES, ExperimentConfig


class ProcessGroup:
    """Own subprocesses and their process groups, including torchrun descendants."""

    def __init__(self):
        self.processes = []
        self.logs = []

    def start(self, command, log_path, *, gpu_ids=None):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        self.logs.append(log)
        env = os.environ.copy()
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    def wait(self, processes, *, monitors=()):
        pending = set(processes)
        while pending:
            for process in monitors:
                if process.poll() is not None:
                    raise RuntimeError(f"Required service exited with status {process.returncode}")
            for process in list(pending):
                status = process.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f"Worker exited with status {status}; see run logs")
                    pending.remove(process)
            if pending:
                time.sleep(0.1)

    def close(self):
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for log in self.logs:
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def wait_for_teachers(config, processes, *, timeout=1800):
    deadline = time.monotonic() + timeout
    pending = set(config.teacher.ports)
    while pending:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("A teacher server exited before it was ready; see teacher_logs/")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Teacher servers not ready on ports {sorted(pending)}")
        for port in list(pending):
            try:
                with socket.create_connection((config.teacher.host, port), timeout=1):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.5)


def _train_command(config, resolved):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={len(config.runtime.student_gpus)}",
        "-m",
        "i2vt.run",
        str(resolved),
        "--worker",
        "train",
    ]


def run_training(config, resolved):
    with ProcessGroup() as group:
        if config.runtime.teacher_gpus is None:
            # Each rank builds its own teacher on its own GPU.
            student = group.start(_train_command(config, resolved), config.output_path / "run.log",
                                  gpu_ids=config.runtime.student_gpus)
            group.wait([student])
            return
        # RPC: one teacher server per teacher GPU, students call them over TCP.
        teachers = [
            group.start(
                [sys.executable, "-m", "i2vt.run", str(resolved), "--worker", "teacher", "--index", str(index)],
                config.output_path / "teacher_logs" / f"teacher_{index}.log",
                gpu_ids=[gpu],
            )
            for index, gpu in enumerate(config.runtime.teacher_gpus)
        ]
        wait_for_teachers(config, teachers)
        student = group.start(_train_command(config, resolved), config.output_path / "run.log",
                              gpu_ids=config.runtime.student_gpus)
        group.wait([student], monitors=teachers)


def run_evaluation(config, resolved):
    with ProcessGroup() as group:
        workers = [
            group.start(
                [sys.executable, "-m", "i2vt.run", str(resolved), "--worker", "infer", "--index", str(rank)],
                config.inference_path / "logs" / f"worker_{rank}.log",
                gpu_ids=[gpu],
            )
            for rank, gpu in enumerate(config.runtime.eval_gpus)
        ]
        group.wait(workers)
    if config.eval.scoring.enabled:
        with ProcessGroup() as group:
            score = group.start(
                [sys.executable, "-m", "i2vt.run", str(resolved), "--worker", "score"],
                config.inference_path / "logs" / "scoring.log",
            )
            group.wait([score])


def preflight(config):
    if config.run.mode != "eval" and not Path(config.data.train_dataset).exists():
        raise FileNotFoundError(config.data.train_dataset)
    if config.run.mode != "train":
        dataset_file = config.eval.bench_csv
        if not dataset_file or not Path(dataset_file).is_file():
            raise FileNotFoundError(f"Evaluation input missing: {dataset_file}")
        if config.run.mode == "eval" and config.checkpoint_path is not None:
            if not config.checkpoint_path.is_file():
                raise FileNotFoundError(config.checkpoint_path)
        if config.eval.scoring.enabled:
            key_envs = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            if not config.eval.scoring.api_key_file and not any(os.environ.get(key) for key in key_envs):
                raise ValueError(f"Set eval.scoring.api_key_file or one of {key_envs}")


def run(config, *, dry_run=False):
    if dry_run:
        print(config.to_yaml(), end="")
        print(f"# checkpoint: {config.checkpoint_path or 'none (base weights)'}")
        print(f"# inference outputs: {config.inference_path}")
        if config.runtime.teacher_gpus is None:
            print(f"# training processes: {len(config.runtime.student_gpus)}, each with a student and a teacher")
        else:
            print(f"# training processes: {len(config.runtime.student_gpus)} students, "
                  f"{len(config.runtime.teacher_gpus)} teacher servers (RPC)")
        print(f"# evaluation processes: {len(config.runtime.eval_gpus)}")
        return
    preflight(config)
    config.output_path.mkdir(parents=True, exist_ok=True)
    resolved = config.output_path / "resolved_config.yaml"
    if resolved.exists() and resolved.read_text() != config.to_yaml():
        previous = ExperimentConfig.from_yaml(resolved).to_dict()
        current = config.to_dict()
        previous["run"]["mode"] = current["run"]["mode"]
        # Evaluating the same checkpoint on another benchmark changes only the eval section, which
        # training never reads. A train_eval run with another eval section would retrain and
        # overwrite the checkpoint, so only eval runs may differ there.
        eval_only = config.run.mode == "eval" and {**previous, "eval": current["eval"]} == current
        if previous != current and not eval_only:
            raise ValueError("Run directory already contains a different resolved configuration; use a new run name")
        suffix = ""
        if previous["eval"] != current["eval"]:
            import hashlib
            import json

            suffix = "." + hashlib.sha1(json.dumps(current["eval"], sort_keys=True).encode()).hexdigest()[:8]
        resolved = config.output_path / f"resolved_config.{config.run.mode}{suffix}.yaml"
    config.to_yaml(resolved)
    if config.run.mode != "eval":
        run_training(config, resolved)
    if config.run.mode != "train":
        if config.checkpoint_path is not None and not config.checkpoint_path.is_file():
            raise FileNotFoundError(f"Final checkpoint missing: {config.checkpoint_path}")
        run_evaluation(config, resolved)


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed every generator the training loop draws from.

    torch.manual_seed alone leaves the stdlib and numpy generators free, and leaves cuDNN free to
    pick kernels by benchmark and cuBLAS free to reduce in whatever order is fastest, so two runs
    diverge within a few steps.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not deterministic:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cuBLAS needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) in the environment
    # before the first handle is created; without it this call raises.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # warn_only=False is what pins SDPA: its flash backward accumulates dq with atomics and only
    # takes the deterministic path when this is strict. With warn_only=True it warns, keeps the
    # fast kernel, and two runs diverge from the second optimizer step.
    torch.use_deterministic_algorithms(True, warn_only=False)


def train_worker(config):
    import torch
    import torch.distributed as dist

    from i2vt.models import create_teacher
    from i2vt.trainer import Trainer

    timeout_sec = int(os.environ.get("I2VT_DIST_TIMEOUT", "7200"))
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_sec))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        seed_everything(config.train.seed + rank, config.train.deterministic)
        # new_group does not inherit the timeout from init_process_group, and every training
        # collective runs on this group, so pass the timeout again.
        group = dist.new_group(ranks=list(range(world)), timeout=timedelta(seconds=timeout_sec))
        if config.runtime.teacher_gpus is None:
            teacher = create_teacher(config, device)
        else:
            from i2vt.models.teacher_rpc import RPCTeacherBackend

            teacher = RPCTeacherBackend(
                [f"{config.teacher.host}:{port}" for port in config.teacher.ports],
                rank,
                timeout=config.teacher.rpc_timeout,
                max_workers=config.teacher.client_workers,
            )
        # JoyEdit's build_model() seeds every rank with train.seed. Reseed so ranks draw different
        # rollout noise, and so the student starts from the same RNG state as with the offline
        # teacher, which never touches the generators.
        seed_everything(config.train.seed + rank, config.train.deterministic)
        try:
            Trainer(config, teacher=teacher).train(rank, world, device, group)
        finally:
            teacher.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def teacher_worker(config, index):
    import torch

    from i2vt.models import create_teacher
    from i2vt.models.teacher_rpc import TeacherServer

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    teacher = create_teacher(config, device)
    with TeacherServer((config.teacher.host, config.teacher.ports[index]), teacher) as server:
        server.serve_forever()


def apply_inference_steps(config, steps: int | None) -> None:
    """--steps N: override eval.inference.num_inference_steps.

    The inference directory names a step count other than the backend's, so its finished samples
    are never mixed with those of the default.
    """
    if steps is None:
        return
    if steps <= 0:
        raise SystemExit("--steps: must be positive")
    config.eval.inference.num_inference_steps = steps


def _mode(value):
    value = MODE_ALIASES.get(value, value)
    if value not in MODES:
        raise argparse.ArgumentTypeError(f"expected one of {', '.join(MODES)}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="i2vt", description="Train or evaluate an I2VT model from a YAML config")
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", type=_mode, metavar="{train,eval,train_eval}",
                        help="train (default), eval (evaluate a checkpoint) or train_eval")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved configuration and exit")
    parser.add_argument("--dataset", help="named training set: i2vt_17k (default), or pools such as action+relight")
    parser.add_argument("--bench", help="benchmark: openve (default) or openve_pp")
    parser.add_argument("--checkpoint", type=Path, help="LoRA to evaluate (default: the run's final checkpoint)")
    parser.add_argument("--steps", type=int, help="inference steps for evaluation (default: the backend's)")
    parser.add_argument("--judge", help="Gemini model that scores the edits (default: the benchmark's judge)")
    parser.add_argument("--worker", choices=["train", "teacher", "infer", "score"], help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    launcher_options = (args.mode, args.dry_run, args.dataset, args.bench, args.checkpoint, args.steps, args.judge)
    if args.worker and any(option for option in launcher_options):
        parser.error("Workers read the resolved configuration; launcher options do not apply")
    if args.index is not None and args.worker not in {"teacher", "infer"}:
        parser.error("--index applies only to teacher and inference workers")
    config = ExperimentConfig.from_yaml(args.config)
    if args.worker in {"teacher", "infer"}:
        count = len(config.runtime.teacher_gpus or []) if args.worker == "teacher" else len(config.runtime.eval_gpus)
        if args.index is None or not 0 <= args.index < count:
            parser.error(f"{args.worker} worker requires --index between 0 and {count - 1}")
    if args.worker == "train":
        train_worker(config)
    elif args.worker == "teacher":
        teacher_worker(config, args.index)
    elif args.worker == "infer":
        from i2vt.evaluation.infer import infer

        infer(config, args.index)
    elif args.worker == "score":
        from i2vt.evaluation.score import score_config

        score_config(config)
    else:
        if args.mode:
            config.run.mode = args.mode
        from i2vt.data.registry import apply_cli_choices, resolve_hf_sources

        apply_cli_choices(config, args.dataset, args.bench)
        if args.checkpoint:
            config.eval.checkpoint_path = str(args.checkpoint)
        if args.judge:
            config.eval.scoring.model = args.judge
        apply_inference_steps(config, args.steps)
        config.validate()
        # Download named sets before the resolved configuration is written, so every worker reads
        # local paths that pin the Hub revision.
        if not args.dry_run:
            for line in resolve_hf_sources(config):
                print(f"[data] {line}", flush=True)
        run(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
