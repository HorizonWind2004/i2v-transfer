"""Edit one video with a released I2VT checkpoint (Bernini-R 1.3B, Bernini-R 14B or VideoCoF students).

    python scripts/infer_video.py \
        --config configs/train/i2vt-1.3b.yaml \
        --checkpoint /path/to/bernini_lora_step_200.safetensors \
        --video source.mp4 --prompt "Make the person raise their right arm." --output-dir outputs/demo

The config supplies the base weights and the inference settings (steps, guidance, frames), so the
result matches the benchmark pipeline. ``--base`` runs the pretrained editor without the I2VT
LoRA, and ``--steps`` overrides the number of inference steps. For Bernini-R 14B, pass the
high-noise LoRA ``bernini_lora_step_<N>.safetensors``; the low-noise one
(``bernini_lora_low_step_<N>.safetensors``) is read from the same directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from i2vt.config import ExperimentConfig


def edit_video(config, video, prompt, output_dir, *, checkpoint=None, base=False, steps=None, device="cuda:0"):
    """Run the config's inference pipeline on one video and save it under ``output_dir``."""
    from i2vt.evaluation.infer import save_pipeline_output
    from i2vt.pipelines import create_pipeline

    config.run.mode = "eval"
    if steps is not None:
        config.eval.inference.num_inference_steps = steps
    if base:
        config.eval.skip_checkpoint = True
    elif checkpoint is not None:
        config.eval.checkpoint_path = str(checkpoint)
    if config.checkpoint_path is not None and not config.checkpoint_path.is_file():
        raise FileNotFoundError(f"I2VT checkpoint not found: {config.checkpoint_path} (pass --checkpoint or --base)")
    pipeline = create_pipeline(config, device=device)
    output = pipeline(video=str(video), prompt=prompt)
    stem = Path(video).stem
    save_pipeline_output(output, output_dir, stem)
    return Path(output_dir) / f"gen_{stem}.mp4"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Edit one video with an I2VT checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="the model's config, e.g. configs/train/i2vt-1.3b.yaml")
    parser.add_argument("--video", type=Path, required=True, help="source video")
    parser.add_argument("--prompt", required=True, help="editing instruction")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--checkpoint", type=Path, help="I2VT LoRA (.safetensors); default: the config's checkpoint")
    parser.add_argument("--base", action="store_true", help="run the pretrained editor without the I2VT LoRA")
    parser.add_argument("--steps", type=int, help="inference steps (default: the base editor's)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.base and args.checkpoint:
        parser.error("--base and --checkpoint are mutually exclusive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    config = ExperimentConfig.from_yaml(args.config)
    path = edit_video(config, args.video, args.prompt, args.output_dir,
                      checkpoint=args.checkpoint, base=args.base, steps=args.steps, device=args.device)
    print(f"Saved {path} (side-by-side comparison: {path.with_name(path.stem + '_compare.mp4')})")


if __name__ == "__main__":
    main()
