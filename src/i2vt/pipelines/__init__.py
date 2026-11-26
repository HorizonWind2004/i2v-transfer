"""Inference and training-preview factories with lazy backend imports."""

import importlib

_PIPELINES = {
    "bernini": "BerniniVideoEditPipeline",
    "videocof": "VideoCoFVideoEditPipeline",
}


def training_preview(model, image, prompt, *, seed, rollout_steps=None):
    """Decode the student's own rollout without retaining training graphs."""
    import torch
    from PIL import Image

    from i2vt.config import RolloutConfig
    from i2vt.pipelines.pipeline_utils import frame_to_image

    config = model.config
    rollout = RolloutConfig(
        num_inference_steps=rollout_steps or config.rollout_steps,
        chunk_size=1,
        train_steps=config.rollout_train_steps,
        timestep_mode=config.rollout_timestep_mode,
        timestep_min=config.rollout_timestep_min,
        timestep_max=config.rollout_timestep_max,
        guidance_scale=config.rollout_guidance_scale,
        student_guidance_scale=config.rollout_guidance_scale,
        kwargs={},
    )
    trainable = model.trainable_module
    was_training = trainable.training
    trainable.eval()
    try:
        from i2vt.pipelines.pipeline_utils import training_size

        width, height = training_size(image.size, height=config.height, width=config.width,
                                      mode=getattr(config, "resize_mode", "area"))
        resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        devices = (
            [model.device.index if model.device.index is not None else torch.cuda.current_device()]
            if model.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            trajectory = model.rollout({"source": resized, "video_edit_prompt": prompt}, rollout)
            while True:
                try:
                    next(trajectory)
                except StopIteration as done:
                    latents = done.value
                    break
            if latents is None:
                raise RuntimeError("This student does not provide a final latent for preview")
            decoded = model.vae.decode(latents.to(dtype=model.weight_dtype), return_dict=False)[0]
            prediction = frame_to_image(decoded[0, :, 0])
        compare = Image.new("RGB", (width * 2, height))
        compare.paste(resized, (0, 0))
        compare.paste(prediction.resize((width, height)), (width, 0))
        return resized, prediction, compare
    finally:
        trainable.train(was_training)


def create_pipeline(config, device="cuda:0"):
    name = config.student.backend
    if name not in _PIPELINES:
        raise ValueError(f"Unsupported student backend: {name!r}")
    module = importlib.import_module(f"i2vt.pipelines.{name}.pipeline_{name}")
    cls = getattr(module, _PIPELINES[name])
    return cls.from_config(config, device=device)
