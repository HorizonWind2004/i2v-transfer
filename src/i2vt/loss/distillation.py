import torch
import torch.nn.functional as F


def x0_channel_cosine_loss(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
) -> torch.Tensor:
    """Average cosine distance over channel vectors at each latent position."""

    return (
        1.0
        - F.cosine_similarity(
            student_x0.float(),
            teacher_x0.float(),
            dim=1,
            eps=1e-8,
        ).mean()
    )


def velocity_channel_cosine_loss(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
) -> torch.Tensor:
    """Average cosine distance over detached teacher velocity channel vectors."""

    return (
        1.0
        - F.cosine_similarity(
            student_velocity.float(),
            teacher_velocity.detach().float(),
            dim=1,
            eps=1e-8,
        ).mean()
    )


def distillation_loss(step, teacher_prediction, loss_type="v", teacher_prediction_type="velocity"):
    """Compare normalized target predictions; never propagate teacher gradients.

    ``loss_type`` is "v" (v-MSE, the default), "v_cos", "x" (x0-MSE) or "x_cos".
    """
    student = step.prediction.float()
    teacher = teacher_prediction.detach().float()
    if student.shape != teacher.shape or student.shape != step.noisy_latents.shape:
        raise RuntimeError(f"prediction mismatch: student={tuple(student.shape)} teacher={tuple(teacher.shape)}")
    sample = step.noisy_latents.detach().float()
    sigma = step.sigma.to(device=sample.device)

    def convert(value, prediction_type, target_type):
        if prediction_type not in {"velocity", "sample"}:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        if prediction_type == target_type:
            return value
        if target_type == "sample":
            return sample - sigma * value
        if torch.any(sigma == 0):
            raise ValueError("Cannot convert x0 to velocity at sigma=0")
        return (sample - value) / sigma

    if loss_type not in {"v", "v_cos", "x", "x_cos"}:
        raise ValueError(f"Unsupported distillation loss: {loss_type}")
    target_type = "sample" if loss_type in {"x", "x_cos"} else "velocity"
    student = convert(student, step.prediction_type, target_type)
    teacher = convert(teacher, teacher_prediction_type, target_type)
    if loss_type == "x_cos":
        return x0_channel_cosine_loss(student, teacher)
    if loss_type == "v_cos":
        return velocity_channel_cosine_loss(student, teacher)
    return F.mse_loss(student, teacher)
