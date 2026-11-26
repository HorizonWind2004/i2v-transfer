"""I2VT: improving and expanding pretrained video editors with an image editor as the teacher."""

from importlib import import_module

from .config import ExperimentConfig, TrainingConfig

__version__ = "0.1.0"
_EXPORTS = {
    "StudentBackend": ("models.modeling_utils", "StudentBackend"),
    "TeacherBackend": ("models.modeling_utils", "TeacherBackend"),
    "DistillationStep": ("models.modeling_utils", "DistillationStep"),
    "BerniniConfig": ("models.bernini.configuration_bernini", "BerniniConfig"),
    "BerniniStudentBackend": ("models.bernini.modeling_bernini", "BerniniStudentBackend"),
    "VideoCoFConfig": ("models.videocof.configuration_videocof", "VideoCoFConfig"),
    "VideoCoFStudentBackend": ("models.videocof.modeling_videocof", "VideoCoFStudentBackend"),
    "JoyEditConfig": ("models.joyedit.configuration_joyedit", "JoyEditConfig"),
    "JoyEditTeacherBackend": ("models.joyedit.modeling_joyedit", "JoyEditTeacherBackend"),
    "BerniniVideoEditPipeline": ("pipelines.bernini.pipeline_bernini", "BerniniVideoEditPipeline"),
    "VideoCoFVideoEditPipeline": ("pipelines.videocof.pipeline_videocof", "VideoCoFVideoEditPipeline"),
    "Trainer": ("trainer", "Trainer"),
}
__all__ = ["ExperimentConfig", "TrainingConfig", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value
