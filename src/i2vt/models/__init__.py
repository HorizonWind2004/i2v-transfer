"""Backend registry. Importing configuration does not import model runtimes."""

import importlib

_STUDENT_BACKENDS = {
    "bernini": "BerniniStudentBackend",
    "videocof": "VideoCoFStudentBackend",
}


def _backend_module(backend, name):
    if backend not in _STUDENT_BACKENDS:
        raise ValueError(f"Unsupported student backend: {backend!r}")
    return importlib.import_module(f"i2vt.models.{backend}.{name}")


def validate_backend_config(config):
    """Fill the backend's defaults into ``config`` and check its backend-specific fields."""
    _backend_module(config.student.backend, "inference_config").validate(config)


def default_inference_steps(config):
    return _backend_module(config.student.backend, "inference_config").INFERENCE_STEPS


def create_student(config, device):
    backend = config.student.backend
    config_module = _backend_module(backend, f"configuration_{backend}")
    model_module = _backend_module(backend, f"modeling_{backend}")
    cls = getattr(model_module, _STUDENT_BACKENDS[backend])
    return cls(config_module.model_config(config), device)


def create_teacher(config, device):
    if config.teacher.backend == "offline":
        from .offline.configuration_offline import model_config as offline_model_config
        from .offline.modeling_offline import OfflineTeacherBackend

        return OfflineTeacherBackend(offline_model_config(config), device)

    from .joyedit.configuration_joyedit import model_config
    from .joyedit.modeling_joyedit import JoyEditTeacherBackend

    return JoyEditTeacherBackend(model_config(config), device)
