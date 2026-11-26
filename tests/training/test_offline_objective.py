"""Offline training differs from online training only in the target.

The offline teacher's velocity is the exact one that carries the student's noisy latent to the
stored edit, the noise levels follow the sampler's shifted schedule, and the objective and the
teacher backend are chosen together.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from i2vt.loss.distillation import distillation_loss
from i2vt.models.modeling_utils import DistillationStep
from i2vt.models.offline.modeling_offline import OfflineTeacherBackend


def _teacher(target_latents):
    teacher = OfflineTeacherBackend.__new__(OfflineTeacherBackend)
    teacher.device = torch.device("cpu")
    teacher.weight_dtype = torch.float32
    teacher._target_latents = lambda image, height, width: target_latents
    return teacher


def _step(x0, eps, sigma):
    noisy = (1.0 - sigma) * x0 + sigma * eps
    return DistillationStep(noisy, torch.zeros_like(noisy), sigma * 1000.0, sigma)


class OfflineTargetTeacherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x0 = torch.randn(1, 16, 1, 8, 8)
        self.eps = torch.randn_like(self.x0)
        self.conditioning = {"target": Image.new("RGB", (8, 8)), "height": 512, "width": 512}

    def test_velocity_is_the_exact_path_to_the_target(self):
        # x_t = (1-s)x0 + s*eps and v = eps - x0, so (x_t - x0)/s must recover v
        # at every noise level without the teacher ever seeing the noise.
        for sigma in (0.0244, 0.395, 0.8631, 1.0):
            with self.subTest(sigma=sigma):
                sigma_t = torch.tensor(sigma)
                step = _step(self.x0, self.eps, sigma_t)
                got = _teacher(self.x0).predict(step, self.conditioning)
                torch.testing.assert_close(got, self.eps - self.x0, atol=1e-5, rtol=1e-4)

    def test_a_student_that_already_matches_the_target_has_no_loss(self):
        sigma = torch.tensor(0.5)
        noisy = (1.0 - sigma) * self.x0 + sigma * self.eps
        step = DistillationStep(noisy, self.eps - self.x0, sigma * 1000.0, sigma)
        target = _teacher(self.x0).predict(step, self.conditioning)
        for loss_type in ("v", "v_cos", "x", "x_cos"):
            with self.subTest(loss_type=loss_type):
                loss = distillation_loss(step, target, loss_type, "velocity")
                self.assertLess(float(loss), 1e-5)

    def test_sigma_zero_is_refused_rather_than_dividing_by_it(self):
        step = _step(self.x0, self.eps, torch.tensor(0.0))
        with self.assertRaises(ValueError):
            _teacher(self.x0).predict(step, self.conditioning)

    def test_a_row_without_a_target_is_refused(self):
        step = _step(self.x0, self.eps, torch.tensor(0.5))
        with self.assertRaises(ValueError):
            _teacher(self.x0).predict(step, {"height": 512, "width": 512})

    def test_a_mismatched_target_shape_is_refused(self):
        step = _step(self.x0, self.eps, torch.tensor(0.5))
        with self.assertRaises(RuntimeError):
            _teacher(torch.randn(1, 16, 1, 4, 4)).predict(step, self.conditioning)


class OfflineConfigPairingTest(unittest.TestCase):
    def _config(self, objective, teacher_backend):
        from tests.helpers import offline_config

        config = offline_config()
        config.train.objective = objective
        config.teacher.backend = teacher_backend
        config.teacher.model_path = "/assets/checkpoints/JoyAI-Image-Edit"
        return config

    def test_offline_objective_pairs_with_the_offline_teacher(self):
        config = self._config("offline", "offline")
        config.validate()
        self.assertEqual(config.train.rollout.num_inference_steps, 1)

    def test_an_offline_objective_against_a_live_teacher_is_rejected(self):
        config = self._config("offline", "joyedit")
        with self.assertRaises(ValueError):
            config.validate()

    def test_an_offline_teacher_under_an_online_rollout_is_rejected(self):
        config = self._config("online", "offline")
        with self.assertRaises(ValueError):
            config.validate()


class OfflineSigmaCoverageTest(unittest.TestCase):
    """Offline training covers the noise levels at which inference renders detail.

    The rollout's ten shifted timesteps leave (0.024, 0.395) empty, so sigma is drawn from a
    continuous density instead.
    """

    def _sigmas(self, sampling, draws=2000, seed=0):

        from diffusers import FlowMatchEulerDiscreteScheduler

        from i2vt.config import OfflineConfig
        from i2vt.models.bernini.modeling_bernini import BerniniStudentBackend

        student = BerniniStudentBackend.__new__(BerniniStudentBackend)
        student.device = torch.device("cpu")
        student.weight_dtype = torch.float32
        student.config = SimpleNamespace(width=512, height=512, flow_shift=5.0)
        student.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
        source, target = torch.randn(1, 16, 1, 4, 4), torch.randn(1, 16, 1, 4, 4)
        src_img = Image.new("RGB", (4, 4))
        tgt_img = Image.new("RGB", (4, 4), (1, 2, 3))
        student._image_latents = lambda im, size=None: source if im is src_img else target
        student._encode_prompt = lambda p: (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
        student._velocity = lambda noisy, t, pe, ne, src, **kw: noisy * 2.0
        batch = {"source": src_img, "target": tgt_img, "video_edit_prompt": "edit"}
        rollout = SimpleNamespace(num_inference_steps=1, train_steps=0)
        config = OfflineConfig(sigma_sampling=sampling)
        torch.manual_seed(seed)
        return [float(next(iter(student.offline_steps(batch, rollout, config))).sigma) for _ in range(draws)]

    def test_uniform_covers_the_band_the_rollout_schedule_leaves_empty(self):
        sigmas = self._sigmas("uniform")
        in_band = sum(1 for s in sigmas if 0.024 < s < 0.395) / len(sigmas)
        # The 40-step sampler puts roughly an eighth of its steps in this band.
        self.assertGreater(in_band, 0.05)

    def test_every_draw_is_a_usable_noise_level(self):
        for sampling in ("uniform", "logit_normal"):
            with self.subTest(sampling=sampling):
                sigmas = self._sigmas(sampling)
                self.assertTrue(all(0.0 < s <= 1.0 for s in sigmas))

    def test_timestep_tracks_sigma_on_the_thousand_scale(self):

        from diffusers import FlowMatchEulerDiscreteScheduler

        from i2vt.config import OfflineConfig
        from i2vt.models.bernini.modeling_bernini import BerniniStudentBackend

        student = BerniniStudentBackend.__new__(BerniniStudentBackend)
        student.device = torch.device("cpu")
        student.weight_dtype = torch.float32
        student.config = SimpleNamespace(width=512, height=512, flow_shift=5.0)
        student.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
        latents = torch.randn(1, 16, 1, 4, 4)
        src_img = Image.new("RGB", (4, 4))
        student._image_latents = lambda im, size=None: latents
        student._encode_prompt = lambda p: (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
        student._velocity = lambda noisy, t, pe, ne, src, **kw: noisy * 2.0
        batch = {"source": src_img, "target": src_img, "video_edit_prompt": "edit"}
        torch.manual_seed(1)
        step = next(iter(student.offline_steps(batch, SimpleNamespace(num_inference_steps=1, train_steps=0), OfflineConfig())))
        self.assertAlmostEqual(float(step.timestep), float(step.sigma) * 1000.0, places=3)
