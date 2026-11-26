from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from i2vt.loss import (
    velocity_channel_cosine_loss,
    x0_channel_cosine_loss,
)


class X0ChannelCosineLossTest(unittest.TestCase):
    def test_reference_angles_and_float32_computation(self) -> None:
        first_axis = torch.tensor([1.0, 0.0], dtype=torch.bfloat16).view(1, 2, 1, 1, 1)
        second_axis = torch.tensor([0.0, 1.0], dtype=torch.bfloat16).view_as(first_axis)

        identical = x0_channel_cosine_loss(first_axis, first_axis)
        orthogonal = x0_channel_cosine_loss(first_axis, second_axis)
        opposite = x0_channel_cosine_loss(first_axis, -first_axis)

        self.assertEqual(identical.dtype, torch.float32)
        self.assertEqual(orthogonal.dtype, torch.float32)
        self.assertEqual(opposite.dtype, torch.float32)
        torch.testing.assert_close(identical, torch.tensor(0.0))
        torch.testing.assert_close(orthogonal, torch.tensor(1.0))
        torch.testing.assert_close(opposite, torch.tensor(2.0))


class VelocityChannelCosineLossTest(unittest.TestCase):
    def test_matches_reference_reduction_and_detaches_teacher(self) -> None:
        student = (
            torch.tensor(
                [1.0, 0.0, 0.0, 2.0],
                dtype=torch.bfloat16,
            )
            .view(1, 2, 1, 1, 2)
            .requires_grad_()
        )
        teacher = (
            torch.tensor(
                [1.0, 0.0, 0.0, -3.0],
                dtype=torch.bfloat16,
            )
            .view_as(student)
            .requires_grad_()
        )

        loss = velocity_channel_cosine_loss(student, teacher)
        expected = (
            1.0
            - F.cosine_similarity(
                student.float(),
                teacher.detach().float(),
                dim=1,
                eps=1e-8,
            ).mean()
        )

        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
