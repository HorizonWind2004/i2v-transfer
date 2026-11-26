"""The RPC teacher must hand the server what the teacher needs to pick its CFG per sample."""
import threading

import torch
from PIL import Image

from i2vt.models.modeling_utils import DistillationStep, TeacherBackend
from i2vt.models.teacher_rpc import RPCTeacherBackend, TeacherServer


class RecordingTeacher(TeacherBackend):
    device = torch.device("cpu")

    def __init__(self):
        self.seen = []

    def predict(self, step, conditioning):
        self.seen.append(conditioning)
        return torch.zeros_like(step.noisy_latents)

    def vae_signature(self):
        return {"ok": 1}


def test_edit_type_and_task_reach_the_teacher():
    teacher = RecordingTeacher()
    server = TeacherServer(("127.0.0.1", 0), teacher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    client = RPCTeacherBackend([f"127.0.0.1:{port}"], rank=0, timeout=30)
    try:
        latent = torch.zeros(1, 4, 1, 2, 2)
        step = DistillationStep(latent, latent, torch.tensor([500.0]), torch.tensor([0.5]))
        cond = {"joy_edit_prompt": "make the person angry", "source": Image.new("RGB", (8, 8)),
                "height": 8, "width": 8, "edit_type": "action_edit", "task": "angry"}
        out = client.predict(step, cond)
        assert out.shape == latent.shape
        seen = teacher.seen[0]
        assert seen["edit_type"] == "action_edit" and seen["task"] == "angry"
        assert seen["joy_edit_prompt"] == "make the person angry"
        assert client.vae_signature() == {"ok": 1}
    finally:
        client.close()
        server.shutdown()
        server.server_close()
