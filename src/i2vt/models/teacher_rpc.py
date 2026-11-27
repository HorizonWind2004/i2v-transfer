"""Remote teacher prediction and TCP serving."""

from __future__ import annotations

import pickle
import socket
import socketserver
import struct
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import Iterable

from PIL import Image
import torch

from i2vt.models.modeling_utils import DistillationStep, TeacherBackend

import io

HEADER = struct.Struct("!Q")


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def bytes_to_pil(x: bytes) -> Image.Image:
    return Image.open(io.BytesIO(x)).convert("RGB")


def _recv_exact(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading RPC payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock):
    size = HEADER.unpack(_recv_exact(sock, HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))


def _send_message(sock, payload):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(data)))
    sock.sendall(data)


def call_teacher(endpoint: str, payload: dict, timeout: float | None = None):
    host, port = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        _send_message(sock, payload)
        response = _recv_message(sock)
    if response.get("status") != "ok":
        raise RuntimeError(response.get("error", "Teacher RPC failed"))
    return response["result"]


class RPCTeacherBackend(TeacherBackend):
    def __init__(self, endpoints: Iterable[str], rank: int, timeout: float | None = None, max_workers: int = 0):
        self.endpoints = [x.strip() for x in endpoints if x.strip()]
        if not self.endpoints:
            raise ValueError("At least one teacher endpoint is required")
        self.rank = rank
        self.timeout = timeout
        self._counter = count()
        if max_workers <= 0:
            max_workers = len(self.endpoints)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        # (image, png) for the sample being rolled out; see _source_png.
        self._png_memo: tuple[Image.Image | None, bytes | None] = (None, None)

    def _source_png(self, source):
        """Encode the source frame once per sample instead of once per step.

        Every rollout step of a sample sends the same image, and PIL spends
        ~95 ms compressing a 512x512 frame to PNG -- 15 s per optimizer step at
        160 requests, paid on the training thread. Holding a reference to the
        memoized image keeps its identity valid, so `is` cannot alias a freed
        image onto a new one at the same address.
        """
        cached_source, cached_png = self._png_memo
        if cached_source is not source:
            cached_png = pil_to_bytes(source)
            self._png_memo = (source, cached_png)
        return cached_png

    def vae_signature(self):
        return call_teacher(self.endpoints[self.rank % len(self.endpoints)], {"type": "vae_signature"}, self.timeout)

    @staticmethod
    def _to_host(tensors):
        """Start the device-to-host copies without blocking the training thread.

        A plain .cpu() waits until the GPU has finished every kernel queued so far, so the training
        thread could not queue the next rollout step while the teacher request was being built. The
        copies go into pinned buffers on the current stream instead, and the sender thread waits on
        an event recorded after them. The bytes sent are the same.
        """
        if not all(t.is_cuda for t in tensors):
            return [t.cpu() for t in tensors], None
        host = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors]
        for h, t in zip(host, tensors):
            h.copy_(t, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return host, event

    def submit(self, step, conditioning):
        source = conditioning.get("joyedit_source") or conditioning["source"]
        (sigma, latent, timestep), ready = self._to_host(
            [step.sigma.detach(), step.noisy_latents.detach(), step.timestep.float().reshape(1).detach()]
        )
        payload = {
            "type": "velocity",
            "sigma": sigma,
            "latent": latent,
            "timestep": timestep,
            "prompt": conditioning["joy_edit_prompt"],
            "source_png": self._source_png(source),
            "height": int(conditioning["height"]),
            "width": int(conditioning["width"]),
            # The teacher picks its CFG per sample from these two (guidance_scale_by_type/by_task);
            # without them every sample would silently use the default scale.
            "edit_type": conditioning.get("edit_type"),
            "task": conditioning.get("task"),
        }
        endpoint = self.endpoints[(self.rank + next(self._counter)) % len(self.endpoints)]

        def send():
            if ready is not None:
                ready.synchronize()
            return call_teacher(endpoint, payload, self.timeout)

        return self._executor.submit(send)

    def predict(self, step, conditioning):
        return self.submit(step, conditioning).result().to(step.noisy_latents.device).detach()

    def close(self):
        self._executor.shutdown(wait=True)


class _TeacherHandler(socketserver.BaseRequestHandler):
    def handle(self):
        request = _recv_message(self.request)
        try:
            result = self.server.handle_rpc(request)
            _send_message(self.request, {"status": "ok", "result": result})
        except Exception as exc:
            _send_message(self.request, {"status": "error", "error": repr(exc)})


class TeacherServer(socketserver.TCPServer):
    allow_reuse_address = True
    # One connection per RPC and several in-flight requests per student rank, so the
    # stock backlog of 5 refuses connects once teacher.client_workers is raised.
    request_queue_size = 256

    def __init__(self, server_address, teacher: TeacherBackend):
        super().__init__(server_address, _TeacherHandler)
        self.teacher = teacher

    def handle_rpc(self, request: dict):
        kind = request.get("type")
        if kind == "vae_signature":
            return self.teacher.vae_signature()
        if kind != "velocity":
            raise ValueError(f"Unknown teacher RPC type: {kind}")
        latent = request["latent"].to(device=self.teacher.device)
        timestep = request["timestep"].to(device=self.teacher.device)
        step = DistillationStep(latent, latent, timestep, request["sigma"].to(self.teacher.device))
        velocity = self.teacher.predict(
            step,
            {
                "joy_edit_prompt": request["prompt"],
                "source": bytes_to_pil(request["source_png"]),
                "height": int(request["height"]),
                "width": int(request["width"]),
                "edit_type": request.get("edit_type"),
                "task": request.get("task"),
            },
        )
        return velocity.detach().cpu()
