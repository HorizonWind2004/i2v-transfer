# Adapted from https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo/attention

import os
import sys
import torch
from einops import rearrange

_FLASH_ATTN_IMPORT_ERROR = None

try:
    # Check for Flash Attention 3 installation path
    flash_attn3_path = os.getenv("FLASH_ATTN3_PATH")
    if flash_attn3_path:
        print(f"Using Flash Attention 3 from: {flash_attn3_path}")
        sys.path.insert(0, flash_attn3_path)
        from flash_attn_interface import flash_attn_varlen_func
    else:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError as exc:
    flash_attn_varlen_func = None
    _FLASH_ATTN_IMPORT_ERROR = exc



def is_flash_attn_available() -> bool:
    return flash_attn_varlen_func is not None


def get_preferred_attention_backend() -> str:
    return "flash_attn" if is_flash_attn_available() else "torch_spda"


def describe_attention_backend() -> str:
    backend = get_preferred_attention_backend()
    if backend == "flash_attn":
        return "flash_attn"
    if _FLASH_ATTN_IMPORT_ERROR is None:
        return "torch_spda"
    return f"torch_spda (flash_attn unavailable: {_FLASH_ATTN_IMPORT_ERROR})"


def _valid_token_mask(cu_seqlens, batch_size: int, max_len: int, device) -> torch.Tensor:
    """Recover which positions of each batch element hold real tokens.

    get_cu_seqlens packs every batch element into a slot of `max_len` and splits
    it in two: [0, s) for image tokens plus unpadded text, and [s, max_len) for
    the right-hand text padding. Entry 2*i+1 of cu_seqlens is i*max_len + s.
    """
    offsets = torch.arange(batch_size, device=cu_seqlens.device, dtype=torch.long) * max_len
    lengths = cu_seqlens[1 : 2 * batch_size : 2].to(torch.long) - offsets
    positions = torch.arange(max_len, device=device)
    return positions.unsqueeze(0) < lengths.to(device).unsqueeze(1)


def _varlen_block_mask(cu_seqlens_q, cu_seqlens_kv, batch_size, max_seqlen_q, max_seqlen_kv, device):
    """Express flash_attn_varlen's sequence isolation as a boolean SDPA mask.

    Varlen attention treats the valid and padding segments as separate
    sequences, so a token attends only to positions in its own segment.
    """
    valid_q = _valid_token_mask(cu_seqlens_q, batch_size, max_seqlen_q, device)
    valid_kv = _valid_token_mask(cu_seqlens_kv, batch_size, max_seqlen_kv, device)
    return (valid_q.unsqueeze(2) == valid_kv.unsqueeze(1)).unsqueeze(1)


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1],
                             dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "flash_attn",
    *,
    causal: bool = False,
    softmax_scale: float = None,
    attn_kwargs: dict = None,
):
    """
    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads
    """
    assert backend in ["flash_attn"], f"Unsupported attention backend: {backend}"
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Input tensors must be 4D"
    batch_size = q.shape[0]

    cu_seqlens_q = attn_kwargs['cu_seqlens_q']
    cu_seqlens_kv = attn_kwargs['cu_seqlens_kv']
    max_seqlen_q = attn_kwargs['max_seqlen_q']
    max_seqlen_kv = attn_kwargs['max_seqlen_kv']
    if flash_attn_varlen_func is None:
        # torch_spda path promised by get_preferred_attention_backend(): flash-attn
        # ships no build for this torch/arch combination, so run exact attention
        # through SDPA and rebuild varlen's segment isolation as a mask.
        assert not causal, "the torch_spda fallback does not implement causal attention"
        attn_mask = _varlen_block_mask(
            cu_seqlens_q, cu_seqlens_kv, batch_size, max_seqlen_q, max_seqlen_kv, q.device
        )
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            scale=softmax_scale,
        ).transpose(1, 2)

    x = flash_attn_varlen_func(
        q.view(q.shape[0] * q.shape[1], *q.shape[2:]),
        k.view(k.shape[0] * k.shape[1], *k.shape[2:]),
        v.view(v.shape[0] * v.shape[1], *v.shape[2:]),
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )
    output = x.view(
        batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
    )

    return output
