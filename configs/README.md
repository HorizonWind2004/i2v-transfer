# Configs

One file per released model: the configuration of its released checkpoint. Each file sets only the
values that differ from the defaults in [`src/i2vt/config.py`](../src/i2vt/config.py) and in the
backend's `inference_config.py`; `i2vt <config> --dry-run` prints every value.

| Config | Model | Teacher CFG | lr | Steps | Effective batch |
|---|---|---|---|---|---|
| [train/i2vt-1.3b.yaml](train/i2vt-1.3b.yaml) | I2VT-1.3B (Bernini-R-1.3B) | 1; action and body movement 2 | 2e-5 | 200 | 48 = 3 × 16 |
| [train/i2vt-14b.yaml](train/i2vt-14b.yaml) | I2VT-14B (Bernini-R-14B, two experts) | 1; action and body movement 2 | 5e-5 | 200 | 48 = 2 × 24 |
| [train/i2vt-cof.yaml](train/i2vt-cof.yaml) | I2VT-CoF (VideoCoF) | 2 | 5e-5 | 200 | 48 = 2 × 24 |

All three train on I2VT-17K with the v-MSE loss, LoRA rank 128 / alpha 64 on every linear layer,
512 × 512 `area` resize, a 10-step rollout with flow shift 5, and student and rollout CFG 1.
Evaluation uses the base editor's own inference settings: 40 steps and 81 frames for Bernini-R,
4 steps with the FusionX LoRA and 33 frames for VideoCoF.

Per-GPU batch is 1 and each GPU holds a student rank and its teacher, so the effective batch is
`len(student_gpus) × train.gradient_accumulation_steps`. On N GPUs, list N ids and set
`gradient_accumulation_steps` to 48 / N.

Weights are read from `${I2V_ASSET_ROOT}/checkpoints/`. Checkpoints go to `outputs/<name>/`,
evaluation outputs to `outputs/inference/<name>/<bench>/` and caches to `cache/<name>/`, all in the
working directory. W&B logging is offline.

```bash
i2vt configs/train/i2vt-1.3b.yaml --dry-run   # print the resolved configuration
i2vt configs/train/i2vt-1.3b.yaml             # train
i2vt configs/train/i2vt-1.3b.yaml --mode train_eval   # train, then evaluate the final checkpoint
```

## Datasets and benchmarks

`--dataset <name>` picks the training set and `--bench <name>` the benchmark. The names resolve to
Hugging Face repos in [`src/i2vt/data/registry.py`](../src/i2vt/data/registry.py). Pools can be
combined, as in `--dataset action+relight`.

| `--dataset` | Contents |
|---|---|
| `i2vt_17k` | [I2VT-17K](https://huggingface.co/datasets/sanaka87/I2VT-17K), 17,227 pairs (default) |
| `openve` | the OpenVE-Bench edit types, 6,026 pairs |
| `action` | action and expression changes, 7,306 pairs |
| `body_movement` | body movements, 1,305 pairs |
| `relight`, `relight_wide` | scene relighting, 726 and 1,864 pairs |

| `--bench` | Contents | Judge |
|---|---|---|
| `openve` | [OpenVE-Bench](https://huggingface.co/datasets/Lewandofski/OpenVE-Bench) without camera edits, 388 clips | `gemini-2.5-pro` |
| `openve_pp` | [OpenVE++](https://huggingface.co/datasets/sanaka87/OpenVEPP-Bench) action change, expression change, body movement and scene relighting, 207 clips | `gemini-3.8-flash` |
