# Yukka Video AI

**Modular, resumable long-video generation from sequential short clips.**

Yukka Video AI divides a requested video (30 s, 2 min, 10 min — no hard-coded
maximum) into ~5-second segments, generates them sequentially with a
Diffusers-based model adapter, maintains visual continuity by feeding the last
frame of each clip into image-to-video-capable models, checkpoints every
segment to a durable on-disk job manifest, and assembles the result into one
final MP4 with FFmpeg. Jobs can be paused, cancelled, and **resumed after any
interruption** (crash, disconnect, Colab pre-emption) without regenerating
completed segments.

---

## Quick start on Google Colab (recommended if you have no local GPU)

Notebook path in this repository:

```
notebooks/Yukka_Video_AI_Colab.ipynb
```

### Steps

1. Open GitHub → `Hamsan2013/Yukka-video-ai` → switch to the branch containing
   the notebook (e.g. the development branch).
2. Click `notebooks/Yukka_Video_AI_Colab.ipynb`. Alternatively open directly at:
   `https://colab.research.google.com/github/Hamsan2013/Yukka-video-ai/blob/<branch>/notebooks/Yukka_Video_AI_Colab.ipynb`
3. In Colab: **Runtime → Change runtime type → T4 (or A100) GPU**.
4. Run the cells in order:
   - **§1 Environment check** — Python/PyTorch/CUDA/GPU/disk.
   - **§2 Mount Google Drive** — browser authorization window will pop up;
     jobs, clips, reference frames and manifests are stored under
     `/content/drive/MyDrive/yukka_video_ai` so they survive disconnects.
   - **§3 Clone/update repo**, **§4 Install dependencies** — installs
     `requirements.txt` *without* replacing Colab's working PyTorch/CUDA stack
     (torch/torchvision are skipped when already present).
   - **§5 Import checks + lightweight tests** — runs the CPU-only pytest suite;
     no GPU or downloads required.
   - **§6 Model capability table** — real flags read from the adapters.
   - **§7 Choose model / prompt / duration / FPS / resolution.**
   - **§8 Size & VRAM check** — shows approximate download size and peak VRAM,
     compares against free GPU memory, then requires you to manually set
     `APPROVE_DOWNLOAD = True` before **§8b** loads weights. Nothing large is
     downloaded until you do that.
   - **§9 Short test clip** — generate a ~2-second clip first; the file is
     decoded back and validated before you commit to a long job.
   - **§10 Sequential generation** — creates a `GenerationRunner` job and runs
     it segment-by-segment with checkpointing.
   - **§11–12 Progress & resume** — status display; after any disconnect just
     re-run §1–§8b and use the resume cell. Completed segments are skipped.
   - **§13–14 Assembly & download** — real FFmpeg concat, final metadata
     (duration/resolution/FPS/size) and a download button.

No web server or public tunnel is needed for the notebook; the CLI and API
work independently of it.

### Honest Colab caveats

* Colab GPU time is **not unlimited**; sessions get pre-empted. That is exactly
  why every segment is checkpointed — resume, don't restart.
* First generation downloads several GB of weights (only after your explicit
  approval). Keep the HF cache on local disk (`~/.cache/huggingface`) rather
  than Drive — Drive FUSE is slow for many small files. Outputs and manifests
  go to Drive.
* **Real end-to-end GPU generation has not yet been verified by the maintainers
  in a live Colab session.** The controller, adapters, exporter, assembler and
  all plumbing are covered by automated tests with fake adapters and real
  FFmpeg clips, but please run §9 (short test clip) before longer jobs and
  report issues.

---

## Supported models (real capabilities)

| Model ID | Adapter | Text→Video | Image→Video (continuation) | Frame rule | Approx. weights | Notes |
|---|---|---|---|---|---|---|
| `THUDM/CogVideoX-2b` | `CogVideoAdapter` | ✅ | ❌ (pipeline has no image input — never faked) | 8k+1, ≤81 frames | ~9 GB | Fits a 16 GB T4 with CPU offload; best default on Colab |
| `Lightricks/LTX-Video` | `LTXAdapter` | ✅ | ✅ via `LTXImageToVideoPipeline` | model-specific snapping | ~6 GB | Supports reference-guided continuation (last-frame → next clip) |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `WanAdapter` | ✅ (T2V variant) | ✅ via `WanImageToVideoPipeline` | 4k+1 frames | large (14B) | Gated on Hugging Face (accept license + token); realistically A100-class |

Capabilities are reported by `adapter.get_capabilities()` — the runner checks
`supports_image_to_video` before ever passing a reference frame to a model.
Text-to-video-only models simply continue from the scene plan instead.

**Continuity limitation (documented honestly):** last-frame references and
fixed seeds *improve* coherence between clips but cannot guarantee identical
characters or perfectly continuous motion across segments.

Model selection helper (no downloads):

```python
from yukkavideo.models import ADAPTER_REGISTRY, detect_model_type
adapter = ADAPTER_REGISTRY[detect_model_type("Lightricks/LTX-Video")](
    model_name="Lightricks/LTX-Video")
print(adapter.get_capabilities())      # supports i2v? dimensions? frame rule?
print(adapter.get_memory_requirements())# approx download & peak VRAM
```

Adapters load genuine pretrained Diffusers pipelines
(`CogVideoXPipeline.from_pretrained(...)` etc.). There is no mock generation
path in production code.

---

## Installation (local machine)

```bash
git clone https://github.com/Hamsan2013/Yukka-video-ai.git
cd Yukka-video-ai
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA build
pip install -r requirements.txt        # diffusers, transformers, fastapi, ...
pip install -r requirements-dev.txt    # + pytest, httpx (for tests)
```

FFmpeg is required for assembly/validation. Either install it system-wide
(`apt install ffmpeg` / `brew install ffmpeg`), or the bundled fallback from
the `imageio-ffmpeg` pip package is used automatically. You may also point at a
binary explicitly: `export YUKKA_FFMPEG_PATH=/usr/bin/ffmpeg`.

CPU-only execution works with these models but is **extremely slow** (hours per
clip). It is supported, not recommended.

---

## CLI usage

All commands work without Colab. Run as `python -m yukkavideo.cli.main …` or an
installed `yukkavideo` entry point.

```bash
# Create and immediately run a 30-second job at 8 FPS (no hard max duration)
python -m yukkavideo.cli.main job-start "a red fox running through a snowy forest" \
    --duration 30 --fps 8 --height 480 --width 720 \
    -m Lightricks/LTX-Video --segment-seconds 5 \
    --jobs-dir outputs/jobs --cache-dir ~/.cache/huggingface

# Only queue the job (a worker will pick it up later)
... job-start "..." --enqueue-only

# Inspect
python -m yukkavideo.cli.main job-list   --jobs-dir outputs/jobs
python -m yukkavideo.cli.main job-status <job_id>

# Pause safely at the next segment boundary / cancel outright
python -m yukkavideo.cli.main job-pause  <job_id>
python -m yukkavideo.cli.main job-cancel <job_id>

# Resume — completed segments are validated on disk and skipped
python -m yukkavideo.cli.main job-run    <job_id>

# Assemble existing completed segments into the final MP4
python -m yukkavideo.cli.main job-assemble <job_id> -o final.mp4

# Background worker loop (single-GPU lock prevents concurrent jobs)
python -m yukkavideo.cli.main worker --jobs-dir outputs/jobs [--once]

# Hardware/model info
python -m yukkavideo.cli.main info
python -m yukkavideo.cli.main generate "single short clip" -o clip.mp4   # legacy single-clip mode
```

## HTTP API

```bash
uvicorn yukkavideo.api.server:app
```

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | Create a persisted job (**does not load/download weights**) |
| `GET /jobs` / `GET /jobs/{id}` | List / inspect jobs (survives restarts) |
| `POST /jobs/{id}/pause` / `/resume` / `/cancel` | Lifecycle signals honoured at segment boundaries |
| `GET /jobs/{id}/video` | Download finished MP4 (409 until ready) |
| `GET /health`, `GET /models` | Health & capability listing |

A background worker (`python -m yukkavideo.jobs.worker`) executes queued jobs
under a cross-process GPU lock so concurrent jobs cannot exhaust VRAM.

---

## Persistent checkpoints & resume

Each job lives in `<jobs-dir>/<job_id>/` with:

* `manifest.json` — atomic writes (tmp + fsync + rename); records status
  (`queued/running/paused/completed/failed/canceled`), prompt, scene plan,
  duration/FPS/resolution, model config, per-segment paths, reference-frame
  paths, seeds, errors/retry counts, timestamps, progress.
* `segments/segment_XXXX.mp4` — generated clips (never deleted on later failure)
* `references/segment_XXXX_last.png` — extracted final frames for continuity
* `scene_plan.json`, pause/cancel signal files

After a crash/disconnect: list unfinished jobs (`job-list`, `store.unfinished_jobs()`,
or `GET /jobs`) and resume with `job-run <id>` / `POST /jobs/{id}/resume`.
Segments already marked done are re-validated on disk and **not regenerated**.

Set `--jobs-dir` to a mounted persistent drive (e.g. Google Drive) for durability.

---

## Configuration

Presets in `configs/`: `colab_t4.yaml` (low-VRAM: CogVideoX-2b, 480×720, CPU
offload), `colab_a100.yaml` (larger models/resolutions), `default.yaml`.
Configurable: model & cache dir, device/precision, resolution/FPS, segment
length, output/temp dirs, offloading & VAE options, max concurrent jobs, retry
behaviour, FFmpeg path, checkpoint interval, opt-in intermediate cleanup.
Never put HF tokens/API keys in source or manifests — use environment
variables (`HUGGING_FACE_HUB_TOKEN`).

On out-of-memory the runner fails the segment with an actionable message
(smaller resolution/steps, enable offload). Automatic downscale-retry happens
only if explicitly enabled in `RunnerOptions`, and the deviation is recorded in
the manifest — quality is never changed silently.

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -q          # full suite, CPU-only, no downloads
PYTHONPATH=src python -m pytest tests/ -q -k "not gpu and not integration"
```

The suite covers config loading, frame-budget math, segment planning (including
final partial segments), job-state transitions, manifest persistence/recovery,
resume-without-regeneration, invalid clip handling, reference-frame extraction
failures, capability checks, API validation (incl. proof that `POST /jobs`
loads no models), FFmpeg assembly, and mocked OOM paths — all with fake
adapters. GPU/model integration tests are marked and only run when you
explicitly configure a suitable model + hardware.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower resolution (`256x512`), fewer steps, enable CPU offload (default in Colab preset), or choose CogVideoX-2b / LTX-Video over 14B Wan. |
| Model won't fit GPU | The capability/VRAM check (§8) prints the shortfall before download; pick the suggested smaller model. |
| FFmpeg not found | `apt install ffmpeg`, or rely on the bundled `imageio-ffmpeg`, or set `YUKKA_FFMPEG_PATH`. |
| Colab disconnected mid-job | Re-run setup cells, then the resume cell / `job-run <id>` — completed segments are skipped. |
| `gated repo` error (Wan) | Accept the license on Hugging Face and export `HUGGING_FACE_HUB_TOKEN`. |
| Corrupt manifest | Writes are atomic; the loader tolerates damage and re-validates segment files before trusting them. |
| Slow Drive I/O | Keep the weight cache on local disk (see notebook §2 note); only outputs/manifests need Drive. |

---

## Licensing

* This application code: see repository license file.
* Model weights carry their own licenses (CogVideoX: Apache-2.0; LTX-Video:
  LTX Community License — review terms; Wan2.1: Apache-2.0, some repos gated).
* Dependencies (PyTorch, Diffusers, Transformers, FFmpeg) have separate
  licenses. Verify commercial-use terms for both weights and outputs.

## Limitations

* Long videos consume substantial GPU time and storage (minutes per second of
  video on a T4; expect hours for multi-minute targets).
* Cross-clip identity/motion consistency is improved — not guaranteed.
* Audio track support is a planned future feature (assembly is audio-ready).
* Offline operation requires all code, dependencies and weights to already be
  present locally.
