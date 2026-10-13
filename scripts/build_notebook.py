"""Generate notebooks/Yukka_Video_AI_Colab.ipynb.

Kept as a script so notebook content is reviewable in plain Python source.
Run: python scripts/build_notebook.py
"""
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text}


CELLS = []

CELLS.append(md("""
# Yukka Video AI — Google Colab Long-Video Generation

This notebook runs the **real** `GenerationRunner` from your repository
(`Hamsan2013/Yukka-video-ai`) on a Colab GPU and builds long videos from
sequential ~5-second clips.

**Before you start:**
* Enable a GPU runtime: `Runtime → Change runtime type → T4/A100 GPU`.
* Google Drive stores jobs, clips, reference frames and manifests. Colab
  sessions **will** eventually disconnect (GPU time is not unlimited) — just
  re-run the notebook and **resume**; completed segments are never regenerated.
* Model weights are **never downloaded automatically**. You explicitly approve
  each download after seeing its size and VRAM requirements.
"""))

CELLS.append(md("## 1. Environment check — Python, PyTorch, CUDA, GPU, disk space"))
CELLS.append(code('''
import shutil, sys, subprocess

print(f"Python: {sys.version.split()[0]}")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU memory: {free_b/1e9:.1f} GB free / {total_b/1e9:.1f} GB total")
else:
    print("WARNING: No GPU detected. Switch the runtime to GPU "
          "(Runtime -> Change runtime type). CPU generation is supported but "
          "extremely slow for video models.")

total, used, free = shutil.disk_usage("/")
print(f"Local disk: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")

ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin is None:
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"ffmpeg (imageio-ffmpeg bundled): {ffmpeg_bin}")
    except Exception:
        print("WARNING: ffmpeg not found; final assembly unavailable until installed.")
else:
    ver = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True)
    print(f"ffmpeg: {ver.stdout.splitlines()[0]}")
'''))

CELLS.append(md("""## 2. Mount Google Drive (persistent outputs & checkpoints)

A browser authorization window will open. Grant access so jobs, clips,
reference frames and manifests survive session restarts."""))
CELLS.append(code('''
from google.colab import drive
drive.mount("/content/drive")

import os
BASE_DIR = "/content/drive/MyDrive/yukka_video_ai"
JOBS_DIR = os.path.join(BASE_DIR, "jobs")       # durable job manifests + clips
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")  # model weights (explicit downloads)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
print("Persistent base:", BASE_DIR)
print("Jobs/checkpoints:", JOBS_DIR)
print("Model cache:", CACHE_DIR)
'''))

CELLS.append(md("""### Optional: keep the weight cache on local disk (recommended)
Google Drive FUSE is slow for many small files, and diffusers caches models as
many shards. Clips/manifests above still go to Drive (few large files); only
the *weight cache* can move to fast local disk (lost on disconnect — it must
re-download, which is why this is opt-in)."""))
CELLS.append(code('''
# Uncomment to use local disk for the weight cache (faster loads within one session):
# CACHE_DIR = "/root/.cache/yukka_models"; os.makedirs(CACHE_DIR, exist_ok=True)
print("Model cache dir in use:", CACHE_DIR)
'''))

CELLS.append(md("## 3. Clone or safely update the repository"))
CELLS.append(code('''
import subprocess, os, sys

REPO_URL = "https://github.com/Hamsan2013/Yukka-video-ai.git"
REPO_DIR = "/content/yukka-repo"

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); raise RuntimeError(f"Command failed: {cmd}\\n{r.stderr}")
    return r.stdout

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    branch = run(f"cd {REPO_DIR} && git rev-parse --abbrev-ref HEAD").strip()
    print(run(f"cd {REPO_DIR} && git pull --ff-only origin {branch}"))
else:
    print(run(f"git clone {REPO_URL} {REPO_DIR}"))

print(run(f"cd {REPO_DIR} && git log --oneline -1"))
sys.path.insert(0, f"{REPO_DIR}/src")
print("Repo added to sys.path:", f"{REPO_DIR}/src")
'''))

CELLS.append(md("""## 4. Install dependencies (without breaking Colab's PyTorch/CUDA)

Installs the repo's `requirements.txt` **excluding torch/torchvision**, which
Colab already provides with matching CUDA builds. Upgrading them here can
break GPU access."""))
CELLS.append(code('''
import subprocess
pkgs = []
for l in open(f"{REPO_DIR}/requirements.txt"):
    l = l.split("#")[0].strip()
    if not l:
        continue
    name = l.split(">")[0].split("=")[0].split("[")[0].strip().lower()
    if name.startswith("torch"):
        continue  # keep Colab's working torch + CUDA stack
    pkgs.append(l)
print("Installing:", pkgs)
r = subprocess.run(["pip", "install", "-q", *pkgs], capture_output=True, text=True)
print((r.stdout or "")[-2000:])
if r.returncode != 0:
    print((r.stderr or "")[-3000:]); raise RuntimeError("Dependency install failed")
print("torch untouched:", subprocess.run(
    ["python", "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"],
    capture_output=True, text=True).stdout.strip())
'''))

CELLS.append(md("## 5. Import checks + lightweight tests (no GPU, no downloads)"))
CELLS.append(code('''
from yukkavideo.jobs.runner import GenerationRunner, RunnerOptions
from yukkavideo.jobs.manifest import JobStore, JobManifest
from yukkavideo.jobs.control import JobController, GPULock
from yukkavideo.assembly.assembler import VideoAssembler
from yukkavideo.models import (ADAPTER_REGISTRY, detect_model_type,
                               CogVideoAdapter, LTXAdapter, WanAdapter)
from yukkavideo.inference.planning import build_plan
from yukkavideo.continuity.frames import validate_clip_file, extract_final_frame
from yukkavideo.inference.video_exporter import VideoExporter
print("All core imports OK")

import pytest
rc = pytest.main(["-q", f"{REPO_DIR}/tests", "-k", "not gpu and not integration"])
print("Lightweight test exit code:", rc, "(0 = passed)")
'''))

CELLS.append(md("""## 6. Supported models and their REAL capabilities

Flags come from the adapters themselves. CogVideoX honestly reports
`i2v=False` (its pipeline has no image input), so reference-guided
continuation is skipped for it instead of being faked."""))
CELLS.append(code('''
probe_models = [
    "THUDM/CogVideoX-2b",                     # T2V only, ~9 GB weights
    "Lightricks/LTX-Video",                   # T2V + I2V continuation, ~6 GB
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",   # I2V, A100-class GPU, gated
]
rows = []
for name in probe_models:
    a = ADAPTER_REGISTRY[detect_model_type(name)](model_name=name)  # NO loading
    caps = a.get_capabilities()
    mem = a.get_memory_requirements()
    rows.append({
        "model": name,
        "t2v": caps.supports_text_to_video,
        "i2v (continuation)": caps.supports_image_to_video,
        "frames rule": caps.frame_count_constraint or "-",
        "native fps": caps.preferred_fps,
        "min VRAM fp16 GB": caps.approx_vram_gb_fp16,
        "weights approx GB": mem.approx_model_gb if mem else "?",
        "license": caps.license,
        "gated?": caps.gated,
    })
import pandas as pd
pd.DataFrame(rows)
'''))

CELLS.append(md("""## 7. Choose model, prompt, duration, FPS, resolution, settings

Defaults target a **T4 (16 GB)**: CogVideoX-2b @ 480×720 @ 8 fps. On an A100
you can pick LTX-Video (real last-frame continuation) or Wan 14B I2V."""))
CELLS.append(code('''
import ipywidgets as widgets
from IPython.display import display

model_select = widgets.Dropdown(
    options=[("THUDM/CogVideoX-2b (T2V only, fits T4)", "THUDM/CogVideoX-2b"),
             ("Lightricks/LTX-Video (T2V + I2V continuation)", "Lightricks/LTX-Video"),
             ("Wan2.1-I2V-14B-480P (needs A100, gated)", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")],
    description="Model:")
prompt_box = widgets.Text(value="A red fox running through a snowy forest at dusk, cinematic",
                          description="Prompt:", layout=widgets.Layout(width="80%"))
duration_box = widgets.FloatText(value=10.0, min=1.0, step=5.0, description="Seconds:")
fps_box = widgets.IntText(value=8, min=1, max=30, description="FPS:")
res_box = widgets.Dropdown(options=["480x720", "256x512", "720x1280"], value="480x720",
                           description="HxW:")
steps_box = widgets.IntText(value=30, min=10, max=200, description="Steps:")
seg_box = widgets.FloatText(value=5.0, min=1.0, description="Clip sec:")
display(widgets.VBox([model_select, prompt_box, duration_box, fps_box,
                      res_box, steps_box, seg_box]))
print("Make selections, then run section 8.")
'''))

CELLS.append(md("""## 8. Model size / hardware check — explicit approval required

Shows checkpoint size, known VRAM needs, and compares against this GPU.
**Nothing downloads until you set `APPROVE_DOWNLOAD = True` and run 8b.**"""))
CELLS.append(code('''
SELECTED_MODEL = model_select.value
HEIGHT, WIDTH = [int(v) for v in res_box.value.lower().replace("x", " ").split()]
adapter = ADAPTER_REGISTRY[detect_model_type(SELECTED_MODEL)](
    model_name=SELECTED_MODEL, cache_dir=CACHE_DIR)
caps = adapter.get_capabilities()
mem = adapter.get_memory_requirements()

try:
    from huggingface_hub import list_repo_files
    n_files = len(list_repo_files(SELECTED_MODEL))
    print(f"Repo lists {n_files} files (sizes shown by HF; download happens only when you approve).")
except Exception as exc:
    print(f"(Could not query Hugging Face: {exc})")

free_gb = torch.cuda.mem_get_info()[0]/1e9 if torch.cuda.is_available() else 0.0
print(f"Model:       {SELECTED_MODEL}")
if mem:
    print(f"Weights:     ~{mem.approx_model_gb} GB download | peak inference ~{mem.approx_peak_inference_gb} GB")
    print(f"Notes:       {mem.notes}")
print(f"This GPU now: {free_gb:.1f} GB free")
print(f"T2V={caps.supports_text_to_video}  I2V={caps.supports_image_to_video}  "
      f"frame rule: {caps.frame_count_constraint or 'any'}  license: {caps.license}"
      + ("  [GATED: needs HF token]" if caps.gated else ""))
if caps.approx_vram_gb_fp16 and free_gb < caps.approx_vram_gb_fp16:
    print(f"\\nINCOMPATIBLE: needs ~{caps.approx_vram_gb_fp16} GB VRAM, only {free_gb:.1f} GB free. "
          f"Pick a smaller model/resolution (e.g. CogVideoX-2b on T4).")
else:
    print("\\nLooks feasible. Set APPROVE_DOWNLOAD = True and run cell 8b.")

APPROVE_DOWNLOAD = False   # <-- flip manually; large download!
'''))

CELLS.append(md("### 8b. Load the approved model (this performs the big download)"))
CELLS.append(code('''
if not APPROVE_DOWNLOAD:
    print("Not approved. Edit APPROVE_DOWNLOAD = True in cell 8, re-run it, then run this.")
else:
    if torch.cuda.is_available():
        adapter.enable_cpu_offload()  # applied after pipeline construction (fixed lifecycle)
    adapter.load_model()
    print(f"Loaded {SELECTED_MODEL}. is_loaded={adapter.is_loaded}")
'''))

CELLS.append(md("""## 9. Generate a short TEST clip first (~a few minutes on T4)

Uses the real adapter + exporter. The file is validated by decoding it back —
we never claim success without a readable video."""))
CELLS.append(code('''
store = JobStore(JOBS_DIR)
runner = GenerationRunner(adapter, store,
                          options=RunnerOptions(num_inference_steps=steps_box.value))
if not adapter.is_loaded:
    raise RuntimeError("Load the model first (cell 8b).")

test_out = os.path.join(JOBS_DIR, "test_clip.mp4")
num_frames = adapter.adjust_num_frames(int(2 * fps_box.value))   # 2-second test
out = adapter.generate_clip(prompt=prompt_box.value, num_frames=num_frames,
                            height=HEIGHT, width=WIDTH, seed=42,
                            num_inference_steps=steps_box.value)
VideoExporter().export(out, test_out, fps=fps_box.value)
frames = validate_clip_file(test_out)   # raises ClipValidationError if unreadable
print(f"REAL test clip generated & verified: {test_out} ({frames} frames, "
      f"{os.path.getsize(test_out)/1e6:.1f} MB)")
from IPython.display import Video as ShowVideo
ShowVideo(test_out, embed=True, width=480)
'''))

CELLS.append(md("""## 10. Create + run the sequential long-video job

`create_job` plans the frame budget into ~5 s segments; `run()` generates them
sequentially, checkpointing clip file + final reference frame + metadata +
manifest to Drive after every segment. If the model supports image-to-video,
each next segment continues from the previous clip's last frame; otherwise
continuation is honestly skipped."""))
CELLS.append(code('''
for w in runner.preflight(HEIGHT, WIDTH):
    print("WARNING:", w)

manifest = runner.create_job(
    prompt=prompt_box.value,
    duration_seconds=duration_box.value,
    fps=fps_box.value,
    height=HEIGHT, width=WIDTH,
    segment_seconds=seg_box.value,
    seed=42,
    model_name=SELECTED_MODEL,
)
JOB_ID = manifest.job_id
print(f"Job {JOB_ID}: {len(manifest.segments)} segments, "
      f"{manifest.total_frames} frames planned -> {JOBS_DIR}/{JOB_ID}")

result = runner.run(JOB_ID)
print(f"Final status: {result.status}  progress: {result.progress:.0%}")
if result.status == "failed":
    print("Errors:", *result.errors, sep="\\n  ")
'''))

CELLS.append(md("""## 11. Progress, pause/cancel, checkpoint locations"""))
CELLS.append(code('''
m = store.load(JOB_ID)
print(f"status={m.status} progress={m.progress:.0%} updated={m.updated_at}")
for s in m.segments:
    print(f"  seg {s.index:02d}: {s.status:<8} frames={s.generated_frames} "
          f"ref={'yes' if s.reference_frame_path else '-'} "
          f"out={s.output_path or '-'}" + (f" err={s.error}" if s.error else ""))
print("Job folder (persistent):", store.job_dir(JOB_ID))

# Stop safely at the next segment boundary by running one of these:
#   JobController(store, JOB_ID).request_pause()
#   JobController(store, JOB_ID).request_cancel()
'''))

CELLS.append(md("""## 12. Resume an interrupted job (after any disconnect/restart)

Re-run sections 1–8b after reconnecting, then use this. Completed segments are
validated on disk and **skipped**; only missing/incomplete ones regenerate."""))
CELLS.append(code('''
unfinished = [j for j in store.unfinished_jobs()
              if j.status in ("queued", "running", "paused", "failed")]
opts = [(f"{j.job_id} ({j.status}, "
         f"{sum(1 for s in j.segments if s.status=='done')}/{len(j.segments)} done)",
         j.job_id) for j in unfinished] or [("(none)", "")]
resume_select = widgets.Dropdown(options=opts, description="Resume:")
display(resume_select)
'''))
CELLS.append(code('''
if resume_select.value:
    JobController(store, resume_select.value).clear_signals()
    resumed = runner.run(resume_select.value)
    JOB_ID = resume_select.value
    print(f"After resume: status={resumed.status} progress={resumed.progress:.0%}")
else:
    print("No resumable job selected.")
'''))

CELLS.append(md("""## 13. Assemble segments into the final MP4 (real FFmpeg path)

`VideoAssembler` verifies resolution/fps/codec compatibility, uses fast
stream-copy concat when possible, re-encodes otherwise, and reports honest
duration deviation. (The runner also auto-assembles on completion; this cell
lets you rebuild manually.)"""))
CELLS.append(code('''
m = store.load(JOB_ID)
done_segs = sorted([s for s in m.segments if s.status == "done"],
                   key=lambda s: s.index)
paths = [s.output_path for s in done_segs]
final_path = os.path.join(JOBS_DIR, f"final_{m.job_id}.mp4")
asm = VideoAssembler()
res = asm.assemble(segment_paths=paths, output_path=final_path,
                   target_fps=m.fps, target_size=(m.width, m.height),
                   requested_duration=m.duration_seconds)
print(f"Assembled via '{res.method}'. Requested {res.requested_duration}s, "
      f"actual {res.actual_duration:.2f}s (deviation {res.deviation:+.2f}s)")
for n in res.notes:
    print(" note:", n)
'''))

CELLS.append(md("## 14. Final video: metadata, preview, download"))
CELLS.append(code('''
p = m.final_video_path if (m.final_video_path and os.path.exists(m.final_video_path)) else final_path
if not os.path.exists(p):
    print("No final video yet.")
else:
    info = asm.probe(p)
    print(f"File:      {p}")
    print(f"Duration:  {info.duration:.2f} s | Resolution: {info.width}x{info.height} "
          f"| FPS: {info.fps:.2f} | Codec: {info.codec} | Size: {os.path.getsize(p)/1e6:.1f} MB")
    from IPython.display import Video as SV
    SV(p, embed=True, width=720)
    if os.path.getsize(p) < 50_000_000:
        from google.colab import files
        files.download(p)   # larger files: fetch them from Google Drive instead
    print("Persistent copy lives in:", JOBS_DIR)
'''))

CELLS.append(md("""## Appendix: background worker for very long jobs

For long jobs, run the repo's real worker process so the kernel stays
responsive; poll the manifest from the notebook."""))
CELLS.append(code('''
# Uncomment when needed:
# proc = subprocess.Popen(["python", "-m", "yukkavideo.jobs.worker",
#                          "--jobs-dir", JOBS_DIR, "--cache-dir", CACHE_DIR],
#                         cwd=REPO_DIR, env={**os.environ,
#                                            "PYTHONPATH": f"{REPO_DIR}/src"})
# print("worker pid:", proc.pid)
# Watch progress with: store.load(JOB_ID).progress
print("Appendix reviewed (nothing started).")
'''))

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3",
                       "language": "python"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("notebooks/Yukka_Video_AI_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote notebooks/Yukka_Video_AI_Colab.ipynb,", len(CELLS), "cells")
