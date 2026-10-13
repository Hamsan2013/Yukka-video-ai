"""
Final-video assembly from sequential segment clips using FFmpeg.

Strategy:
1. Probe every segment (resolution, fps, codec, pixel format, frame count).
2. If ALL segments share compatible parameters -> try fast stream-copy
   concat (``-c copy``). This is lossless and avoids frame duplication.
3. If any parameter differs, or stream-copy fails -> re-encode all segments
   to a common profile (libx264 / yuv420p / target fps) via the concat
   filter, which also handles variable frame rates without gaps.
4. Optional short crossfade transitions (xfade) between consecutive clips.
   Crossfades overlap ``transition_seconds`` per join, so the final duration
   shrinks by transition_seconds * (n_clips - 1); this is reported honestly.

FFmpeg discovery order: explicit path -> YUKKA_FFMPEG_PATH env -> ``ffmpeg``
on PATH -> bundled imageio-ffmpeg binary.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


class FFmpegNotFoundError(RuntimeError):
    pass


class AssemblyError(RuntimeError):
    pass


def find_ffmpeg(explicit: Optional[Union[str, Path]] = None) -> str:
    """Locate an ffmpeg executable; raise with install instructions if absent."""
    candidates: List[str] = []
    if explicit:
        candidates.append(str(explicit))
    env = os.environ.get("YUKKA_FFMPEG_PATH")
    if env:
        candidates.append(env)
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in candidates:
        if c and Path(c).exists():
            return c
        if c and shutil.which(c):
            return shutil.which(c)
    raise FFmpegNotFoundError(
        "FFmpeg was not found. Install it with one of:\n"
        "  apt-get install ffmpeg   (Debian/Ubuntu/Colab)\n"
        "  brew install ffmpeg      (macOS)\n"
        "  pip install imageio-ffmpeg  (bundled binary fallback)\n"
        "or set YUKKA_FFMPEG_PATH=/path/to/ffmpeg"
    )


@dataclass
class ClipInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    codec: str
    pix_fmt: str
    duration: float

    def to_dict(self) -> dict:
        return {
            "path": self.path, "width": self.width, "height": self.height,
            "fps": round(self.fps, 4), "frames": self.frames,
            "codec": self.codec, "pix_fmt": self.pix_fmt,
            "duration": round(self.duration, 4),
        }


@dataclass
class AssemblyResult:
    output_path: str
    clips_used: List[str]
    method: str            # "stream_copy" | "reencode" | "crossfade"
    actual_duration: float
    requested_duration: Optional[float]
    deviation: Optional[float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "clips_used": self.clips_used,
            "method": self.method,
            "actual_duration_seconds": round(self.actual_duration, 4),
            "requested_duration_seconds": self.requested_duration,
            "deviation_seconds": (
                round(self.deviation, 4) if self.deviation is not None else None
            ),
            "notes": self.notes,
        }


class VideoAssembler:
    def __init__(self, ffmpeg_path: Optional[Union[str, Path]] = None):
        self.ffmpeg = find_ffmpeg(ffmpeg_path)

    # ------------------------------------------------------------------
    def _run(self, args: List[str], timeout: int = 3600) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [str(a) for a in args], capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise AssemblyError(
                f"FFmpeg failed (exit {proc.returncode}): "
                f"{' '.join(str(a) for a in args[:8])} ...\n"
                f"stderr tail:\n{proc.stderr[-2000:]}"
            )
        return proc

    def probe(self, path: Union[str, Path]) -> ClipInfo:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise AssemblyError(f"Segment file missing or empty: {p}")
        ffprobe = str(p.parent) and None  # noqa: use ffmpeg itself for robustness
        cmd = [self.ffmpeg, "-hide_banner", "-i", str(p), "-map", "0:v:0",
               "-c:v", "copy", "-f", "null", os.devnull]
        # Get structured info via ffprobe when available, else parse ffmpeg stderr.
        ffprobe_exe = self._find_ffprobe()
        if ffprobe_exe:
            probe_cmd = [
                ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt:format=duration",
                "-of", "json", str(p),
            ]
            proc = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise AssemblyError(f"ffprobe failed on {p}: {proc.stderr[-500:]}")
            data = json.loads(proc.stdout or "{}")
            st = (data.get("streams") or [{}])[0]
            fmt = data.get("format") or {}
            num, _, den = str(st.get("r_frame_rate", "0/1")).partition("/")
            try:
                fps = float(num) / float(den) if float(den or 1) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            frames = int(st.get("nb_frames") or 0)
            duration = float(fmt.get("duration") or 0.0)
            if frames <= 0 and duration > 0 and fps > 0:
                frames = int(round(duration * fps))
            return ClipInfo(
                path=str(p), width=int(st.get("width") or 0),
                height=int(st.get("height") or 0), fps=fps, frames=frames,
                codec=str(st.get("codec_name") or ""),
                pix_fmt=str(st.get("pix_fmt") or ""),
                duration=duration or (frames / fps if fps else 0.0),
            )
        # Fallback: run the null mux to ensure decodability, report minimal info.
        self._run(cmd)
        return ClipInfo(path=str(p), width=0, height=0, fps=0.0, frames=0,
                        codec="unknown", pix_fmt="unknown", duration=0.0)

    @staticmethod
    def _find_ffprobe() -> Optional[str]:
        cand = os.environ.get("YUKKA_FFPROBE_PATH") or shutil.which("ffprobe")
        if cand:
            return cand
        try:
            import imageio_ffmpeg

            d = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
            for name in ("ffprobe", "ffprobe.exe"):
                for sub in (d, d / "binaries"):
                    q = sub / name
                    if q.exists():
                        return str(q)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def assemble(
        self,
        segment_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        target_fps: Optional[float] = None,
        target_size: Optional[tuple] = None,  # (width, height)
        transition_seconds: float = 0.0,
        requested_duration: Optional[float] = None,
        cleanup_intermediates: bool = False,
        workdir: Optional[Union[str, Path]] = None,
    ) -> AssemblyResult:
        """Concatenate ordered segment clips into one final video.

        Args:
            segment_paths: Ordered existing clip files.
            output_path: Destination file.
            target_fps / target_size: Normalize everything to these when set
                (forces re-encode if any clip differs).
            transition_seconds: If > 0, apply xfade crossfades of this length
                at each join (re-encode required; shortens total duration).
            requested_duration: For honest deviation reporting only.
            cleanup_intermediates: Opt-in deletion of source clips AFTER a
                verified successful assembly. Never deletes on failure.
        """
        paths = [Path(p) for p in segment_paths]
        if not paths:
            raise AssemblyError("No segments to assemble")
        infos = [self.probe(p) for p in paths]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        notes: List[str] = []

        # Decide compatibility.
        first = infos[0]
        compatible = all(
            i.width == first.width and i.height == first.height
            and abs(i.fps - first.fps) < 0.01 and i.codec == first.codec
            and i.pix_fmt == first.pix_fmt
            for i in infos
        )
        if target_size and any(
                (i.width, i.height) != tuple(target_size) for i in infos):
            compatible = False
            notes.append(
                f"Segments are {first.width}x{first.height}, requested "
                f"{target_size[0]}x{target_size[1]} -> re-encoding/scaling."
            )
        if target_fps and abs(first.fps - float(target_fps)) > 0.01:
            compatible = False
            notes.append(
                f"Segments are {first.fps:.2f} fps, requested {target_fps} -> re-timing."
            )
        if transition_seconds > 0:
            compatible = False  # xfade requires filtering

        if len(paths) == 1 and compatible and transition_seconds == 0:
            shutil.copyfile(paths[0], out)
            method = "stream_copy"
        elif compatible and transition_seconds == 0:
            try:
                method = self._concat_stream_copy(paths, out, workdir)
            except AssemblyError as exc:
                notes.append(f"Stream-copy failed ({exc}); falling back to re-encode.")
                method = self._concat_reencode(paths, out, infos, target_fps,
                                               target_size, 0.0)
        else:
            if transition_seconds > 0:
                method = self._concat_reencode(paths, out, infos, target_fps,
                                               target_size, transition_seconds)
                method = "crossfade"
                shrink = transition_seconds * (len(paths) - 1)
                notes.append(
                    f"Crossfades shorten total duration by {shrink:.2f}s "
                    f"({len(paths)-1} joins x {transition_seconds}s)."
                )
            else:
                method = self._concat_reencode(paths, out, infos, target_fps,
                                               target_size, 0.0)
                method = "reencode"

        result_info = self.probe(out)
        actual = result_info.duration or sum(i.duration for i in infos)
        deviation = (
            actual - float(requested_duration) if requested_duration else None
        )
        if deviation is not None and abs(deviation) > 0.5:
            notes.append(
                f"Final duration {actual:.2f}s deviates {deviation:+.2f}s from "
                f"requested {requested_duration:.2f}s (model frame-count snapping)."
            )
        res = AssemblyResult(
            output_path=str(out), clips_used=[str(p) for p in paths],
            method=method, actual_duration=actual,
            requested_duration=float(requested_duration) if requested_duration else None,
            deviation=deviation, notes=notes,
        )
        if cleanup_intermediates:
            for p in paths:
                try:
                    p.unlink()
                except OSError:
                    pass
            notes.append("Intermediate clips deleted (opt-in cleanup).")
        return res

    # ------------------------------------------------------------------
    def _concat_stream_copy(
        self, paths: List[Path], out: Path, workdir: Optional[Union[str, Path]]
    ) -> str:
        wd = Path(workdir) if workdir else out.parent
        wd.mkdir(parents=True, exist_ok=True)
        list_file = wd / f".concat-{abs(hash(tuple(str(p) for p in paths)))}.txt"
        lines = ["ffprobe_dir_placeholder"]  # will be replaced below
        with open(list_file, "w", encoding="utf-8") as f:
            for p in paths:
                safe = str(p.resolve()).replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
        try:
            self._run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                       "-f", "concat", "-safe", "0", "-i", str(list_file),
                       "-c", "copy", str(out)])
        finally:
            try:
                list_file.unlink()
            except OSError:
                pass
        if not out.exists() or out.stat().st_size == 0:
            raise AssemblyError("Stream-copy produced no output")
        return "stream_copy"

    def _concat_reencode(
        self,
        paths: List[Path],
        out: Path,
        infos: List[ClipInfo],
        target_fps: Optional[float],
        target_size: Optional[tuple],
        transition_seconds: float,
    ) -> str:
        w, h = (target_size if target_size else (infos[0].width, infos[0].height))
        fps = float(target_fps) if target_fps else max(
            1.0, round(infos[0].fps) if infos[0].fps else 8.0
        )
        inputs: List[str] = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        for p in paths:
            inputs += ["-i", str(p)]
        filt_parts = []
        labels = []
        for idx in range(len(paths)):
            lbl = f"v{idx}"
            filt_parts.append(
                f"[{idx}:v]fps={fps},scale={w}:{h},setsar=1,format=yuv420p[{lbl}]"
            )
            labels.append(lbl)
        if transition_seconds > 0:
            # Chain xfades; offsets computed from cumulative durations.
            durs = []
            for i in infos:
                d = i.duration or (i.frames / i.fps if i.fps else 0)
                durs.append(max(d, transition_seconds + 0.04))
            chain = f"[{labels[0]}][{labels[1]}]"
            acc = durs[0] - transition_seconds
            prev_out = "x0"
            filt_parts.append(
                f"[{labels[0]}][{labels[1]}]xfade=transition=fade:"
                f"duration={transition_seconds}:offset={acc:.4f}[{prev_out}]"
            )
            for k in range(2, len(labels)):
                acc += durs[k] - transition_seconds
                nxt = f"x{k-1}"
                filt_parts.append(
                    f"[{prev_out}][{labels[k]}]xfade=transition=fade:"
                    f"duration={transition_seconds}:offset={acc:.4f}[{nxt}]"
                )
                prev_out = nxt
            final_label = prev_out
        else:
            final_label = "cat"
            filt_parts.append(
                "".join(f"[{l}]" for l in labels)
                + f"concat=n={len(labels)}:v=1:a=0[{final_label}]"
            )
        graph = ";".join(filt_parts)
        cmd = inputs + [
            "-filter_complex", graph, "-map", f"[{final_label}]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(out),
        ]
        self._run(cmd, timeout=7200)
        if not out.exists() or out.stat().st_size == 0:
            raise AssemblyError("Re-encode produced no output")
        return "reencode"
