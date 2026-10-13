"""Tests for FFmpeg-based final-video assembly.

Lightweight clips are synthesized with the local ffmpeg binary (tiny, fast).
FFmpeg-missing and corrupt-input error paths are exercised via monkeypatching
(no network, no GPU).
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from yukkavideo.assembly.assembler import (
    AssemblyError,
    ClipInfo,
    FFmpegNotFoundError,
    VideoAssembler,
    find_ffmpeg,
)


def _make_clip(path: Path, seconds: float = 0.5, fps: int = 8,
               size=(96, 64), color="red") -> None:
    """Create a tiny real mp4 using lavfi testsrc (fast, deterministic)."""
    w, h = size
    cmd = [
        find_ffmpeg(), "-y", "-f", "lavfi", "-i",
        f"testsrc=size={w}x{h}:rate={fps}:duration={seconds}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


@pytest.fixture(scope="module")
def assembler():
    return VideoAssembler()


class TestFindFfmpeg:
    def test_found(self):
        assert Path(find_ffmpeg()).exists() or find_ffmpeg() == "ffmpeg"

    def test_missing_raises_with_instructions(self, monkeypatch, tmp_path):
        monkeypatch.delenv("YUKKA_FFMPEG_PATH", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        # Break the imageio-ffmpeg fallback too.
        import sys
        fake = {"imageio_ffmpeg": None}
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        with pytest.raises(FFmpegNotFoundError, match="apt-get install ffmpeg"):
            find_ffmpeg()


class TestProbe:
    def test_probe_reads_metadata(self, assembler, tmp_path):
        clip = tmp_path / "a.mp4"
        _make_clip(clip)
        info = assembler.probe(clip)
        assert isinstance(info, ClipInfo)
        assert (info.width, info.height) == (96, 64)
        assert info.frames >= 3
        assert 0.0 < info.duration <= 1.0

    def test_probe_missing_file(self, assembler, tmp_path):
        with pytest.raises((AssemblyError, FileNotFoundError)):
            assembler.probe(tmp_path / "nope.mp4")

    def test_probe_corrupt_file(self, assembler, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"\x00garbage")
        with pytest.raises(AssemblyError):
            assembler.probe(bad)


class TestAssemble:
    def test_stream_copy_concat_and_duration(self, assembler, tmp_path):
        c1, c2 = tmp_path / "s0.mp4", tmp_path / "s1.mp4"
        _make_clip(c1); _make_clip(c2)
        out = tmp_path / "final.mp4"
        res = assembler.assemble([c1, c2], out, target_fps=8,
                                 requested_duration=1.0)
        assert out.exists() and out.stat().st_size > 0
        assert res.method in ("stream_copy", "reencode")
        # actual duration ~ sum of clips; deviation reported honestly
        assert abs(res.actual_duration - 1.0) < 0.35
        assert res.deviation is not None
        info = assembler.probe(out)
        assert info.frames >= 6

    def test_mismatched_sizes_force_reencode(self, assembler, tmp_path):
        c1 = tmp_path / "big.mp4"; c2 = tmp_path / "small.mp4"
        _make_clip(c1, size=(96, 64)); _make_clip(c2, size=(64, 48))
        out = tmp_path / "final.mp4"
        res = assembler.assemble([c1, c2], out, target_size=(96, 64))
        assert res.method != "stream_copy"
        assert any("re-encod" in n.lower() or "scal" in n.lower()
                   for n in res.notes)
        info = assembler.probe(out)
        assert (info.width, info.height) == (96, 64)

    def test_crossfade_shortens_duration_reported(self, assembler, tmp_path):
        c1, c2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
        _make_clip(c1, seconds=1.0); _make_clip(c2, seconds=1.0)
        out = tmp_path / "xfade.mp4"
        res = assembler.assemble([c1, c2], out, transition_seconds=0.25)
        assert res.method == "crossfade"
        # honest reporting: overlap subtracts ~0.25s from 2.0s total
        assert res.actual_duration < 2.0
        assert res.actual_duration > 1.4

    def test_no_segments_error(self, assembler, tmp_path):
        with pytest.raises(AssemblyError, match="No segments"):
            assembler.assemble([], tmp_path / "x.mp4")

    def test_missing_segment_error(self, assembler, tmp_path):
        good = tmp_path / "g.mp4"; _make_clip(good)
        with pytest.raises((AssemblyError, FileNotFoundError)):
            assembler.assemble([good, tmp_path / "gone.mp4"],
                               tmp_path / "out.mp4")

    def test_cleanup_is_opt_in_and_only_on_success(self, assembler, tmp_path):
        c1 = tmp_path / "k.mp4"; _make_clip(c1)
        out = tmp_path / "o.mp4"
        res = assembler.assemble([c1], out, cleanup_intermediates=False)
        assert c1.exists()  # default: never delete user's clips
        res2 = assembler.assemble([c1], tmp_path / "o2.mp4",
                                  cleanup_intermediates=True)
        assert not c1.exists()  # explicit opt-in deletes after success

    def test_ffmpeg_failure_surfaces_assembly_error(self, assembler):
        # A broken ffmpeg binary must raise AssemblyError (with stderr tail)
        # or an OS-level error — never silently "succeed".
        with pytest.raises((AssemblyError, OSError, FileNotFoundError)):
            assembler._run(["definitely-not-ffmpeg-xyz", "-version"])
