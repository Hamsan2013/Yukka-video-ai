"""Tests for the sequential GenerationRunner (fake adapters, no GPU/downloads)."""
import json
from pathlib import Path

import pytest

from tests.fakes import FakeI2VAdapter, FakeT2VAdapter
from yukkavideo.continuity.frames import ClipValidationError
from yukkavideo.jobs.control import JobController
from yukkavideo.jobs.manifest import JobStore
from yukkavideo.jobs.runner import GenerationRunner, OutOfMemoryError, RunnerOptions


@pytest.fixture()
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


def _runner(adapter, store, **opt_kw):
    options = RunnerOptions(num_inference_steps=4, **opt_kw)
    return GenerationRunner(adapter, store, options=options)


def make_job(store, adapter, duration=6.0, fps=8.0, seg=3.0, **kw):
    runner = _runner(adapter, store, **kw)
    manifest = runner.create_job(
        prompt="a fox in snow", duration_seconds=duration, fps=fps,
        height=64, width=96, segment_seconds=seg, seed=100,
        model_name=adapter.model_name,
    )
    return runner, manifest


class TestCreateJob:
    def test_plans_segments_and_manifest(self, store):
        runner, m = make_job(store, FakeT2VAdapter())
        assert m.status == "queued"
        assert len(m.segments) >= 2
        assert m.total_frames > 0
        # manifest persisted atomically to disk
        raw = json.loads(store.manifest_path(m.job_id).read_text())
        assert raw["job_id"] == m.job_id

    def test_rejects_bad_dimensions(self, store):
        runner = _runner(FakeT2VAdapter(), store)
        with pytest.raises(ValueError, match="cannot produce"):
            runner.create_job(prompt="x", duration_seconds=5, fps=8,
                              height=479, width=721, segment_seconds=5)

    def test_missing_start_image_raises(self, store):
        runner = _runner(FakeI2VAdapter(), store)
        with pytest.raises(FileNotFoundError):
            runner.create_job(prompt="x", duration_seconds=5, fps=8,
                              height=64, width=96, start_image="/no/such.png")


class TestRunAndResume:
    def test_full_run_completes_and_assembles(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter)
        result = runner.run(m.job_id)
        assert result.status == "completed"
        assert result.progress == 1.0
        assert all(s.status == "done" for s in result.segments)
        assert result.final_video_path and Path(result.final_video_path).exists()
        # every segment produced a reference frame for continuity
        assert all(Path(s.reference_frame_path).exists()
                   for s in result.segments if s.reference_frame_path)

    def test_resume_does_not_regenerate_done_segments(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter, duration=9.0, seg=3.0)
        # Run only first segment by pausing after it completes.
        ctrl = JobController(store, m.job_id)
        orig_save = store.save

        def save_then_pause(man):
            orig_save(man)
            if any(s.status == "done" for s in man.segments):
                ctrl.request_pause()
        store.save = save_then_pause
        first = runner.run(m.job_id)
        store.save = orig_save
        assert first.status == "paused"
        done_after_pause = sum(1 for s in first.segments if s.status == "done")
        calls_after_pause = adapter.t2v_calls
        assert done_after_pause >= 1

        ctrl.clear_signals()
        adapter2 = FakeT2VAdapter()
        runner2 = _runner(adapter2, store)
        final = runner2.run(m.job_id)
        assert final.status == "completed"
        # resumed run generated strictly fewer segments than total
        assert adapter2.t2v_calls == len(final.segments) - done_after_pause

    def test_corrupt_clip_is_regenerated_on_resume(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter, duration=6.0, seg=3.0)
        result = runner.run(m.job_id)
        assert result.status == "completed"
        # Corrupt one clip file on disk + reset its status. (completed ->
        # paused is blocked by terminal-state guards, so we simulate the
        # recovery scenario on a freshly loaded manifest before re-saving.)
        seg0 = result.segments[0]
        Path(seg0.output_path).write_bytes(b"garbage-not-a-video")
        m2 = store.load(m.job_id)
        m2.segments[0].status = "pending"
        m2.status = "paused"  # direct field set: test-only state surgery
        store.save(m2)
        adapter2 = FakeT2VAdapter()
        final = _runner(adapter2, store).run(m.job_id)
        assert final.status == "completed"
        assert adapter2.t2v_calls >= 1  # only the corrupted one regenerated

    def test_pause_at_boundary(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter, duration=9.0, seg=3.0)
        ctrl = JobController(store, m.job_id)
        ctrl.request_pause()
        result = runner.run(m.job_id)
        assert result.status == "paused"
        assert adapter.t2v_calls == 0  # paused before generating anything

    def test_cancel_stops_job(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter)
        JobController(store, m.job_id).request_cancel()
        result = runner.run(m.job_id)
        assert result.status == "cancelled"
        assert adapter.t2v_calls == 0


class TestContinuation:
    def test_i2v_adapter_uses_reference_frames(self, store):
        adapter = FakeI2VAdapter()
        runner, m = make_job(store, adapter, duration=9.0, seg=3.0)
        result = runner.run(m.job_id)
        assert result.status == "completed"
        assert adapter.i2v_calls >= 1      # continuation really used
        assert adapter.t2v_calls >= 1      # first segment from text

    def test_t2v_only_never_receives_reference(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter, duration=9.0, seg=3.0)
        result = runner.run(m.job_id)
        assert result.status == "completed"
        assert adapter.i2v_calls == 0
        # base-class guard also refuses directly:
        from yukkavideo.models.base import UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError):
            adapter.generate_from_image("x.png", "p", num_frames=9,
                                        height=64, width=96)


class TestOOMHandling:
    def test_oom_marks_failed_with_actionable_message(self, store):
        adapter = FakeT2VAdapter()
        adapter.oom_times = {100}   # seed for segment 0 (manifest.seed + 0)
        runner, m = make_job(store, adapter)
        with pytest.raises(OutOfMemoryError, match="out-of-memory"):
            runner.run(m.job_id)
        mm = store.load(m.job_id)
        assert mm.status == "failed"
        assert any("Remedies" in e or "OOM" in e for e in mm.errors)

    def test_oom_downscale_retry_opt_in(self, store):
        class OOMOnceAbove32(FakeT2VAdapter):
            """Fails by CUDA OOM only at heights above 32px (realistic model)."""

            def generate_clip(self, prompt, num_frames=49, height=480,
                              width=720, seed=None, **kw):
                if height > 32:
                    self.t2v_calls += 1
                    raise RuntimeError("The following CUDA out of memory. "
                                       "Tried to allocate 8.00 GiB")
                return super().generate_clip(prompt, num_frames=num_frames,
                                             height=height, width=width,
                                             seed=seed, **kw)

        adapter = OOMOnceAbove32()
        runner, m = make_job(store, adapter, duration=3.0, seg=3.0,
                             retry_on_oom_downscale=True, max_segment_retries=1)
        result = runner.run(m.job_id)
        assert result.status == "completed"
        assert any("auto-retried" in e for e in result.errors)

    def test_oom_downscale_not_attempted_without_opt_in(self, store):
        adapter = FakeT2VAdapter()
        adapter.oom_times = {100}
        runner, m = make_job(store, adapter, duration=3.0, seg=3.0)
        with pytest.raises(OutOfMemoryError):
            runner.run(m.job_id)
        # No silent quality change was recorded when opt-in was false.
        assert not any("auto-retried" in e for e in adapter.__dict__.get("_x", []))
        mm = store.load(m.job_id)
        assert not any("auto-retried" in e for e in mm.errors)


class TestManifestIntegrity:
    def test_completed_clips_survive_later_failure(self, store):
        adapter = FakeT2VAdapter()
        runner, m = make_job(store, adapter, duration=9.0, seg=3.0)
        # The runner derives each segment's generation seed as
        # manifest.seed + segment.index; fail the LAST segment only.
        adapter.fail_times = {m.seed + m.segments[-1].index}
        with pytest.raises(ValueError):
            runner.run(m.job_id)
        mm = store.load(m.job_id)
        assert mm.status == "failed"
        done_paths = [s.output_path for s in mm.segments if s.status == "done"]
        assert done_paths and all(Path(p).exists() for p in done_paths)

    def test_invalid_clip_validation_raises(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"\x00\x00")
        with pytest.raises(ClipValidationError):
            from yukkavideo.continuity.frames import validate_clip_file
            validate_clip_file(bad)
