"""Tests for durable job endpoints on the FastAPI server.

Key regression guarantee: POST /jobs must ONLY persist a manifest. It must
never call _get_shared_adapter(), never invoke load_model(), and never touch
the network to download weights. Model initialization/generation happens in
the worker when the job actually runs.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from yukkavideo.api import server as server_mod
from yukkavideo.api.server import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setenv("YUKKA_JOBS_DIR", str(jobs_dir))
    app = create_app()
    # Never load models during API tests (legacy /generate path too).
    monkeypatch.setattr(server_mod, "_pipeline", object())
    return TestClient(app)


@pytest.fixture()
def no_adapter_load(monkeypatch):
    """Fail loudly if anything tries to construct/load real model adapters."""
    calls = {"shared": 0, "load": 0, "hf_download": 0}

    def _boom(self, *a, **k):
        calls["load"] += 1
        raise AssertionError("load_model() must not be called by POST /jobs")

    from yukkavideo.models.base import BaseModelAdapter
    monkeypatch.setattr(BaseModelAdapter, "load_model", _boom)

    def _no_shared(*a, **k):
        calls["shared"] += 1
        raise AssertionError("_get_shared_adapter must not be called")

    monkeypatch.setattr(server_mod, "_get_shared_adapter", _no_shared)

    import huggingface_hub
    real_dl = getattr(huggingface_hub, "hf_hub_download", None)
    if real_dl is not None:
        def _no_dl(*a, **k):
            calls["hf_download"] += 1
            raise AssertionError("network weight download attempted!")
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_dl)
    return calls


class TestJobCreationNoModelLoad:
    def test_post_jobs_persists_without_loading(self, client, no_adapter_load):
        resp = client.post("/jobs", json={
            "prompt": "a robot exploring ruins",
            "duration_seconds": 8.0,
            "fps": 8,
            "height": 480,
            "width": 720,
            "model": "THUDM/CogVideoX-2b",
            "segment_seconds": 5.0,
            "num_inference_steps": 8,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "queued"
        assert body["segments"] >= 2
        # Regression assertions: nothing was loaded or downloaded.
        assert no_adapter_load["shared"] == 0
        assert no_adapter_load["load"] == 0
        assert no_adapter_load["hf_download"] == 0
        # Manifest really exists on disk with segments pending (via GET).
        detail = client.get(f"/jobs/{body['job_id']}")
        assert detail.status_code == 200
        m = detail.json()
        assert m["status"] == "queued"
        assert all(s["status"] == "pending" for s in m["segments"])

    def test_invalid_model_rejected_422(self, client, no_adapter_load):
        resp = client.post("/jobs", json={
            "prompt": "x", "model": "not-a-real/video-model"})
        assert resp.status_code == 422

    def test_bad_dimensions_rejected_422(self, client, no_adapter_load):
        resp = client.post("/jobs", json={
            "prompt": "x", "model": "THUDM/CogVideoX-2b",
            "height": 479, "width": 721})
        assert resp.status_code == 422
        assert no_adapter_load["load"] == 0


class TestJobLifecycleEndpoints:
    def _make_job(self, client):
        resp = client.post("/jobs", json={
            "prompt": "waves at sunset", "duration_seconds": 6.0,
            "fps": 8, "height": 64, "width": 96,
            "model": "THUDM/CogVideoX-2b", "segment_seconds": 3.0})
        assert resp.status_code == 200
        return resp.json()["job_id"]

    def test_list_and_status(self, client):
        jid = self._make_job(client)
        listing = client.get("/jobs").json()
        assert any(j["job_id"] == jid for j in listing)
        st = client.get(f"/jobs/{jid}").json()
        assert st["status"] == "queued"

    def test_pause_cancel_signals(self, client):
        jid = self._make_job(client)
        r = client.post(f"/jobs/{jid}/pause")
        assert r.status_code == 200 and r.json()["pause_requested"]
        r = client.post(f"/jobs/{jid}/cancel")
        assert r.status_code == 200 and r.json()["cancel_requested"]
        # signal files exist next to the manifest
        jobs_dir = Path("outputs/jobs")  # env default; tmp dir used instead:
        # find actual dir from store fixture env
        import os
        d = Path(os.environ["YUKKA_JOBS_DIR"]) / jid
        assert (d / "cancel.signal").exists()

    def test_unknown_job_404(self, client):
        assert client.get("/jobs/nope").status_code == 404
        assert client.post("/jobs/nope/pause").status_code == 404
        assert client.post("/jobs/nope/cancel").status_code == 404
        assert client.get("/jobs/nope/video").status_code == 404

    def test_video_not_ready_409(self, client):
        jid = self._make_job(client)
        assert client.get(f"/jobs/{jid}/video").status_code == 409
