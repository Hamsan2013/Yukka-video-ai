"""
Scene planning and visual-continuity context for long videos.

Structured schema:
    ScenePlan  -> global style, characters, location, ordered scenes
    Scene      -> per-segment description + prompt fragment

A deterministic fallback planner is always available; an optional LLM can
enrich the plan but is never required (no paid API dependency).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Scene:
    index: int
    description: str
    prompt_fragment: str = ""
    start_time_seconds: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "prompt_fragment": self.prompt_fragment,
            "start_time_seconds": self.start_time_seconds,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scene":
        return cls(
            index=int(d["index"]),
            description=d.get("description", ""),
            prompt_fragment=d.get("prompt_fragment", ""),
            start_time_seconds=float(d.get("start_time_seconds", 0.0)),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
        )


@dataclass
class ScenePlan:
    original_prompt: str
    style: str = ""
    characters: List[str] = field(default_factory=list)
    location: str = ""
    scenes: List[Scene] = field(default_factory=list)
    reference_images: List[str] = field(default_factory=list)

    def segment_prompt(self, scene_index: int) -> str:
        """Prompt for one segment: global style + continuity + scene text."""
        parts: List[str] = []
        if self.style:
            parts.append(f"Style: {self.style}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.characters:
            parts.append("Characters: " + ", ".join(self.characters))
        if 0 <= scene_index < len(self.scenes):
            scene = self.scenes[scene_index]
            frag = scene.prompt_fragment or scene.description
            parts.append(f"Scene {scene_index + 1}: {frag}")
        else:
            parts.append(self.original_prompt)
        parts.append("Continue smoothly from the previous shot.")
        return ". ".join(p.rstrip(".") for p in parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "style": self.style,
            "characters": self.characters,
            "location": self.location,
            "reference_images": self.reference_images,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenePlan":
        return cls(
            original_prompt=d.get("original_prompt", ""),
            style=d.get("style", ""),
            characters=list(d.get("characters", [])),
            location=d.get("location", ""),
            scenes=[Scene.from_dict(s) for s in d.get("scenes", [])],
            reference_images=list(d.get("reference_images", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ScenePlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _split_clauses(prompt: str) -> List[str]:
    """Split a prompt into sentence-ish clauses deterministically."""
    clauses = re.split(r"(?<=[.;!?])\s+|\s*\n+\s*|,\s*(?=\w+\s+(?:and|then|while)\b)", prompt)
    return [c.strip() for c in clauses if c and c.strip()]


def deterministic_plan(
    prompt: str,
    num_scenes: int,
    total_duration: float,
    style: str = "",
    characters: Optional[List[str]] = None,
    location: str = "",
) -> ScenePlan:
    """Fallback planner requiring no LLM.

    Distributes the prompt's clauses across ``num_scenes`` scenes; when
    there are fewer clauses than scenes, later scenes reuse the full prompt
    with an explicit 'continuation' note so motion keeps progressing.
    """
    num_scenes = max(1, int(num_scenes))
    clauses = _split_clauses(prompt) or [prompt]
    per_scene_dur = total_duration / num_scenes

    scenes: List[Scene] = []
    for i in range(num_scenes):
        if len(clauses) >= num_scenes:
            chunk = clauses[i]
        else:
            # Cycle clauses slowly, marking progression.
            chunk = clauses[i % len(clauses)]
            if i >= len(clauses):
                chunk = f"{chunk} — continuing the action"
        scenes.append(
            Scene(
                index=i,
                description=chunk,
                prompt_fragment=chunk,
                start_time_seconds=round(i * per_scene_dur, 4),
                duration_seconds=round(per_scene_dur, 4),
            )
        )

    return ScenePlan(
        original_prompt=prompt,
        style=style,
        characters=characters or [],
        location=location,
        scenes=scenes,
    )


def llm_plan(
    prompt: str,
    num_scenes: int,
    total_duration: float,
    style: str = "",
    characters: Optional[List[str]] = None,
    location: str = "",
) -> Optional[ScenePlan]:
    """Optional LLM-based planner.

    Uses a locally served OpenAI-compatible endpoint only if the user has
    configured YUKKA_LLM_API_BASE / YUKKA_LLM_MODEL / YUKKA_LLM_API_KEY
    environment variables. Returns None on any failure so callers fall back
    to the deterministic planner. Never mandatory, never paid-by-default.
    """
    import os

    api_base = os.environ.get("YUKKA_LLM_API_BASE")
    model = os.environ.get("YUKKA_LLM_MODEL")
    if not api_base or not model:
        return None

    try:  # pragma: no cover - network dependent
        import requests

        schema_hint = (
            '{"scenes":[{"index":int,"description":str,"prompt_fragment":str}]}'
        )
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('YUKKA_LLM_API_KEY', '')}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Split this video idea into exactly {num_scenes} sequential "
                            f"scenes of ~{total_duration / max(1, num_scenes):.1f}s each. "
                            f"Return JSON only, shaped like {schema_hint}. Idea: {prompt}"
                        ),
                    }
                ],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
        scenes = []
        per = total_duration / max(1, num_scenes)
        for i, s in enumerate(data.get("scenes", [])[:num_scenes]):
            scenes.append(
                Scene(
                    index=i,
                    description=s.get("description", ""),
                    prompt_fragment=s.get("prompt_fragment", s.get("description", "")),
                    start_time_seconds=round(i * per, 4),
                    duration_seconds=round(per, 4),
                )
            )
        if not scenes:
            return None
        while len(scenes) < num_scenes:  # top up if LLM returned fewer
            last = scenes[-1]
            scenes.append(
                Scene(
                    index=len(scenes),
                    description=f"{last.description} — continued",
                    prompt_fragment=f"{last.prompt_fragment} — continued",
                    start_time_seconds=round(len(scenes) * per, 4),
                    duration_seconds=round(per, 4),
                )
            )
        return ScenePlan(
            original_prompt=prompt,
            style=style,
            characters=characters or [],
            location=location,
            scenes=scenes,
        )
    except Exception:
        return None


def create_scene_plan(
    prompt: str,
    num_scenes: int,
    total_duration: float,
    style: str = "",
    characters: Optional[List[str]] = None,
    location: str = "",
    use_llm: bool = False,
) -> ScenePlan:
    """Public entry: LLM if explicitly enabled & configured, else deterministic."""
    if use_llm:
        plan = llm_plan(prompt, num_scenes, total_duration, style, characters, location)
        if plan is not None:
            return plan
    return deterministic_plan(prompt, num_scenes, total_duration, style, characters, location)
