"""Map bot names to personas and voices. Editable at runtime (web settings page) and
persisted back to personas.toml."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

log = logging.getLogger("voicehub.personas")

HEADER = """# Bot personas. Keys are the clean Omni-bot names (without the [BOT] prefix).
# Any connected bot without an entry uses [default]. Set voiced = false on
# [default] to restrict speech to the named bots only.
#   voice   = Piper voice (engine = "piper")
#   speaker = Qwen3-TTS preset (engine = "qwen"): aiden, dylan, eric, ryan, uncle_fu (male); ono_anna, serena, sohee, vivian (female)
#   style   = Qwen3-TTS voice description, prepended to the emotion instruction
#   speed   = playback rate multiplier on top of [tts] speed (1.0 = unchanged)
#   volume  = gain applied to this bot's audio (1.0 = unchanged)
# This file is rewritten by the settings page (https://<hub>/settings); comments below are not kept.
"""

# Field -> (type, min, max) for values coming from the web page.
EDITABLE: dict[str, tuple[type, float | None, float | None]] = {
    "voiced": (bool, None, None),
    "voice": (str, None, None),
    "speaker": (str, None, None),
    "style": (str, None, None),
    "persona": (str, None, None),
    "speed": (float, 0.5, 2.0),
    "volume": (float, 0.2, 3.0),
}


@dataclass
class Persona:
    name: str
    voiced: bool
    voice: str          # piper voice
    speaker: str        # qwen3-tts preset speaker
    style: str          # qwen3-tts voice description (age, attitude), prepended to the emotion instruct
    persona: str
    speed: float = 1.0
    volume: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(key: str, value: Any) -> Any:
    typ, lo, hi = EDITABLE[key]
    if typ is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if typ is float:
        v = float(value)
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return round(v, 3)
    return str(value).strip()


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)


def dump_personas(raw: dict[str, Any]) -> str:
    out = [HEADER, "[default]"]
    for k, v in (raw.get("default") or {}).items():
        out.append(f"{k} = {_toml_value(v)}")
    for name, vals in (raw.get("bots") or {}).items():
        key = name if name.replace("_", "").replace("-", "").isalnum() else json.dumps(name)
        out += ["", f"[bots.{key}]"]
        for k, v in vals.items():
            out.append(f"{k} = {_toml_value(v)}")
    return "\n".join(out) + "\n"


class Personas:
    def __init__(self, raw: dict[str, Any], default_voice: str, default_speaker: str = "Ryan",
                 path: Path | None = None):
        self.path = path
        self._fallback_voice = default_voice
        self._fallback_speaker = default_speaker
        self._lock = threading.Lock()
        self.last_write = 0.0
        self.raw: dict[str, Any] = {}
        self.default: Persona
        self.bots: dict[str, Persona] = {}
        self.load(raw)

    # ------------------------------------------------------------- building
    def _build(self, name: str, v: dict[str, Any], base: Persona | None) -> Persona:
        b = base
        return Persona(
            name,
            bool(v.get("voiced", b.voiced if b else True)),
            str(v.get("voice", b.voice if b else self._fallback_voice)),
            str(v.get("speaker", b.speaker if b else self._fallback_speaker)),
            str(v.get("style", b.style if b else "")),
            str(v.get("persona", b.persona if b else "a seasoned soldier with a dry sense of humour")),
            float(v.get("speed", b.speed if b else 1.0)),
            float(v.get("volume", b.volume if b else 1.0)),
        )

    def load(self, raw: dict[str, Any]) -> None:
        """(Re)build from a personas.toml dict."""
        self.raw = {"default": dict(raw.get("default") or {}), "bots": {k: dict(v) for k, v in (raw.get("bots") or {}).items()}}
        self.default = self._build("default", self.raw["default"], None)
        self.bots = {name.lower(): self._build(name, v, self.default) for name, v in self.raw["bots"].items()}

    # ------------------------------------------------------------- queries
    def for_name(self, clean_name: str) -> Persona:
        p = self.bots.get(clean_name.lower())
        if p:
            return p
        d = self.default
        return Persona(clean_name, d.voiced, d.voice, d.speaker, d.style, d.persona, d.speed, d.volume)

    def has_override(self, clean_name: str) -> bool:
        return clean_name.lower() in self.bots

    def voices_in_use(self) -> list[str]:
        return sorted({self.default.voice, *(p.voice for p in self.bots.values())})

    def configured_names(self) -> list[str]:
        return [p.name for p in self.bots.values()]

    # ------------------------------------------------------------- editing
    def _raw_key(self, name: str) -> str | None:
        for k in self.raw["bots"]:
            if k.lower() == name.lower():
                return k
        return None

    def update(self, name: str | None, changes: dict[str, Any]) -> Persona:
        """Apply `changes` (only EDITABLE keys) to bot `name`, or to [default] when name is None,
        then persist. Unknown keys are ignored."""
        clean = {k: _coerce(k, v) for k, v in changes.items() if k in EDITABLE and v is not None}
        with self._lock:
            if name is None:
                self.raw["default"].update(clean)
            else:
                key = self._raw_key(name) or name.strip()
                if not key:
                    raise ValueError("empty bot name")
                self.raw["bots"].setdefault(key, {}).update(clean)
            self.load(self.raw)
            self._save()
        return self.default if name is None else self.for_name(name)

    def remove(self, name: str) -> bool:
        """Drop a bot's override so it falls back to [default]."""
        with self._lock:
            key = self._raw_key(name)
            if key is None:
                return False
            del self.raw["bots"][key]
            self.load(self.raw)
            self._save()
        return True

    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".toml.tmp")
        tmp.write_text(dump_personas(self.raw), encoding="utf-8")
        tmp.replace(self.path)
        self.last_write = self.path.stat().st_mtime
        log.info("personas.toml saved (%d bots)", len(self.raw["bots"]))
