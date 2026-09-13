"""Runtime settings editable from the browser (/settings): per-bot voices and personas,
plus the hot-reloadable [tts], [policy] and [voice] knobs of config.toml.

Changes are applied in memory at once and written back to personas.toml / config.toml so
they survive a restart and stay editable by hand."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .personas import EDITABLE as PERSONA_FIELDS, Personas
from .toml_edit import update_file

log = logging.getLogger("voicehub.settings")

# Which config.toml keys the page may change, with their type and (optional) range.
GLOBAL_FIELDS: dict[str, dict[str, tuple[type, Any, Any]]] = {
    "tts": {
        "speed": (float, 0.5, 2.0),
        "emotion": (bool, None, None),
    },
    "policy": {
        "rating": (("family", "pg13", "r", "explicit"), None, None),
        "idle_chatter": (bool, None, None),
        "idle_chatter_interval_s": (float, 0, 3600),
        "idle_chatter_jitter": (float, 0, 0.95),
        "idle_chatter_min_quiet_s": (float, 0, 600),
        "global_min_gap_s": (float, 0, 120),
        "unprompted_cooldown_s": (float, 0, 600),
        "kill_banter_prob": (float, 0, 1),
        "revive_thanks_prob": (float, 0, 1),
        "greet_prob": (float, 0, 1),
        "mapstart_prob": (float, 0, 1),
        "open_mic_reply_prob": (float, 0, 1),
        "bot_reply_prob": (float, 0, 1),
        "bot_thread_max": (int, 0, 10),
        "chitchat_prob": (float, 0, 1),
        "objective_prob": (float, 0, 1),
        "max_queue": (int, 1, 20),
    },
    "voice": {
        "team_only": (bool, None, None),
        "spectators_hear_all": (bool, None, None),
        "relay_human_voice": (bool, None, None),
        "open_mic_default": (bool, None, None),
        "tts_streaming": (bool, None, None),
        "silence_ms": (int, 200, 5000),
        "max_utterance_s": (int, 2, 60),
        "ptt_timeout_s": (int, 1, 60),
    },
}


def _coerce(spec: tuple[type | tuple, Any, Any], value: Any) -> Any:
    typ, lo, hi = spec
    if isinstance(typ, tuple):           # enumeration
        v = str(value).strip().lower()
        if v not in typ:
            raise ValueError(f"must be one of {', '.join(typ)}")
        return v
    if typ is bool:
        return value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
    v = typ(value)
    if lo is not None and v < lo:
        v = typ(lo)
    if hi is not None and v > hi:
        v = typ(hi)
    return round(v, 3) if typ is float else v


class Settings:
    def __init__(self, root: Path, cfg_sections: dict[str, dict[str, Any]], personas: Personas):
        self.root = root
        self.sections = cfg_sections          # {"tts": cfg.tts, "policy": cfg.policy, "voice": cfg.voice}: live dicts
        self.personas = personas
        self.last_write = 0.0

    @property
    def personas_last_write(self) -> float:
        return self.personas.last_write

    # ------------------------------------------------------------- read
    def snapshot(self, state: Any, brain: Any) -> dict[str, Any]:
        """Everything the page needs: connected bots merged with configured personas."""
        online = {b.clean.lower(): b for b in state.bots()}
        names = list(online)
        for n in self.personas.configured_names():
            if n.lower() not in online:
                names.append(n.lower())
        tts = brain.tts
        speakers = tts.available() if brain.engine == "qwen" else []
        canon = {s.lower(): s for s in speakers}
        bots = []
        for key in names:
            pl = online.get(key)
            p = self.personas.for_name(pl.clean if pl else key)
            bots.append({
                **p.to_dict(),
                "speaker": canon.get(p.speaker.lower(), p.speaker),
                "name": pl.clean if pl else self.personas.bots[key].name,
                "online": pl is not None,
                "team": pl.team_name if pl else None,
                "class": pl.class_name if pl else None,
                "override": self.personas.has_override(key),
            })
        bots.sort(key=lambda b: (not b["online"], b["name"].lower()))
        default = self.personas.default.to_dict()
        default["speaker"] = canon.get(default["speaker"].lower(), default["speaker"])
        return {
            "engine": brain.engine,
            "speakers": speakers,
            "piper_voices": tts.available() if brain.engine == "piper" else _piper_voices(self.root, self.sections["tts"]),
            "emotions": _emotions(),
            "tts_ready": getattr(tts, "model", True) is not None,
            "map": state.map,
            "default": default,
            "bots": bots,
            "fields": {k: {"type": v[0].__name__, "min": v[1], "max": v[2]} for k, v in PERSONA_FIELDS.items()},
            "global": {sec: {k: self.sections[sec].get(k) for k in keys} for sec, keys in GLOBAL_FIELDS.items()},
            "global_fields": {sec: {k: {"type": (v[0].__name__ if not isinstance(v[0], tuple) else "enum"),
                                        "choices": list(v[0]) if isinstance(v[0], tuple) else None,
                                        "min": v[1], "max": v[2]} for k, v in keys.items()}
                              for sec, keys in GLOBAL_FIELDS.items()},
        }

    # ------------------------------------------------------------- write
    def update_bot(self, name: str | None, changes: dict[str, Any]) -> dict[str, Any]:
        p = self.personas.update(name, changes)
        return p.to_dict()

    def reset_bot(self, name: str) -> bool:
        return self.personas.remove(name)

    def update_global(self, changes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """changes = {section: {key: value}}. Validates, applies live, writes config.toml."""
        clean: dict[str, dict[str, Any]] = {}
        for sec, vals in changes.items():
            spec = GLOBAL_FIELDS.get(sec)
            if not spec or not isinstance(vals, dict):
                continue
            for k, v in vals.items():
                if k in spec and v is not None:
                    clean.setdefault(sec, {})[k] = _coerce(spec[k], v)
        for sec, vals in clean.items():
            self.sections[sec].update(vals)
        if clean:
            path = self.root / "config.toml"
            update_file(path, clean)
            self.last_write = path.stat().st_mtime
            log.info("config.toml updated from the settings page: %s", clean)
        return clean


def _piper_voices(root: Path, tts_opts: dict[str, Any]) -> list[str]:
    d = Path(tts_opts.get("voices_dir", "voices"))
    d = d if d.is_absolute() else root / d
    return sorted(p.stem for p in d.glob("*.onnx")) if d.exists() else []


def _emotions() -> list[str]:
    from .tts import EMOTIONS
    return list(EMOTIONS)
