"""Map bot names to personas and voices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Persona:
    name: str
    voiced: bool
    voice: str          # piper voice
    speaker: str        # qwen3-tts preset speaker
    style: str          # qwen3-tts voice description (age, attitude), prepended to the emotion instruct
    persona: str


class Personas:
    def __init__(self, raw: dict[str, Any], default_voice: str, default_speaker: str = "Ryan"):
        d = raw.get("default", {})
        self.default = Persona("default", bool(d.get("voiced", True)), d.get("voice", default_voice),
                               d.get("speaker", default_speaker), d.get("style", ""),
                               d.get("persona", "a seasoned soldier with a dry sense of humour"))
        self.bots: dict[str, Persona] = {}
        for name, v in (raw.get("bots") or {}).items():
            self.bots[name.lower()] = Persona(name, bool(v.get("voiced", True)), v.get("voice", self.default.voice),
                                              v.get("speaker", self.default.speaker), v.get("style", self.default.style),
                                              v.get("persona", self.default.persona))

    def for_name(self, clean_name: str) -> Persona:
        p = self.bots.get(clean_name.lower())
        if p:
            return p
        return Persona(clean_name, self.default.voiced, self.default.voice, self.default.speaker,
                       self.default.style, self.default.persona)

    def voices_in_use(self) -> list[str]:
        return sorted({self.default.voice, *(p.voice for p in self.bots.values())})
