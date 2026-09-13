"""Configuration loading for voicehub."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    root: Path
    web: dict[str, Any]
    game: dict[str, Any]
    stt: dict[str, Any]
    llm: dict[str, Any]
    tts: dict[str, Any]
    policy: dict[str, Any]
    voice: dict[str, Any]
    personas: dict[str, Any] = field(default_factory=dict)

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def exchange_dir(self) -> Path:
        return self.path(self.game["homepath"]) / "voicehub"

    @property
    def rcon_password(self) -> str:
        f = self.game.get("rcon_password_file")
        if f:
            try:
                return self.path(f).read_text().strip()
            except OSError:
                return ""
        return self.game.get("rcon_password", "")


def load(root: Path | None = None) -> Config:
    root = (root or Path(__file__).resolve().parent.parent).resolve()
    with open(root / "config.toml", "rb") as f:
        raw = tomllib.load(f)
    from .personas import load_with_fallback
    personas = load_with_fallback(root / "personas.toml")
    return Config(
        root=root,
        web=raw.get("web", {}),
        game=raw.get("game", {}),
        stt=raw.get("stt", {}),
        llm=raw.get("llm", {}),
        tts=raw.get("tts", {}),
        policy=raw.get("policy", {}),
        voice=raw.get("voice", {}),
        personas=personas,
    )
