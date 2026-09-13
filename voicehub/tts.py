"""Text to speech with Piper. Returns WAV bytes.

Piper has no emotion control of its own, so an emotion tag chosen by the LLM is mapped
to what Piper does expose: speaking rate (length_scale), pitch/energy variance
(noise_scale), phoneme-duration variance (noise_w_scale), plus volume and a little
dynamics shaping applied to the samples afterwards.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("voicehub.tts")

# emotion -> (rate multiplier, noise_scale, noise_w_scale, gain). Rate >1 is faster.
# Piper defaults are noise_scale 0.667, noise_w 0.8.
EMOTIONS: dict[str, tuple[float, float, float, float]] = {
    "neutral":   (1.00, 0.667, 0.80, 1.00),
    "happy":     (1.06, 0.80, 0.90, 1.05),
    "excited":   (1.14, 0.95, 1.00, 1.15),
    "angry":     (1.10, 0.90, 0.70, 1.25),
    "shout":     (1.05, 0.95, 0.75, 1.40),
    "smug":      (0.94, 0.60, 0.95, 1.00),
    "sarcastic": (0.92, 0.55, 1.05, 1.00),
    "sad":       (0.86, 0.50, 0.95, 0.85),
    "tired":     (0.84, 0.45, 1.00, 0.85),
    "scared":    (1.12, 1.00, 1.10, 0.95),
    "whisper":   (0.95, 0.40, 0.90, 0.55),
    "confused":  (0.92, 0.75, 1.10, 0.95),
    "grim":      (0.88, 0.55, 0.85, 1.00),
}


def parse_emotion(text: str) -> tuple[str, str]:
    """Split a leading [emotion] tag off an LLM line. Unknown tags are dropped as neutral."""
    m = re.match(r"^\s*[\[\(<{]\s*([A-Za-z]+)\s*[\]\)>}]\s*[:\-]?\s*(.*)$", text, re.S)
    if not m:
        return "neutral", text.strip()
    tag = m.group(1).lower()
    return (tag if tag in EMOTIONS else "neutral"), m.group(2).strip()


def _shape(samples: np.ndarray, emotion: str, gain: float) -> np.ndarray:
    x = samples.astype(np.float32) / 32768.0
    if emotion in ("angry", "shout", "excited"):
        # mild compression: louder consonants, more "projected" voice
        x = np.tanh(x * (1.0 + 0.8 * (gain - 1.0)) * 1.6) / np.tanh(1.6)
    x = x * gain
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.98:
        x = x / peak * 0.98
    return (x * 32767.0).astype(np.int16)


class TTS:
    def __init__(self, voices_dir: Path, default_voice: str, opts: dict[str, Any] | None = None):
        self.voices_dir = voices_dir
        self.default_voice = default_voice
        self.opts = opts or {}
        self._voices: dict[str, object] = {}
        self.device_desc = "piper (cpu)"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.voices_dir.glob("*.onnx"))

    def _voice(self, name: str):
        from piper.voice import PiperVoice

        if name not in self._voices:
            path = self.voices_dir / f"{name}.onnx"
            if not path.exists():
                log.warning("voice %s missing, using %s", name, self.default_voice)
                name = self.default_voice
                path = self.voices_dir / f"{name}.onnx"
            if name not in self._voices:
                self._voices[name] = PiperVoice.load(str(path))
        return self._voices[name]

    def _synth(self, text: str, voice: str, emotion: str, speed_mul: float = 1.0) -> bytes:
        from piper.config import SynthesisConfig

        v = self._voice(voice)
        rate, noise, noise_w, gain = EMOTIONS.get(emotion, EMOTIONS["neutral"])
        speed = float(self.opts.get("speed", 1.15)) * rate * max(speed_mul, 0.1)
        if not self.opts.get("emotion", True):
            noise, noise_w, gain = EMOTIONS["neutral"][1:]
        cfg = SynthesisConfig(length_scale=1.0 / max(speed, 0.3), noise_scale=noise, noise_w_scale=noise_w)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            v.synthesize_wav(text, w, syn_config=cfg)
        if not self.opts.get("emotion", True) or (gain == 1.0 and emotion not in ("angry", "shout", "excited")):
            return buf.getvalue()
        buf.seek(0)
        with wave.open(buf, "rb") as r:
            params = r.getparams()
            pcm = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(_shape(pcm, emotion, gain).tobytes())
        return out.getvalue()

    async def synthesize(self, text: str, voice: str | None = None, emotion: str = "neutral",
                         style: str | None = None, speed: float = 1.0) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, self._synth, text, voice or self.default_voice, emotion, speed)

    async def preload(self, names: list[str]) -> None:
        for n in names:
            try:
                await asyncio.get_running_loop().run_in_executor(self._pool, self._voice, n)
            except Exception as e:  # noqa: BLE001
                log.warning("voice %s failed to load: %s", n, e)
