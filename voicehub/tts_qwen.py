"""Text to speech with Qwen3-TTS (CustomVoice model): preset speakers plus a free-text
instruction that carries the voice style (age, attitude) and the emotion of the line.

Slow on this box (about 3 s of compute per second of audio, CPU-bound decode loop), so the
brain streams sentence by sentence. Output is 24 kHz WAV; `speed` is applied with ffmpeg.
"""
from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

log = logging.getLogger("voicehub.tts")

# Emotion tag (from the LLM) -> how to say it. Kept short: the instruct text is part of the prompt.
EMOTION_INSTRUCT = {
    "neutral": "Speak naturally.",
    "happy": "Sound happy and upbeat.",
    "excited": "Sound very excited, fast and loud, almost shouting with joy.",
    "angry": "Sound angry and aggressive, raised voice.",
    "shout": "Shout loudly, urgent.",
    "smug": "Sound smug and self-satisfied, a little drawn out.",
    "sarcastic": "Sound sarcastic and dry, deadpan.",
    "sad": "Sound sad and deflated, slower.",
    "tired": "Sound tired and bored, low energy.",
    "scared": "Sound scared and panicky, breathless.",
    "whisper": "Whisper quietly, as if hiding.",
    "confused": "Sound confused and hesitant.",
    "grim": "Sound grim and menacing, low and slow.",
}


class QwenTTS:
    def __init__(self, opts: dict[str, Any], default_speaker: str = "Ryan", default_style: str = ""):
        self.opts = opts
        self.default_voice = default_speaker
        self.default_style = default_style
        self.model = None
        self.device_desc = "not loaded"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    # --- same interface as TTS (piper) -------------------------------------------
    def available(self) -> list[str]:
        if self.model is None:
            return []
        try:
            return sorted(self.model.get_supported_speakers())
        except Exception:  # noqa: BLE001
            return []

    def _load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        name = self.opts.get("qwen_model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        device = self.opts.get("qwen_device", "cuda:0")
        dtype = getattr(torch, self.opts.get("qwen_dtype", "float16"))
        self.model = Qwen3TTSModel.from_pretrained(name, device_map=device, dtype=dtype, attn_implementation="sdpa")
        self.device_desc = f"{name.split('/')[-1]} on {device} ({self.opts.get('qwen_dtype', 'float16')})"
        if self.opts.get("qwen_fast", True):
            # CUDA-graph decode loop (see fast_qwen.py): ~4x faster than stock on this hardware
            from .fast_qwen import patch_model_full
            patch_model_full(self.model.model, max_new=int(self.opts.get("qwen_max_new_tokens", 400)) + 1)
            self.device_desc += " +cudagraph"
        # first call is slow (kernel selection, graph capture): do it now, not on the first bot line
        self.model.generate_custom_voice(text="Radio check, one two.", language="English", speaker=self.default_voice, instruct="")
        log.info("qwen tts loaded: %s", self.device_desc)

    async def preload(self, names: list[str]) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, self._load)

    def instruct_for(self, emotion: str, style: str | None) -> str:
        parts = [style or self.default_style, EMOTION_INSTRUCT.get(emotion, EMOTION_INSTRUCT["neutral"])]
        return " ".join(p.strip() for p in parts if p and p.strip())

    def _synth(self, text: str, speaker: str, emotion: str, style: str | None, speed_mul: float = 1.0) -> bytes:
        if self.model is None:
            raise RuntimeError("qwen tts not loaded")
        speakers = {s.lower() for s in self.available()}
        if speaker.lower() not in speakers:
            log.warning("speaker %s unknown, using %s", speaker, self.default_voice)
            speaker = self.default_voice
        wavs, sr = self.model.generate_custom_voice(
            text=text, language="English", speaker=speaker, instruct=self.instruct_for(emotion, style),
            max_new_tokens=int(self.opts.get("qwen_max_new_tokens", 400)),
        )
        pcm = np.clip(np.asarray(wavs[0], dtype=np.float32), -1, 1)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr))
            w.writeframes((pcm * 32767).astype(np.int16).tobytes())
        data = buf.getvalue()
        speed = float(self.opts.get("speed", 1.0)) * max(speed_mul, 0.1)
        if abs(speed - 1.0) > 0.01:
            data = _atempo(data, speed) or data
        return data

    async def synthesize(self, text: str, voice: str | None = None, emotion: str = "neutral",
                         style: str | None = None, speed: float = 1.0) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, self._synth, text, voice or self.default_voice, emotion, style, speed)


def _atempo(wav: bytes, speed: float) -> bytes | None:
    """Time-stretch without changing pitch (ffmpeg atempo)."""
    try:
        speed = min(max(speed, 0.5), 4.0)
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-filter:a", f"atempo={speed:.3f}",
                            "-f", "wav", "pipe:1"], input=wav, capture_output=True, timeout=20)
        return _rewrap(p.stdout) if p.returncode == 0 and p.stdout else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _rewrap(wav: bytes) -> bytes | None:
    """ffmpeg cannot seek on a pipe, so it leaves the RIFF/data sizes as 0xFFFFFFFF; rewrite
    the header with the real sizes so every consumer (browser, wave module) gets the length."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params._replace(nframes=len(frames) // max(1, params.sampwidth * params.nchannels)))
            w.writeframes(frames)
        return out.getvalue()
    except Exception:  # noqa: BLE001
        return None
