"""Speech to text with faster-whisper; audio decoded through ffmpeg."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

log = logging.getLogger("voicehub.stt")


async def decode_to_pcm16k(data: bytes) -> np.ndarray:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
        "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(data)
    if proc.returncode != 0 or not out:
        raise ValueError(f"ffmpeg decode failed: {err.decode(errors='replace')[:200]}")
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


class STT:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model = None
        self.device_desc = "not loaded"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        name = self.cfg.get("model", "small.en")
        attempts = []
        if self.cfg.get("device", "cuda") == "cuda":
            idx = int(self.cfg.get("device_index", 0))
            attempts.append(("cuda", idx, self.cfg.get("compute_type", "float32")))
            attempts.append(("cuda", 1 - idx if idx in (0, 1) else 0, self.cfg.get("compute_type", "float32")))
        attempts.append(("cpu", 0, self.cfg.get("cpu_compute_type", "int8")))
        for dev, idx, ct in attempts:
            try:
                t = time.time()
                self.model = WhisperModel(name, device=dev, device_index=idx, compute_type=ct,
                                          cpu_threads=int(self.cfg.get("cpu_threads", 8)))
                self.device_desc = f"{name} on {dev}:{idx} ({ct})"
                log.info("whisper loaded: %s in %.1fs", self.device_desc, time.time() - t)
                return
            except Exception as e:  # noqa: BLE001
                log.warning("whisper load on %s:%s failed: %s", dev, idx, str(e)[:120])
        raise RuntimeError("could not load whisper on any device")

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, self._load)

    def _transcribe(self, audio: np.ndarray, prompt: str) -> str:
        segments, _ = self.model.transcribe(
            audio, beam_size=1, language=self.cfg.get("language", "en"),
            vad_filter=True, initial_prompt=prompt or None, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, data: bytes, prompt: str = "") -> str:
        if self.model is None:
            raise RuntimeError("STT not loaded")
        audio = await decode_to_pcm16k(data)
        if len(audio) < 1600:  # < 0.1 s
            return ""
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._transcribe, audio, prompt)
