"""Ollama chat client with prompt building and output sanitising."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import httpx

from .tts import EMOTIONS, parse_emotion

log = logging.getLogger("voicehub.llm")

RULES = (
    "You are playing a bot character in the game Wolfenstein: Enemy Territory on a friends' LAN party. "
    "Reply with exactly ONE short spoken line, at most {max_words} words. Plain ASCII only, no emoji, "
    "no quotation marks, no name prefix, no stage directions or asterisks. Stay in character and be fun. "
    "{rating} Do not mention being an AI unless directly asked. "
    "Address the player you are answering by their in-game name; never say your own name. "
    "Start the line with an emotion tag in square brackets, one of: {emotions}. Pick the tag that fits the "
    "moment and write the line so it sounds that way when spoken aloud: use exclamation marks, question "
    "marks, commas and ... for pauses, and short punchy sentences. Example: [excited] Yes! Got him, Dingo, did you see that?"
)

# [policy] rating in config.toml: how NSFW the bots may get.
RATINGS = {
    "family": "Keep it squeaky clean: no swearing, no innuendo, no insults; suitable for young kids.",
    "pg13": "Be a little cheeky but never cruel; mild language only, keep it PG-13.",
    "r": "Swearing, crude jokes, dark humour and trash talk are fine; insults stay playful between friends. "
         "No slurs and no hate about anyone's race, religion, gender, sexuality or disability.",
    "explicit": "Adults only: swear freely, be crude, vulgar and savage in your trash talk, sexual jokes allowed. "
                "The one hard line: no slurs or hate about anyone's race, religion, gender, sexuality or disability.",
}


class LLM:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.client = httpx.AsyncClient(base_url=cfg.get("url", "http://localhost:11434"),
                                        timeout=float(cfg.get("timeout_s", 12)))
        self.model = cfg.get("model", "qwen3.5:4b")
        self.max_words = int(cfg.get("max_words", 30))
        self.max_chars = int(cfg.get("max_chars", 140))
        self.ok = False

    async def close(self) -> None:
        await self.client.aclose()

    async def alive(self) -> bool:
        try:
            r = await self.client.get("/api/tags")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def warm(self) -> None:
        try:
            await self._chat([{"role": "user", "content": "Say hi."}], timeout=180)
            self.ok = True
            log.info("llm %s warm", self.model)
        except Exception as e:  # noqa: BLE001
            log.warning("llm warm-up failed: %s", e)

    async def _chat(self, messages: list[dict[str, str]], timeout: float | None = None) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "think": bool(self.cfg.get("think", False)),
            "keep_alive": self.cfg.get("keep_alive", -1),
            "options": {
                "num_predict": int(self.cfg.get("num_predict", 60)),
                "temperature": float(self.cfg.get("temperature", 0.9)),
            },
            "messages": messages,
        }
        r = await self.client.post("/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def system_prompt(self, bot_name: str, persona: str, team: str, cls: str, game_map: str,
                      roster: str, context: str, rating: str = "pg13", activity: str = "") -> str:
        # Constant text first so Ollama can reuse its cached prompt prefix across bots and turns.
        rating_text = RATINGS.get(rating.lower(), RATINGS["pg13"])
        return (
            RULES.format(max_words=self.max_words, rating=rating_text, emotions=", ".join(EMOTIONS)) +
            f"\n\nYour name is {bot_name}. You are a {cls} on the {team} team on the map {game_map}. "
            f"Personality: {persona}."
            + (f" Your situation in the match: {activity}." if activity else "") +
            f"\n\nPlayers currently on the server:\n{roster}\n\nRecent events and chat (oldest first):\n{context or '(nothing yet)'}"
        )

    def clean(self, text: str) -> tuple[str, str]:
        """Returns (emotion, spoken line)."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = text.strip().splitlines()[0] if text.strip() else ""
        name_re = r"^\s*\*?[A-Za-z0-9_ -]{1,24}:\s*(?=\S)"
        text = re.sub(name_re, "", text)  # "Name: "
        emotion, text = parse_emotion(text)
        text = re.sub(name_re, "", text)  # "[tag] Name: "
        text = re.sub(r"\*[^*]*\*", "", text)  # *stage directions*
        text = text.strip().strip('"\'`').strip()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        if len(words) > self.max_words:
            text = " ".join(words[: self.max_words])
        if len(text) > self.max_chars:
            cut = text[: self.max_chars].rsplit(" ", 1)[0]
            text = cut
        return emotion, text.strip()

    async def reply(self, system: str, trigger: str) -> tuple[str, str]:
        raw = await self._chat([{"role": "system", "content": system}, {"role": "user", "content": trigger}])
        emotion, out = self.clean(raw)
        log.debug("llm raw=%r emotion=%s clean=%r", raw[:200], emotion, out)
        return emotion, out
