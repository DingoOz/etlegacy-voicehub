"""FastAPI app: static page, REST endpoints, WebSocket sessions (one per open page)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .game_events import GameState, TEAM_SPECTATOR

log = logging.getLogger("voicehub.web")
STATIC = Path(__file__).resolve().parent.parent / "static"


@dataclass
class Session:
    ws: WebSocket
    slot: int | None = None          # in-game client slot the page claims to be
    status: str = "idle"             # last capture state reported by the page
    joined: float = field(default_factory=time.time)


class Hub:
    """Tracks open pages and delivers messages, optionally restricted to one team."""

    def __init__(self, state: GameState, voice: dict[str, Any]):
        self.state = state
        self.voice = voice
        self.sessions: dict[WebSocket, Session] = {}

    # ------------------------------------------------------------- queries
    def team_of(self, s: Session) -> int:
        pl = self.state.players.get(s.slot) if s.slot is not None else None
        return pl.team if pl else TEAM_SPECTATOR

    def sessions_for_slot(self, slot: int) -> list[Session]:
        return [s for s in self.sessions.values() if s.slot == slot]

    def hears(self, s: Session, team: int | None) -> bool:
        if team is None or not self.voice.get("team_only", True):
            return True
        mine = self.team_of(s)
        if mine == team:
            return True
        return mine == TEAM_SPECTATOR and bool(self.voice.get("spectators_hear_all", True))

    # ------------------------------------------------------------- delivery
    async def _send(self, s: Session, data: str) -> None:
        try:
            await s.ws.send_text(data)
        except Exception:  # noqa: BLE001
            self.sessions.pop(s.ws, None)

    async def broadcast(self, msg: dict[str, Any], team: int | None = None,
                        exclude_slot: int | None = None) -> None:
        """Send to every page; with `team`, only to pages on that team (and spectators if allowed)."""
        data = json.dumps(msg)
        for s in list(self.sessions.values()):
            if exclude_slot is not None and s.slot == exclude_slot:
                continue
            if self.hears(s, team):
                await self._send(s, data)

    async def send_slot(self, slot: int, msg: dict[str, Any]) -> int:
        """Send to the page(s) claiming this slot. Returns how many received it."""
        targets = self.sessions_for_slot(slot)
        data = json.dumps(msg)
        for s in targets:
            await self._send(s, data)
        return len(targets)

    def to_dict(self) -> dict[str, Any]:
        return {"pages": len(self.sessions),
                "claimed_slots": sorted({s.slot for s in self.sessions.values() if s.slot is not None})}


def create_app(ctx: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="voicehub")
    hub: Hub = ctx["hub"]
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    async def state():
        brain = ctx["brain"]
        st = ctx["state"].to_dict()
        st["voiced_bots"] = [b.clean for b in brain.voiced_bots()]
        st["voice"] = ctx["voice"]
        st["hub"] = hub.to_dict()
        return st

    @app.get("/api/health")
    async def health():
        llm, rcon, stt, state, brain = ctx["llm"], ctx["rcon"], ctx["stt"], ctx["state"], ctx["brain"]
        return {
            "stt": stt.device_desc,
            "llm": {"model": llm.model, "reachable": await llm.alive(), "warm": llm.ok},
            "rcon": await rcon.alive() if rcon else None,
            "game": state.to_dict(),
            "tts": ctx["tts"].device_desc,
            "voices": ctx["tts"].available(),
            "timings": brain.timings[-10:],
            "hub": hub.to_dict(),
            "policy": brain.p,
            "voice": ctx["voice"],
            "uptime_s": round(time.time() - ctx["started"], 1),
        }

    @app.post("/api/utterance")
    async def utterance(audio: UploadFile = File(...), speaker_slot: str = Form(""), mode: str = Form("hold")):
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        slot = int(speaker_slot) if speaker_slot.strip().isdigit() else None
        try:
            return await ctx["brain"].handle_utterance(data, slot, mode)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/test_speak")
    async def test_speak(request: Request, bot: str = "", text: str = "hello from voicehub"):
        brain = ctx["brain"]
        bots = brain.voiced_bots()
        target = next((b for b in bots if b.clean.lower() == bot.lower()), None) if bot else (bots[0] if bots else None)
        if not target:
            raise HTTPException(404, "no such voiced bot connected")
        return await brain.speak(target, text)

    @app.post("/api/ask")
    async def ask(request: Request, bot: str = "", text: str = "", speaker: str = ""):
        """Text-only path for testing the LLM without a microphone."""
        from .brain import Job
        brain = ctx["brain"]
        speaker_pl = next((p for p in ctx["state"].humans() if p.clean.lower() == speaker.lower()), None)
        target = brain.find_bot_in_text(bot, speaker_pl) if bot else brain.pick_bot(speaker_pl, text)
        if not target:
            raise HTTPException(404, "no voiced bot connected (on your team)")
        who = speaker_pl.clean if speaker_pl else (speaker or "a player on the voice page")
        brain.jobs.put_nowait(Job(target, f'{who} said to you over the radio: "{text}"', True, time.time()))
        return {"queued_for": target.clean}

    @app.get("/audio/{ident}.wav")
    async def audio(ident: str):
        data = ctx["brain"].audio.get(ident)
        if data is None:
            raise HTTPException(404)
        return Response(content=data, media_type="audio/wav", headers={"Cache-Control": "max-age=600"})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        sess = Session(websocket)
        hub.sessions[websocket] = sess
        brain = ctx["brain"]
        try:
            await websocket.send_text(json.dumps({"type": "state", "state": ctx["state"].to_dict(),
                                                  "voice": ctx["voice"]}))
            while True:
                raw = await websocket.receive_text()
                if raw == "ping" or not raw.startswith("{"):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "hello":
                    slot = msg.get("slot")
                    sess.slot = int(slot) if isinstance(slot, (int, str)) and str(slot).isdigit() else None
                elif kind == "status":
                    sess.status = str(msg.get("state", "idle"))[:32]
                    await brain.page_status(sess.slot, sess.status, str(msg.get("detail", ""))[:120])
        except WebSocketDisconnect:
            pass
        finally:
            hub.sessions.pop(websocket, None)

    return app
