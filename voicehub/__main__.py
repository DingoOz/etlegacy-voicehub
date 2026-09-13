"""Entry point: python -m voicehub"""
from __future__ import annotations

import asyncio
import logging
import time

import uvicorn

from . import config as cfgmod
from .brain import Brain
from .game_bridge import GameBridge, Rcon
from .game_events import EventTailer, GameState
from .llm import LLM
from .personas import Personas
from .stt import STT
from .tls import ensure_cert
from .tts import TTS
from .tts_qwen import QwenTTS
from .web import Hub, create_app

log = logging.getLogger("voicehub")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = cfgmod.load()
    state = GameState()
    state.lines = type(state.lines)(maxlen=int(cfg.policy.get("context_lines", 20)))
    events: asyncio.Queue = asyncio.Queue()
    exchange = cfg.exchange_dir
    exchange.mkdir(parents=True, exist_ok=True)

    rcon = Rcon(cfg.game.get("rcon_host", "127.0.0.1"), int(cfg.game.get("rcon_port", 27960)), cfg.rcon_password) \
        if cfg.rcon_password else None
    bridge = GameBridge(exchange, rcon)
    stt = STT(cfg.stt)
    llm = LLM(cfg.llm)
    if cfg.tts.get("engine", "qwen") == "qwen":
        tts = QwenTTS(cfg.tts, cfg.tts.get("default_speaker", "Ryan"), cfg.tts.get("default_style", ""))
        personas = Personas(cfg.personas, cfg.tts.get("default_voice", "en_US-lessac-medium"), tts.default_voice)
    else:
        tts = TTS(cfg.path(cfg.tts.get("voices_dir", "voices")), cfg.tts.get("default_voice", "en_US-lessac-medium"), cfg.tts)
        personas = Personas(cfg.personas, tts.default_voice, cfg.tts.get("default_speaker", "Ryan"))
    hub = Hub(state, cfg.voice)
    brain = Brain(cfg.policy, cfg.voice, state, bridge, stt, llm, tts, personas, hub)
    brain.page_url = cfg.web.get("public_url") or f"https://{cfg.web.get('cert_sans', ['localhost'])[0]}:{cfg.web.get('https_port', 8443)}"
    ctx = {"state": state, "brain": brain, "hub": hub, "llm": llm, "rcon": rcon, "stt": stt, "tts": tts,
           "voice": cfg.voice, "started": time.time()}
    app = create_app(ctx)

    tailer = EventTailer(exchange / "events.jsonl", state, events)

    async def dispatch() -> None:
        while True:
            ev = await events.get()
            try:
                await brain.handle_event(ev)
            except Exception:  # noqa: BLE001
                log.exception("event handling failed")

    async def roster_refresh() -> None:
        while True:
            bridge.request_roster()
            await asyncio.sleep(30)

    async def config_reload() -> None:
        """Pick up edits to [policy] and [voice] in config.toml without a restart."""
        path = cfg.root / "config.toml"
        last = path.stat().st_mtime
        while True:
            await asyncio.sleep(3)
            try:
                m = path.stat().st_mtime
                if m == last:
                    continue
                last = m
                new = cfgmod.load(cfg.root)
                cfg.policy.clear(); cfg.policy.update(new.policy)
                cfg.voice.clear(); cfg.voice.update(new.voice)
                cfg.tts.clear(); cfg.tts.update(new.tts)
                log.info("config.toml reloaded: policy=%s voice=%s", cfg.policy, cfg.voice)
                await hub.broadcast({"type": "settings", "voice": cfg.voice})
            except Exception as e:  # noqa: BLE001
                log.warning("config reload failed: %s", e)

    servers = []
    if cfg.web.get("tls", True):
        cert, key = cfg.path(cfg.web["cert_file"]), cfg.path(cfg.web["key_file"])
        ensure_cert(cert, key, list(cfg.web.get("cert_sans", ["localhost"])))
        servers.append(uvicorn.Server(uvicorn.Config(app, host=cfg.web.get("host", "0.0.0.0"),
                                                     port=int(cfg.web.get("https_port", 8443)),
                                                     ssl_certfile=str(cert), ssl_keyfile=str(key),
                                                     log_level="warning")))
    if cfg.web.get("http_port"):
        servers.append(uvicorn.Server(uvicorn.Config(app, host=cfg.web.get("host", "0.0.0.0"),
                                                     port=int(cfg.web["http_port"]), log_level="warning")))

    # Load models concurrently; the web server comes up immediately so /api/health works meanwhile.
    async def load_models() -> None:
        await asyncio.gather(stt.load(), tts.preload(personas.voices_in_use()), llm.warm())
        log.info("models ready")

    tasks = [asyncio.create_task(t) for t in (tailer.run(), dispatch(), brain.run(), brain.idle_chatter(),
                                                 roster_refresh(), config_reload(), load_models())]
    tasks += [asyncio.create_task(s.serve()) for s in servers]
    log.info("voicehub up: %s  http://localhost:%s", brain.page_url, cfg.web.get("http_port"))
    try:
        await asyncio.gather(*tasks)
    finally:
        await llm.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
