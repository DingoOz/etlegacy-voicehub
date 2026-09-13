"""Send commands to the game: inbox.jsonl for the Lua module, rcon for the engine."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from pathlib import Path

log = logging.getLogger("voicehub.bridge")


class Rcon:
    def __init__(self, host: str, port: int, password: str, timeout: float = 1.5):
        self.addr = (host, port)
        self.password = password
        self.timeout = timeout

    def _send(self, cmd: str) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(self.timeout)
        try:
            s.sendto(b"\xff\xff\xff\xffrcon " + self.password.encode() + b" " + cmd.encode(), self.addr)
            chunks = []
            while True:
                try:
                    d, _ = s.recvfrom(65535)
                except socket.timeout:
                    break
                chunks.append(d[4:].decode(errors="replace").replace("print\n", "", 1))
                s.settimeout(0.3)
            return "".join(chunks)
        finally:
            s.close()

    async def send(self, cmd: str) -> str:
        return await asyncio.get_running_loop().run_in_executor(None, self._send, cmd)

    async def alive(self) -> bool:
        try:
            out = await self.send("status")
            return "map:" in out or "num score" in out
        except OSError:
            return False


class GameBridge:
    def __init__(self, exchange_dir: Path, rcon: Rcon | None):
        self.dir = exchange_dir
        self.inbox = exchange_dir / "inbox.jsonl"
        self.rcon = rcon
        self.dir.mkdir(parents=True, exist_ok=True)
        # Anything queued before this hub instance started is stale.
        self.inbox.write_text("")

    def _append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=True) + "\n"
        with open(self.inbox, "a", encoding="ascii") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def say(self, slot: int, text: str, team: bool = False) -> None:
        self._append({"cmd": "say", "slot": int(slot), "mode": "team" if team else "all", "text": text})

    def cpm(self, slot: int, text: str) -> None:
        """Popup message in one player's chat area."""
        self._append({"cmd": "cpm", "slot": int(slot), "text": text})

    def console(self, text: str) -> None:
        self._append({"cmd": "console", "text": text})

    def request_roster(self) -> None:
        self._append({"cmd": "roster"})

    def ping(self, ident: int) -> None:
        self._append({"cmd": "ping", "id": ident})
