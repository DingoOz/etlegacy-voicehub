"""Tail events.jsonl written by the Lua module and maintain a GameState."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("voicehub.events")

TEAM_AXIS, TEAM_ALLIES, TEAM_SPECTATOR = 1, 2, 3
TEAM_NAMES = {TEAM_AXIS: "Axis", TEAM_ALLIES: "Allies", TEAM_SPECTATOR: "Spectator"}
CLASS_NAMES = {0: "soldier", 1: "medic", 2: "engineer", 3: "field ops", 4: "covert ops"}

# Means-of-death indices that are not really "kills" worth commenting on.
_BORING_MODS = set()


@dataclass
class Player:
    slot: int
    name: str = ""
    clean: str = ""
    guid: str = ""
    team: int = 0
    cls: int = 0
    bot: bool = False
    health: int | None = None
    kills: int = 0
    deaths: int = 0
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=5))   # this player's own latest doings

    def note(self, what: str) -> None:
        self.recent.append(what)

    def activity_text(self) -> str:
        bits = [f"{self.kills} kills, {self.deaths} deaths"]
        if self.health is not None:
            bits.append("dead right now" if self.health <= 0 else f"health {self.health}")
        s = ", ".join(bits)
        if self.recent:
            s += ". Your latest: " + "; ".join(self.recent)
        return s

    @property
    def team_name(self) -> str:
        return TEAM_NAMES.get(self.team, "unknown")

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.cls, "unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot, "name": self.clean, "team": self.team_name, "team_id": self.team,
            "class": self.class_name, "bot": self.bot,
        }


@dataclass
class GameState:
    map: str = ""
    players: dict[int, Player] = field(default_factory=dict)
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    last_event_ts: float = 0.0
    level_time: int = 0
    start_lt: int = 0
    timelimit_min: float = 0.0
    objective_log: deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def minutes_left(self) -> float | None:
        """Rough time left on the clock (counts from map load, so warm-up skews it a little)."""
        if not self.timelimit_min or not self.level_time:
            return None
        return max(0.0, self.timelimit_min - (self.level_time - self.start_lt) / 60000.0)

    def progress_text(self) -> str:
        return "\n".join(f"- {x}" for x in self.objective_log) or "(no objective completed yet)"

    def humans(self) -> list[Player]:
        return [p for p in self.players.values() if not p.bot and p.clean]

    def bots(self) -> list[Player]:
        return [p for p in self.players.values() if p.bot and p.clean]

    def humans_on(self, team: int) -> list[Player]:
        return [p for p in self.humans() if p.team == team]

    def name_of(self, slot: int | None) -> str:
        if slot is None:
            return "someone"
        p = self.players.get(int(slot))
        return p.clean if p and p.clean else f"player {slot}"

    def add_line(self, text: str) -> None:
        self.lines.append(text)

    def context_text(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map,
            "players": [p.to_dict() for p in sorted(self.players.values(), key=lambda p: p.slot)],
            "last_event_age_s": round(time.time() - self.last_event_ts, 1) if self.last_event_ts else None,
        }


def _player_from(rec: dict[str, Any], old: Player | None = None) -> Player:
    p = Player(
        slot=int(rec.get("slot", -1)),
        name=rec.get("name") or "",
        clean=(rec.get("clean") or "").strip(),
        guid=rec.get("guid") or "",
        team=int(rec.get("team") or 0),
        cls=int(rec.get("class") or 0),
        bot=bool(rec.get("bot")),
        health=int(rec["health"]) if rec.get("health") is not None else None,
        kills=int(rec.get("kills") or 0),
        deaths=int(rec.get("deaths") or 0),
    )
    if old and old.guid == p.guid:
        p.recent = old.recent
    return p


class EventTailer:
    """Follows events.jsonl from its current end, updating state and feeding a queue."""

    def __init__(self, path: Path, state: GameState, queue: asyncio.Queue, poll_s: float = 0.1):
        self.path = path
        self.state = state
        self.queue = queue
        self.poll_s = poll_s
        self._offset = 0
        self._buf = b""

    def apply(self, ev: dict[str, Any]) -> None:
        st = self.state
        st.last_event_ts = time.time()
        st.level_time = int(ev.get("lt") or 0)
        kind = ev.get("ev")
        if kind == "mapstart":
            st.map = ev.get("map") or st.map
            st.players.clear()
            st.lines.clear()
            st.objective_log.clear()
            st.start_lt = st.level_time
            st.timelimit_min = float(ev.get("timelimit") or 0)
            st.add_line(f"[map {st.map} started]")
        elif kind == "announce":
            text = str(ev.get("text") or "").strip()
            if text:
                st.objective_log.append(text)
                st.add_line(f"[objective] {text}")
        elif kind == "roster":
            if ev.get("map"):
                st.map = ev["map"]
            old = dict(st.players)
            st.players.clear()
            for rec in ev.get("players") or []:
                p = _player_from(rec, old.get(int(rec.get("slot", -1))))
                if p.slot >= 0:
                    st.players[p.slot] = p
        elif kind in ("begin", "userinfo"):
            p = _player_from(ev, st.players.get(int(ev.get("slot", -1))))
            if p.slot >= 0 and (p.clean or p.slot not in st.players):
                st.players[p.slot] = p
        elif kind == "disconnect":
            st.players.pop(int(ev.get("slot", -1)), None)
        elif kind == "chat":
            who = st.name_of(ev.get("slot"))
            mode = ev.get("mode")
            tag = " (team)" if mode == "team" else ""
            st.add_line(f"{who}{tag}: {ev.get('text', '')}")
        elif kind == "kill":
            k, v = ev.get("killer"), ev.get("victim")
            if k is None or v is None:
                return
            kp, vp = st.players.get(int(k)), st.players.get(int(v))
            if k == v or k >= 1022:
                st.add_line(f"{st.name_of(v)} died")
                if vp:
                    vp.deaths += 1; vp.health = 0; vp.note("died")
            else:
                st.add_line(f"{st.name_of(k)} killed {st.name_of(v)}")
                if kp:
                    kp.kills += 1; kp.note(f"killed {st.name_of(v)}")
                if vp:
                    vp.deaths += 1; vp.health = 0; vp.note(f"got killed by {st.name_of(k)}")
        elif kind == "revive":
            m, v = st.players.get(int(ev.get("medic", -1))), st.players.get(int(ev.get("victim", -1)))
            st.add_line(f"{st.name_of(ev.get('medic'))} revived {st.name_of(ev.get('victim'))}")
            if m:
                m.note(f"revived {st.name_of(ev.get('victim'))}")
            if v:
                v.health = 50; v.note(f"was revived by {st.name_of(ev.get('medic'))}")
        elif kind == "spawn":
            p = st.players.get(int(ev.get("slot", -1)))
            if p and not ev.get("revived"):
                p.health = 100; p.note("respawned")

    async def run(self) -> None:
        # Start at end of file: history before the hub started is irrelevant.
        try:
            self._offset = self.path.stat().st_size
        except FileNotFoundError:
            self._offset = 0
        while True:
            try:
                self._poll()
            except FileNotFoundError:
                self._offset = 0
            except Exception:  # noqa: BLE001
                log.exception("event poll failed")
            await asyncio.sleep(self.poll_s)

    def _poll(self) -> None:
        size = self.path.stat().st_size
        if size < self._offset:
            self._offset = 0  # rotated / truncated
            self._buf = b""
        if size == self._offset:
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read(size - self._offset)
        self._offset = size
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("bad event line: %r", raw[:120])
                continue
            self.apply(ev)
            self.queue.put_nowait(ev)
