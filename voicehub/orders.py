"""Voice orders for bots: "Walter, stick close" -> Omni-bot console command via the Lua bridge.

Understands three orders, aimed at one bot by name or at every voiced bot on the speaker's team:
  follow   stick close / follow me / stay with me / on me / cover me [within N metres]
  stay     stay here / hold here / hold position / wait here / camp here / stop
  release  carry on / as you were / do your thing / dismissed / go play / you're free
The Omni-bot side is goals/goal_vhorders.gm (see omnibot/ in the repo)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

UNITS_PER_METRE = 40.0     # a player is 72 units tall (~1.8 m)

_ALL = re.compile(r"\b(everyone|everybody|all of you|all bots|bots|the team|guys|squad)\b", re.I)
_RADIUS = re.compile(r"(\d+(?:\.\d+)?)\s*(m\b|meters?|metres?|yards?|feet|ft\b)", re.I)

# order -> patterns. Checked in this sequence; "stay with me" must win over "stay here".
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("follow", re.compile(
        r"\b(follow me|follow|stick (close|with me|to me|near me)|stay (with|close to|near|on) me|stay close|"
        r"come with me|on me|with me|cover me|escort me|keep (close|up)|tag along|watch my back|be my (wingman|shadow))\b", re.I)),
    ("stay", re.compile(
        r"\b(stay here|stay there|stay put|hold (here|there|position|this spot|the line)|wait here|wait there|"
        r"camp here|camp there|hold up|stop here|stop|halt|guard (this|here|the door|this spot)|dig in)\b", re.I)),
    ("release", re.compile(
        r"\b(carry on|as you were|dismissed|do your (own )?thing|go play|you'?re free|free to go|go on without me|"
        r"back to (work|normal|the objective)|release|stand down|forget it|never ?mind)\b", re.I)),
]


@dataclass
class Order:
    kind: str                 # follow | stay | release
    radius_m: float | None    # follow only
    everyone: bool            # aimed at all bots on the team


def parse(text: str, default_radius_m: float = 25.0) -> Order | None:
    """Return the order in `text`, or None if it is not an order."""
    t = " " + text.strip().lower() + " "
    for kind, rx in PATTERNS:
        if rx.search(t):
            radius = None
            if kind == "follow":
                m = _RADIUS.search(t)
                if m:
                    v = float(m.group(1))
                    unit = m.group(2).lower()
                    if unit.startswith("y"):
                        v *= 0.9144
                    elif unit.startswith(("f",)):
                        v *= 0.3048
                    radius = max(2.0, min(v, 200.0))
                else:
                    radius = default_radius_m
            return Order(kind, radius, bool(_ALL.search(t)))
    return None


class Orders:
    """Tracks active orders (for status/expiry) and builds the console commands."""

    def __init__(self) -> None:
        self.active: dict[int, dict[str, Any]] = {}     # bot slot -> {kind, target, radius_m, since}

    def command(self, order: Order, bot_slot: int, player_slot: int) -> str:
        if order.kind == "follow":
            units = int((order.radius_m or 25.0) * UNITS_PER_METRE)
            self.active[bot_slot] = {"kind": "follow", "target": player_slot, "radius_m": order.radius_m, "since": time.time()}
            return f"bot vh follow {bot_slot} {player_slot} {units}"
        if order.kind == "stay":
            self.active[bot_slot] = {"kind": "stay", "target": player_slot, "radius_m": None, "since": time.time()}
            return f"bot vh stay {bot_slot} {player_slot}"
        self.active.pop(bot_slot, None)
        return f"bot vh release {bot_slot}"

    def followers_of(self, player_slot: int) -> list[int]:
        return [b for b, o in self.active.items() if o.get("target") == player_slot]

    def clear(self) -> None:
        self.active.clear()

    def describe(self, order: Order, player: str, bot_names: list[str]) -> str:
        """Trigger text for the bot's spoken acknowledgement."""
        who = " and ".join(bot_names) if len(bot_names) <= 2 else f"{bot_names[0]} and the others"
        if order.kind == "follow":
            return (f"{player} just ordered you to stick with them and stay within about {order.radius_m:.0f} metres. "
                    "You are already doing it. Acknowledge in one short line, in character.")
        if order.kind == "stay":
            return (f"{player} just ordered you to hold this position and wait here. You are doing it. "
                    "Acknowledge in one short line, in character.")
        return (f"{player} just released you from their orders; you are back to playing the objective. "
                "Acknowledge in one short line, in character.")
