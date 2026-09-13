"""Map knowledge for the bots, read straight out of the game's pk3 files.

Every ET map ships `maps/<name>.objdata` (mission text per team + numbered objectives),
`scripts/<name>.arena` (long name, briefing, respawn times) and `maps/<name>.script`
(whose wm_announce strings are the objective events the game prints). Later pk3s (mod,
homepath) override earlier ones, like the engine's own search path.

`maps.toml` next to config.toml can add or override per-map tips for the bots:

    [maps.oasis]
    tips = "Allied engineers: the pumps are faster than the wall. Axis: hold the garrison stairs."
"""
from __future__ import annotations

import logging
import re
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("voicehub.maps")

_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
_ANNOUNCE = re.compile(r'wm_announce\s+"([^"]+)"')
_COLOR = re.compile(r"\^.")


@dataclass
class MapInfo:
    name: str
    longname: str = ""
    briefing: str = ""
    description: dict[str, str] = field(default_factory=dict)           # axis / allied / neutral
    objectives: dict[str, list[str]] = field(default_factory=dict)      # axis / allied -> ["Primary: ...", ...]
    announcements: list[str] = field(default_factory=list)              # wm_announce strings
    respawn: dict[str, int] = field(default_factory=dict)
    timelimit: int = 0
    tips: str = ""

    def brief(self, team: str) -> str:
        """Prompt text for a bot on `team` ("Axis"/"Allies"). Kept compact: it is part of every prompt."""
        side = "axis" if team.lower().startswith("ax") else "allied"
        title = f"{self.longname} ({self.name})" if self.longname and self.longname.lower() != self.name.lower() else self.name
        out = [f"Map: {title}."]
        mission = self.description.get(side) or self.description.get("neutral") or self.briefing
        if mission:
            out.append(f"Your team's mission: {mission}")
        objs = self.objectives.get(side) or []
        if objs:
            out.append("Your team's objectives: " + " ".join(f"({i}) {o}" for i, o in enumerate(objs, 1)))
        if self.tips:
            out.append(f"Tips: {self.tips}")
        return " ".join(out)


def _strip(s: str) -> str:
    s = _COLOR.sub("", s).replace("**", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_objdata(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    desc: dict[str, str] = {}
    objs: dict[str, list[tuple[int, str]]] = {"axis": [], "allied": []}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("//") or not line:
            continue
        m = re.match(r"wm_mapdescription\s+(axis|allied|neutral)\s+(.*)", line)
        if m:
            q = _QUOTED.search(m.group(2))
            if q:
                desc[m.group(1)] = _strip(q.group(1))
            continue
        m = re.match(r"wm_objective_(axis|allied)_desc\s+(\d+)\s+(.*)", line)
        if m:
            q = _QUOTED.search(m.group(3))
            if q:
                objs[m.group(1)].append((int(m.group(2)), _strip(q.group(1))))
    return desc, {k: [t for _, t in sorted(v)] for k, v in objs.items() if v}


def parse_arena(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("longname", "briefing", "timelimit", "axisRespawnTime", "alliedRespawnTime"):
        m = re.search(rf"^\s*{key}\s+(.*)$", text, re.M | re.I)
        if not m:
            continue
        val = m.group(1).strip()
        q = _QUOTED.match(val)
        out[key.lower()] = _strip(q.group(1)) if q else val.strip('"')
    return out


class MapLibrary:
    def __init__(self, pk3_dirs: list[Path], extras_file: Path | None = None):
        self.pk3_dirs = pk3_dirs
        self.extras_file = extras_file
        self._cache: dict[str, MapInfo] = {}
        self._index: dict[str, tuple[Path, str]] = {}   # lowercase member name -> (pk3, member)
        self._indexed = False

    def _build_index(self) -> None:
        self._indexed = True
        pk3s: list[Path] = []
        for d in self.pk3_dirs:
            if d.is_dir():
                pk3s += sorted(d.glob("*.pk3"))
        for pk in pk3s:   # later dirs / later names win, as in the engine
            try:
                with zipfile.ZipFile(pk) as z:
                    for n in z.namelist():
                        ln = n.lower()
                        if ln.endswith((".objdata", ".arena", ".script")) and ("maps/" in ln or "scripts/" in ln):
                            self._index[ln] = (pk, n)
            except (OSError, zipfile.BadZipFile) as e:
                log.warning("skipping %s: %s", pk, e)
        log.info("map index: %d files from %d pk3s", len(self._index), len(pk3s))

    def _read(self, member: str) -> str:
        hit = self._index.get(member.lower())
        if not hit:
            return ""
        pk, name = hit
        try:
            with zipfile.ZipFile(pk) as z:
                raw = z.read(name)
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("cp1252", errors="replace")
        except (OSError, zipfile.BadZipFile, KeyError):
            return ""

    def _extras(self, name: str) -> dict[str, Any]:
        if not self.extras_file or not self.extras_file.exists():
            return {}
        try:
            with open(self.extras_file, "rb") as f:
                raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            log.warning("maps.toml: %s", e)
            return {}
        maps = raw.get("maps") or {}
        for k, v in maps.items():
            if k.lower() == name.lower():
                return dict(v)
        return {}

    def info(self, name: str) -> MapInfo:
        name = (name or "").strip()
        if not name:
            return MapInfo(name="unknown")
        if not self._indexed:
            self._build_index()
        cached = self._cache.get(name.lower())
        if cached:
            return cached
        mi = MapInfo(name=name)
        desc, objs = parse_objdata(self._read(f"maps/{name}.objdata"))
        mi.description, mi.objectives = desc, objs
        arena = parse_arena(self._read(f"scripts/{name}.arena"))
        mi.longname = arena.get("longname", "")
        mi.briefing = arena.get("briefing", "")
        mi.timelimit = int(float(arena.get("timelimit") or 0) or 0)
        for k in ("axisrespawntime", "alliedrespawntime"):
            if arena.get(k, "").isdigit():
                mi.respawn[k.replace("respawntime", "")] = int(arena[k])
        mi.announcements = [_strip(a) for a in _ANNOUNCE.findall(self._read(f"maps/{name}.script"))]
        ex = self._extras(name)
        mi.tips = str(ex.get("tips", "")).strip()
        if ex.get("longname"):
            mi.longname = str(ex["longname"])
        if ex.get("briefing"):
            mi.briefing = str(ex["briefing"])
        for side in ("axis", "allied"):
            if ex.get(f"{side}_objectives"):
                mi.objectives[side] = [str(o) for o in ex[f"{side}_objectives"]]
        if not (mi.description or mi.objectives or mi.briefing):
            log.info("no objdata/arena found for map %s", name)
        self._cache[name.lower()] = mi
        return mi

    def forget(self) -> None:
        """Drop caches (maps.toml edited, or a new pk3 appeared)."""
        self._cache.clear()
        self._index.clear()
        self._indexed = False
