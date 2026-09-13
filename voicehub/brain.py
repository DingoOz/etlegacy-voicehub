"""Decides who speaks and when; runs STT -> LLM -> TTS -> game + browsers."""
from __future__ import annotations

import asyncio
import difflib
import logging
import random
import re
import time
import uuid
import wave
import io
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .game_bridge import GameBridge
from .game_events import GameState, Player, TEAM_AXIS, TEAM_ALLIES, TEAM_SPECTATOR
from .llm import LLM, trashy
from .maps import MapLibrary
from .personas import Personas
from .stt import STT
from .tts import TTS

log = logging.getLogger("voicehub.brain")

# Phrases that the existing Omni-bot chat listeners understand (goal_testbot,
# goal_fireteam). When spoken, they are re-posted as team chat too.
BOT_COMMAND_RE = re.compile(r"\b(bot|bots)\s+(come|go|stop|wait|follow|camp|move|dyno)\b", re.I)

# Things whisper tends to "hear" in silence or noise; dropped in open-mic mode.
HALLUCINATIONS = {"thank you", "thanks", "thank you.", "you", "bye", "thanks for watching", "the end",
                  "so", "okay", "ok", "um", "uh", "hmm", "yeah", "oh"}

# Spontaneous lines. Most are about playing the game well; a share (policy.chitchat_prob) is
# small talk about the bots' lives, using each persona's `life` notes.
TACTICAL_PROMPTS = [
    "Quiet moment. Look at your team's objectives and suggest the next thing the team should do, in character.",
    "Tell your team what you are doing right now and what you plan to do next, based on your class and your latest actions.",
    "Ask a teammate bot by name to do something specific that would help the current objective (cover, build, revive, ammo, scout).",
    "Remind the team of the current objective and where on the map it is.",
    "Give one practical tip for your class on this map.",
    "Report what you can see from where you are and whether it is safe, in character.",
    "Suggest a route or a place to hold based on the map, in one line.",
    "Ask a teammate what they need: ammo, a medic, an engineer, cover.",
    "Make a short remark about how the match is going and what would turn it around, constructive.",
    "Coordinate with a teammate bot by name: propose that the two of you push or hold together.",
]
CHITCHAT_PROMPTS = [
    "Nothing is happening. Ask a teammate bot by name something about their life outside the game.",
    "Mention a small thing from your own life outside the game, the way friends do between rounds.",
    "Answer nobody in particular: share how your day has been going so far, briefly.",
    "Ask a teammate bot by name how something from their life is going (their pet, their job, their weekend).",
    "Make a bit of friendly small talk about food, weather, pets or weekend plans.",
]
IDLE_PROMPTS = TACTICAL_PROMPTS + CHITCHAT_PROMPTS


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends; glue very short bits (e.g. "Yes!") onto the next sentence."""
    raw = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Za-z\"'(])", text.strip()) if p.strip()]
    out: list[str] = []
    for p in raw:
        if out and len(out[-1].split()) < 3:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text]


def apply_gain(wav_bytes: bytes, gain: float) -> bytes:
    """Scale 16-bit PCM WAV samples by `gain`, soft-limited so boosts do not clip harshly."""
    try:
        import numpy as np
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            params = r.getparams()
            if params.sampwidth != 2:
                return wav_bytes
            pcm = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        x = pcm * gain
        if gain > 1.0:
            x = np.tanh(x * 1.2) / np.tanh(1.2)
        x = np.clip(x, -0.99, 0.99)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params._replace(nframes=len(x) // max(1, params.nchannels)))
            w.writeframes((x * 32767.0).astype(np.int16).tobytes())
        return out.getvalue()
    except Exception:  # noqa: BLE001
        return wav_bytes


@dataclass
class Job:
    bot: Player
    trigger: str
    prompted: bool
    created: float
    depth: int = 0          # position in a bot-to-bot thread (0 = not a thread reply)
    topic: str = "game"     # "game" or "life": what a follow-up from a teammate should be about


class AudioStore:
    def __init__(self, limit: int = 64):
        self.limit = limit
        self._d: OrderedDict[str, bytes] = OrderedDict()

    def put(self, data: bytes) -> str:
        ident = uuid.uuid4().hex
        self._d[ident] = data
        while len(self._d) > self.limit:
            self._d.popitem(last=False)
        return ident

    def get(self, ident: str) -> bytes | None:
        return self._d.get(ident)


class Brain:
    def __init__(self, policy: dict[str, Any], voice: dict[str, Any], state: GameState, bridge: GameBridge,
                 stt: STT, llm: LLM, tts: TTS, personas: Personas, hub: Any, maps: MapLibrary | None = None):
        self.p = policy          # [policy] from config.toml; hot-reloaded in place
        self.maps = maps
        self.v = voice           # [voice]  from config.toml; hot-reloaded in place
        self.state = state
        self.bridge = bridge
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.personas = personas
        self.hub = hub
        self.audio = AudioStore()
        self.jobs: asyncio.Queue[Job] = asyncio.Queue()
        self.last_spoke_global = 0.0
        self.last_spoke: dict[str, float] = {}
        self.timings: list[dict[str, float]] = []
        self.greeted: set[str] = set()
        self.page_url = ""

    # ------------------------------------------------------------------ helpers
    @property
    def team_only(self) -> bool:
        return bool(self.v.get("team_only", True))

    @property
    def rating(self) -> str:
        return str(self.p.get("rating", "pg13"))

    @property
    def trashy(self) -> bool:
        return trashy(self.rating)

    def map_brief(self, team: str) -> str:
        if not self.maps or not self.state.map:
            return ""
        try:
            return self.maps.info(self.state.map).brief(team)
        except Exception as e:  # noqa: BLE001
            log.warning("map brief failed: %s", e)
            return ""

    def voiced_bots(self, team: int | None = None) -> list[Player]:
        bots = [b for b in self.state.bots() if self.personas.for_name(b.clean).voiced]
        if team in (TEAM_AXIS, TEAM_ALLIES):
            bots = [b for b in bots if b.team == team]
        return bots

    def _audience_team(self, speaker: Player | None) -> int | None:
        """Team a speaker's words stay within, or None when anyone may answer."""
        if self.team_only and speaker and speaker.team in (TEAM_AXIS, TEAM_ALLIES):
            return speaker.team
        return None

    def find_bot_in_text(self, text: str, speaker: Player | None = None) -> Player | None:
        words = re.findall(r"[a-z]+", text.lower())
        best, best_score = None, 0.0
        for b in self.voiced_bots(self._audience_team(speaker)):
            name = b.clean.lower()
            for w in words:
                if len(w) < 3:
                    continue
                score = difflib.SequenceMatcher(None, w, name).ratio()
                if w == name:
                    score = 1.0
                if score > best_score:
                    best, best_score = b, score
        return best if best_score >= 0.8 else None

    def pick_bot(self, speaker: Player | None, text: str = "", team: int | None = None) -> Player | None:
        """A bot to answer `speaker`; with team_only, always one on the speaker's own team."""
        if team is None:
            team = self._audience_team(speaker)
        bots = self.voiced_bots(team)
        if not bots:
            return None
        named = self.find_bot_in_text(text, speaker) if text else None
        if named and named in bots:
            return named
        if speaker and speaker.team in (TEAM_AXIS, TEAM_ALLIES):
            same = [b for b in bots if b.team == speaker.team]
            if same:
                bots = same
        return random.choice(bots)

    def bot_with_audience(self) -> Player | None:
        """A random voiced bot whose team has a human to hear it (any bot if teams do not matter)."""
        bots = self.voiced_bots()
        if self.team_only:
            bots = [b for b in bots if self.state.humans_on(b.team)]
        elif not self.state.humans():
            bots = []
        return random.choice(bots) if bots else None

    def roster_text(self) -> str:
        rows = []
        for pl in sorted(self.state.players.values(), key=lambda x: x.slot):
            if not pl.clean:
                continue
            kind = "bot" if pl.bot else "human"
            rows.append(f"- {pl.clean} ({kind}, {pl.team_name}, {pl.class_name})")
        return "\n".join(rows) or "(nobody)"

    def stt_prompt(self) -> str:
        names = ", ".join(b.clean for b in self.state.bots()[:12])
        return f"Enemy Territory voice chat on {self.state.map}. Bots: {names}."

    def notify(self, slot: int | None, text: str) -> None:
        """Short in-game popup for one player (if they are on the server)."""
        if slot is not None and slot in self.state.players:
            self.bridge.cpm(slot, "^3voice: ^7" + text)

    # ------------------------------------------------------------------ inputs
    async def handle_utterance(self, audio: bytes, speaker_slot: int | None, mode: str = "hold") -> dict[str, Any]:
        t0 = time.time()
        text = await self.stt.transcribe(audio, self.stt_prompt())
        t_stt = time.time() - t0
        speaker = self.state.players.get(speaker_slot) if speaker_slot is not None else None
        speaker_name = speaker.clean if speaker else "someone on the voice page"
        if mode == "open" and text.strip(" .!?").lower() in HALLUCINATIONS:
            text = ""
        if not text:
            if mode != "open":
                self.notify(speaker_slot, "nothing heard")
            return {"text": "", "stt_s": round(t_stt, 2)}
        log.info("utterance from %s [%s] (%.2fs): %s", speaker_name, mode, t_stt, text)
        team = speaker.team if speaker else None
        self.state.add_line(f"{speaker_name} (voice): {text}")
        await self.hub.broadcast({"type": "transcript", "speaker": speaker_name, "text": text}, team=team)
        if speaker:
            # Show it in-game as the player's own chat. With team_only it is team chat, which is also
            # what the Omni-bot listeners (bot come/go/stop, class swaps) react to.
            is_cmd = bool(BOT_COMMAND_RE.search(text))
            self.bridge.say(speaker.slot, text, team=self.team_only or is_cmd)
            if self.v.get("relay_human_voice", False) and team in (TEAM_AXIS, TEAM_ALLIES):
                ident = self.audio.put(audio)
                await self.hub.broadcast({"type": "voice", "speaker": speaker_name, "team": speaker.team_name,
                                          "text": text, "audio_url": f"/audio/{ident}.wav"},
                                         team=team, exclude_slot=speaker.slot)
        bot = None
        wants_reply = True
        if mode == "open":
            # Open mic: only answer when a bot is addressed, or now and then.
            addressed = self.find_bot_in_text(text, speaker) or re.search(r"\bbots?\b", text, re.I)
            wants_reply = bool(addressed) or random.random() < float(self.p.get("open_mic_reply_prob", 0.25))
        if wants_reply:
            bot = self.pick_bot(speaker, text)
            if bot:
                self.jobs.put_nowait(Job(bot, f'{speaker_name} said to you over the radio: "{text}"', True, time.time()))
            elif mode != "open":
                self.notify(speaker_slot, "no voiced bot on your team to answer")
        return {"text": text, "stt_s": round(t_stt, 2), "bot": bot.clean if bot else None}

    async def page_status(self, slot: int | None, status: str, detail: str = "") -> None:
        """The page reports capture state changes; mirror them in-game for that player."""
        msgs = {
            "listening": "^2listening...^7 (stops when you stop talking)",
            "sending": "sending...",
            "mic_on": "^2open mic on^7 - everything you say goes to your team",
            "mic_off": "open mic off",
            "timeout": "heard nothing",
            "nomic": "no microphone on the voice page",
        }
        if status in msgs:
            self.notify(slot, msgs[status] + (f" {detail}" if detail else ""))

    async def handle_voice_cmd(self, ev: dict[str, Any]) -> None:
        slot = int(ev.get("slot", -1))
        cmd = ev.get("cmd")
        if cmd == "status":
            n = len(self.hub.sessions_for_slot(slot))
            self.notify(slot, f"{n} page(s) joined as you; team_only={'on' if self.team_only else 'off'}")
            return
        delivered = await self.hub.send_slot(slot, {"type": "cmd", "cmd": cmd, "arg": ev.get("arg", "")})
        if not delivered:
            self.notify(slot, f"open {self.page_url or 'the voice page'} and pick your name first")

    async def handle_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("ev")
        st = self.state
        if kind in ("roster", "begin", "disconnect", "userinfo", "mapstart"):
            await self.hub.broadcast({"type": "state", "state": st.to_dict()})
        if kind == "voice_cmd":
            await self.handle_voice_cmd(ev)
        elif kind == "chat":
            slot = int(ev.get("slot", -1))
            pl = st.players.get(slot)
            text = ev.get("text") or ""
            if not pl or pl.bot or not text.strip():
                return  # never react to bot chat: avoids bot-bot loops
            named = self.find_bot_in_text(text, pl)
            mentions_bots = named or re.search(r"\bbots?\b", text, re.I)
            if mentions_bots:
                bot = named or self.pick_bot(pl, text)
                if bot:
                    self.jobs.put_nowait(Job(bot, f'{pl.clean} typed in chat: "{text}"', True, time.time()))
        elif kind == "kill":
            k, v = ev.get("killer"), ev.get("victim")
            if k is None or v is None or k == v or k >= 1022:
                return
            kp, vp = st.players.get(int(k)), st.players.get(int(v))
            if not kp or not vp:
                return
            if random.random() > float(self.p.get("kill_banter_prob", 0.3)):
                return
            to_team = " Say it to your team." if self.team_only else ""
            if kp.bot and not vp.bot and self.personas.for_name(kp.clean).voiced and self._has_audience(kp):
                what = ("Gloat briefly." if self.trashy else
                        "Report it to your team briefly and usefully (where it happened, what the enemy was doing); no gloating.")
                self.jobs.put_nowait(Job(kp, f"You just killed {vp.clean}. {what}{to_team}", False, time.time()))
            elif vp.bot and not kp.bot and self.personas.for_name(vp.clean).voiced and self._has_audience(vp):
                what = ("React briefly." if self.trashy else
                        "React briefly like a good sport: give credit, or tell your team where the danger is.")
                self.jobs.put_nowait(Job(vp, f"{kp.clean} just killed you. {what}{to_team}", False, time.time()))
        elif kind == "revive":
            m, v = st.players.get(int(ev.get("medic", -1))), st.players.get(int(ev.get("victim", -1)))
            if m and v and v.bot and not m.bot and random.random() < float(self.p.get("revive_thanks_prob", 0.6)):
                self.jobs.put_nowait(Job(v, f"{m.clean} just revived you. Thank them in character.", False, time.time()))
        elif kind == "begin":
            team = int(ev.get("team") or 0)
            key = f"{ev.get('guid') or ev.get('slot')}:{team}"
            if ev.get("bot") or not ev.get("clean") or team not in (TEAM_AXIS, TEAM_ALLIES) or key in self.greeted:
                return
            self.greeted.add(key)
            if random.random() < float(self.p.get("greet_prob", 0.8)):
                bot = self.pick_bot(None, team=team if self.team_only else None)
                if bot:
                    self.jobs.put_nowait(Job(bot, f"{ev['clean']} just joined your team. Greet them.", False, time.time()))
        elif kind == "announce":
            await self.handle_announce(str(ev.get("text") or ""))
        elif kind == "mapstart" and not ev.get("restart"):
            self.greeted.clear()
            if random.random() < float(self.p.get("mapstart_prob", 0.5)):
                await asyncio.sleep(8)  # let the roster settle
                bot = self.bot_with_audience()
                if bot:
                    self.jobs.put_nowait(Job(bot, f"The map {st.map} just started. Say something to get everyone going.", False, time.time()))

    async def handle_announce(self, text: str) -> None:
        """An objective was completed or lost: one bot per side may comment, constructively."""
        if not text or random.random() > float(self.p.get("objective_prob", 0.8)):
            return
        low = text.lower()
        actor = TEAM_AXIS if low.startswith(("axis", "the axis")) else TEAM_ALLIES if low.startswith(("allies", "allied", "the allies")) else None
        for team in (TEAM_AXIS, TEAM_ALLIES):
            bots = [b for b in self.voiced_bots(team) if self._has_audience(b)]
            if not bots:
                continue
            bot = random.choice(bots)
            if actor is None:
                trig = f'Game announcement: "{text}". Tell your team what it means for you and what to do next.'
            elif team == actor:
                trig = (f'Your team just did it: "{text}". Cheer briefly, then say what the team should go for next '
                        "according to your objectives.")
            else:
                trig = (f'Bad news for your team: "{text}". Rally your team without whining: say what to defend or '
                        "retake next and where.")
            self.jobs.put_nowait(Job(bot, trig, False, time.time()))

    def _has_audience(self, bot: Player) -> bool:
        return bool(self.state.humans_on(bot.team)) if self.team_only else bool(self.state.humans())

    # ------------------------------------------------------------------ idle chatter
    async def idle_chatter(self) -> None:
        """Bots speak up on their own every idle_chatter_interval_s seconds (0 = off), with jitter."""
        while True:
            interval = float(self.p.get("idle_chatter_interval_s", 0) or 0)
            if interval <= 0 or not self.p.get("idle_chatter", True):
                await asyncio.sleep(5)
                continue
            jitter = min(max(float(self.p.get("idle_chatter_jitter", 0.5)), 0.0), 0.95)
            due = time.time() + interval * random.uniform(1 - jitter, 1 + jitter)
            while time.time() < due:
                await asyncio.sleep(min(5, due - time.time()))
                new = float(self.p.get("idle_chatter_interval_s", 0) or 0)
                if new != interval:
                    due = min(due, time.time() + new)   # setting changed: apply it now
                    interval = new
            bot = self.bot_with_audience()
            if not bot:
                continue
            if time.time() - self.last_spoke_global < float(self.p.get("idle_chatter_min_quiet_s", 20)):
                continue  # somebody spoke recently; not idle
            life = random.random() < float(self.p.get("chitchat_prob", 0.25))
            if life and not self.personas.for_name(bot.clean).life:
                life = False   # nothing to chat about for this one
            prompt = random.choice(CHITCHAT_PROMPTS if life else TACTICAL_PROMPTS)
            self.jobs.put_nowait(Job(bot, prompt, False, time.time(), topic="life" if life else "game"))

    # ------------------------------------------------------------------ output
    async def speak(self, bot: Player, text: str, team: bool | None = None, emotion: str = "neutral") -> dict[str, Any]:
        """Say `text` as `bot`: in-game chat once, audio to the pages sentence by sentence so the
        first sentence plays while the rest is still being synthesized (Qwen3-TTS is slow here)."""
        persona = self.personas.for_name(bot.clean)
        if team is None:
            team = self.team_only
        self.state.add_line(f"{bot.clean}: {text}")
        self.bridge.say(bot.slot, text, team=team)
        parts = split_sentences(text) if self.tts_streaming else [text]
        t0 = time.time()
        first: dict[str, Any] | None = None
        for i, part in enumerate(parts):
            wav = await self.synth(part, persona, emotion)
            ident = self.audio.put(wav)
            msg = {"type": "speech", "bot": bot.clean, "team": bot.team_name, "text": part, "emotion": emotion,
                   "part": i + 1, "parts": len(parts), "audio_url": f"/audio/{ident}.wav"}
            await self.hub.broadcast(msg, team=bot.team if team else None)
            first = first or msg
            now = time.time()
            self.last_spoke_global = now
            self.last_spoke[bot.clean] = now
        return {"tts_s": round(time.time() - t0, 2), **(first or {}), "text": text}

    @property
    def tts_streaming(self) -> bool:
        return bool(self.v.get("tts_streaming", True))

    @property
    def engine(self) -> str:
        return "qwen" if self.tts.__class__.__name__ == "QwenTTS" else "piper"

    async def synth(self, text: str, persona: Any, emotion: str) -> bytes:
        voice = persona.speaker if self.engine == "qwen" else persona.voice
        wav = await self.tts.synthesize(text, voice, emotion, style=persona.style,
                                        speed=float(getattr(persona, "speed", 1.0)))
        gain = float(getattr(persona, "volume", 1.0))
        return apply_gain(wav, gain) if abs(gain - 1.0) > 0.01 else wav

    async def preview(self, name: str, text: str, emotion: str = "neutral") -> str:
        """Synthesize `text` with the persona configured for `name` (connected or not) and return
        an /audio URL for the settings page to play locally."""
        persona = self.personas.for_name(name)
        wav = await self.synth(text, persona, emotion)
        return f"/audio/{self.audio.put(wav)}.wav"

    async def run(self) -> None:
        while True:
            job = await self.jobs.get()
            try:
                await self._process(job)
            except Exception:  # noqa: BLE001
                log.exception("job failed for %s", job.bot.clean)

    async def _process(self, job: Job) -> None:
        now = time.time()
        max_q = int(self.p.get("max_queue", 3))
        if not job.prompted:
            if now - job.created > 20 or self.jobs.qsize() >= max_q:
                return
            # thread replies skip the per-bot cooldown, otherwise bots could never answer each other
            if job.depth == 0 and now - self.last_spoke.get(job.bot.clean, 0) < float(self.p.get("unprompted_cooldown_s", 45)):
                return
        gap = float(self.p.get("global_min_gap_s", 6))
        wait = self.last_spoke_global + gap - now
        if wait > 0:
            if not job.prompted and job.depth == 0 and wait > gap / 2:
                return
            await asyncio.sleep(wait)
        if job.bot.slot not in self.state.players:
            return
        bot = self.state.players[job.bot.slot]
        persona = self.personas.for_name(bot.clean)
        system = self.llm.system_prompt(bot.clean, persona.persona, bot.team_name, bot.class_name,
                                        self.state.map, self.roster_text(), self.state.context_text(),
                                        rating=self.rating, activity=bot.activity_text(),
                                        map_brief=self.map_brief(bot.team_name), progress=self.state.progress_text(),
                                        minutes_left=self.state.minutes_left(), life=persona.life)
        t0 = time.time()
        emotion, text = await self.llm.reply(system, job.trigger)
        t_llm = time.time() - t0
        if not text:
            log.info("empty reply for %s", bot.clean)
            return
        res = await self.speak(bot, text, emotion=emotion)
        self._maybe_chain(bot, text, job)
        total = time.time() - job.created
        rec = {"llm_s": round(t_llm, 2), "tts_s": res["tts_s"], "total_s": round(total, 2)}
        self.timings.append(rec)
        self.timings = self.timings[-50:]
        log.info("%s [%s]: %r (llm %.2fs, tts %.2fs, total %.2fs)", bot.clean, emotion, text, t_llm, res["tts_s"], total)

    def _maybe_chain(self, bot: Player, text: str, job: Job) -> None:
        """After an unprompted line, sometimes a teammate bot answers, up to bot_thread_max lines."""
        if job.prompted or job.depth + 1 >= int(self.p.get("bot_thread_max", 3)):
            return
        if random.random() > float(self.p.get("bot_reply_prob", 0.7)):
            return
        mates = [b for b in self.voiced_bots(bot.team) if b.slot != bot.slot]
        if not mates or not self._has_audience(bot):
            return
        named = self.find_bot_in_text(text, bot)
        other = named if named and named.slot != bot.slot else random.choice(mates)
        if job.topic == "life":
            trigger = (f'Your teammate {bot.clean} just said over the team radio: "{text}". '
                       "Answer them in one friendly line about that, drawing on your own life outside the game.")
        else:
            trigger = (f'Your teammate {bot.clean} just said over the team radio: "{text}". '
                       "Answer them in one line that helps the team: agree on a plan, say what you yourself are doing "
                       "or just did, or what you need.")
        self.jobs.put_nowait(Job(other, trigger, False, time.time(), depth=job.depth + 1, topic=job.topic))
