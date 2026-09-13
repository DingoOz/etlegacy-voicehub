# voicehub

Spoken, LLM-driven Omni-bots for an ET: Legacy LAN server. Players talk to the bots with a
microphone, the bots talk back in their own voices, chat among themselves about the map and
their lives, and take simple orders ("Walter, stick close").

```
players' browsers  <-- https + WebSocket -->  voicehub (this daemon)  <-- files + rcon -->  game server
   mic / in-game key                    whisper -> Ollama -> Qwen3-TTS / Piper         Lua bridge + Omni-bot goal
```

## Features

* **Talk to bots.** Hold a button on the voice page, press a key in the game, or leave the mic
  open. Speech is transcribed (faster-whisper), shown in-game as your team chat, and answered by
  a bot on your team, in character, spoken on your teammates' pages.
* **Bots that know the game.** Each prompt carries a primer on ET (classes, respawn waves,
  attack/defend), the current map's mission and objectives read from the game's own pk3 files,
  the objectives completed so far (from the game's announcements), the clock, and the roster.
* **Bots that play together.** They comment on kills, revives, joins, map starts and objective
  events, and chip in when it is quiet: mostly tactics and coordination, sometimes small talk
  about their lives (pets, jobs, the vet visit). Tone follows a rating: constructive teammates at
  `family`/`pg13`, trash talk only at `r`/`explicit`.
* **Voice orders.** "Walter, follow me", "everyone hold here", "Merki, carry on": an Omni-bot goal
  script makes the bot do it and the bot acknowledges out loud.
* **Distinct voices.** Qwen3-TTS with nine preset speakers, a per-bot style sentence and an
  emotion tag chosen by the LLM per line, rendered fast enough on old GPUs through a CUDA-graph
  decode loop. Piper as a CPU fallback.
* **Browser settings page.** Per-bot voice, style, personality, life notes, speed, volume and a
  Test button; global chatter and channel knobs. Saves to the TOML files and applies live.

## Requirements

* ET: Legacy dedicated server with Omni-bot, with rcon enabled.
* Ollama serving a chat model (default `qwen3.5:4b`).
* Python 3.12 and [uv](https://docs.astral.sh/uv/); `ffmpeg` on the path.
* A CUDA GPU for Whisper and Qwen3-TTS (tested on Pascal: Tesla P100 / GTX 1070), or Piper on CPU.

## Install

```
git clone https://github.com/DingoOz/etlegacy-voicehub voicehub
cd voicehub
uv sync                                                             # Python 3.12 venv
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice             # ~4.3 GB, once
uv run python -m piper.download_voices --data-dir voices en_US-lessac-medium   # only for engine = "piper"
```

Then wire up the game server (paths relative to the server's `legacy/` mod directory):

1. **Lua bridge**: copy `luascripts/voicehub/main.lua` to `legacy/luascripts/voicehub/main.lua`
   and list `luascripts/voicehub/main.lua` in `lua_modules` (e.g. in `etmain/legacy.cfg`). It
   exports game events to `<homepath>/legacy/voicehub/events.jsonl` and executes the hub's
   commands from `inbox.jsonl`.
2. **Omni-bot goal** (voice orders): copy `omnibot/goals/goal_vhorders.gm` to
   `<omni-bot path>/et/scripts/goals/`. Omni-bot loads new scripts on a full map load
   (`map oasis`), not on `map_restart`.
3. **config.toml**: set `[game] homepath` to the server's homepath mod dir, `rcon_password_file`
   (or `rcon_password`), and `[web] cert_sans` / `public_url` to the LAN address players will use.
   `pk3_dirs` lists where map pk3s live (for the map briefings).

Run it:

```
uv run python -m voicehub                 # or: systemctl --user enable --now voicehub
```

The systemd unit is `voicehub.service` (link it into `~/.config/systemd/user/`; edit the paths).
Models load concurrently at start; `GET /api/health` shows progress. First start generates a
self-signed certificate in `certs/`.

## Playing

Open **https://\<server\>:8443** on a PC or phone, accept the certificate once, pick your in-game
name. Three ways to talk:

* hold the big button (or the space bar);
* press a key **in the game**: the page listens and sends when you stop talking;
* *Always listening* (open mic): every phrase is sent when you pause. Bots answer open-mic speech
  when addressed by name, or now and then (`policy.open_mic_reply_prob`).

For the in-game key, keep the page open in the background and bind once in the console (`~`):

```
bind v "cmd vh_talk"      // press: listen until you stop talking; press again: send now
bind b "cmd vh_mic"       // toggle always-listening on the page
```

`vh_stop` turns the mic off, `vh_status` shows how many pages are joined as you. Feedback pops up
in the game's chat area (`voice: listening...`); if no page has picked your name, the game tells
you the page URL.

Say a bot's name to talk to that bot. Typed team chat that names a bot (or says "bots") gets an
answer too, and the Omni-bot chat phrases (`bot come`, `bot go`, ...) still work.

### Voice orders

Say a bot's name and an order, by voice or in team chat:

| you say | the bot |
|---|---|
| "Walter, stick close" / "follow me" / "stay with me" / "cover me" | follows you within `policy.follow_radius_m` (25 m) |
| "Walter, follow me, stay within 10 metres" | same with that radius |
| "Walter, hold here" / "stay here" / "wait here" / "stop" | holds your current spot, facing your way |
| "Walter, carry on" / "as you were" / "do your thing" | back to normal play |
| "everyone stick close" / "bots hold here" | every voiced bot on your team, up to `policy.max_followers` |

Combat, aiming and class duties keep working while a bot follows; orders survive death and
respawn and end on map change or when you leave. `policy.orders = false` turns the feature off.
From rcon: `bot vh status`, `bot vh debug 1`, `bot vh release all`, and `bot vh wake 1` makes bots
play with no human on the server (handy for testing; `wake 0` restores normal sleeping).

## Settings page

**https://\<server\>:8443/settings** (the ⚙ link on the voice page) lists every bot the game
reports, online first with team and class, plus configured-but-offline personas. Per bot: voiced
on/off, Qwen speaker or Piper voice, voice style sentence, personality, life notes (for small
talk), speed multiplier, volume gain, and **Test** to hear any line with any emotion right in the
browser. Bots without an entry follow **Default**; "use default" drops a bot's overrides; *Add
bot* creates an entry for a bot not on the server right now. The Global section covers the
hot-reloadable `[tts]`, `[policy]` and `[voice]` keys.

Everything saves as you change it and applies to the next line spoken: bots to `personas.toml`
(rewritten, header comment kept), globals to `config.toml` (edited in place, comments kept). Hand
edits to `config.toml`, `personas.toml` and `maps.toml` are picked up within a few seconds; only
switching the TTS engine needs a restart.

Personas are kept safe: saves are atomic and fsynced, the previous file goes to `backups/` (newest
20 kept), and a missing, empty or corrupt `personas.toml` at startup is restored from the newest
good backup (the bad file is kept as `personas.toml.corrupt`).

## Configuration files

* `config.toml`: `[web]`, `[game]`, `[stt]`, `[llm]`, `[tts]`, `[policy]`, `[voice]`. Comments in
  the file describe every key. The main knobs:

  | key | meaning |
  |---|---|
  | `policy.rating` | `family`, `pg13`, `r` or `explicit`; sets tone as well as language |
  | `policy.idle_chatter_interval_s`, `idle_chatter_jitter` | spontaneous bot lines, `0` = off |
  | `policy.chitchat_prob` | share of spontaneous lines that are small talk (rest: game talk) |
  | `policy.objective_prob` | chance a bot per side reacts to an objective event |
  | `policy.bot_reply_prob`, `bot_thread_max` | bots answering each other |
  | `policy.orders`, `follow_radius_m`, `max_followers` | voice orders |
  | `tts.engine`, `speed`, `emotion` | `qwen` or `piper`; playback rate; emotion-driven delivery |
  | `voice.team_only`, `spectators_hear_all`, `relay_human_voice` | who hears what |
  | `voice.silence_ms`, `max_utterance_s`, `ptt_timeout_s` | when an utterance ends |

* `personas.toml`: per-bot `voiced`, `voice` (Piper), `speaker` (Qwen), `style`, `persona`,
  `life`, `speed`, `volume`, with a `[default]` for bots without an entry.
* `maps.toml`: per-map `tips` for the bots on top of the objectives read from the pk3s; add
  custom maps here. Six stock maps included.

## Voices

`[tts] engine = "qwen"` uses **Qwen3-TTS 1.7B CustomVoice** on the GPU: nine preset speakers that
all speak English (male: aiden, dylan, eric, ryan, uncle_fu; female: ono_anna, serena, sohee,
vivian), a per-bot `style` sentence (age, attitude) and an `[emotion]` tag the LLM puts in front
of every line ("[excited] Yes! Got him!"), all passed as the instruct text. `engine = "piper"` is
the CPU path (instant, flat delivery; emotion only nudges rate and loudness).

Stock Qwen3-TTS through HuggingFace runs about 3.8x slower than real time on Pascal GPUs with a
2.4 GHz Xeon (the decode loop is launch-bound). `voicehub/fast_qwen.py` re-implements the decode
step with a static KV cache, fused kernels and one CUDA graph per audio frame, reaching about
0.7x real time (a 4 s sentence renders in under 3 s). Bot lines stream sentence by sentence, so
the first sentence plays while the rest renders. `qwen_fast = false` falls back to the stock loop.

PyTorch is pinned to the CUDA 12.6 wheels (`[tool.uv.sources]` in `pyproject.toml`), the last
builds with kernels for sm_60/61. The 0.6B model overflows in float16; use float32 for it.
`bench/` holds the equivalence and speed scripts that validated the fast path; `samples/` has
example WAVs.

## HTTP API

* `GET /api/health`: models, Ollama, rcon, roster, recent timings
* `GET /api/state`: map and players
* `POST /api/utterance`: multipart audio + `speaker_slot` + `mode` (`hold`/`ptt`/`open`)
* `POST /api/ask?text=...&bot=Name&speaker=Name`: text-only test of the LLM path
* `POST /api/test_speak?bot=Name&text=...`: speak a fixed line (TTS + in-game chat)
* `GET /api/settings`, `PUT /api/settings/bot/{name}`, `DELETE /api/settings/bot/{name}`,
  `PUT /api/settings/default`, `PUT /api/settings/global` (`{"policy": {"rating": "r"}}`),
  `POST /api/settings/preview` (`{"bot": "Walter", "text": "...", "emotion": "angry"}` → `audio_url`)

## Repository layout

```
voicehub/            the daemon: web.py, brain.py, llm.py, stt.py, tts.py, tts_qwen.py,
                     fast_qwen.py, maps.py, orders.py, personas.py, settings.py, game_events.py
static/              voice page and settings page
luascripts/voicehub/ Lua bridge for the game server
omnibot/goals/       Omni-bot goal script for voice orders
bench/, samples/     TTS fast-path validation and example audio
```

## Notes

* The mic needs a secure context, hence HTTPS with a self-signed certificate (`certs/`,
  regenerated if deleted; SANs in `config.toml`). `http://localhost:8081` also works on the
  server itself.
* Whisper runs on the GPU given by `[stt] device_index` and falls back to the other GPU, then
  CPU int8, if it cannot allocate memory.
* Ollama loading takes about a minute; the hub warms the model at start and keeps it resident
  (`keep_alive = -1`).
* Bots sleep when no human is on the server (Omni-bot `SleepBots`); the hub only makes bots talk
  when a human can hear them.

## License

GPL-3.0-or-later. See `LICENSE`.
