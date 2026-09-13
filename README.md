# voicehub

Spoken, LLM-driven Omni-bots for the LAN ET: Legacy server.

```
players' browsers  <-- https + WebSocket -->  voicehub (this daemon)  <-- files -->  game server (Lua module)
   mic / in-game key                        whisper -> Ollama -> Qwen3-TTS        legacy/luascripts/voicehub/main.lua
```

* Players open **https://192.168.1.79:8443** on a PC or phone, accept the self-signed
  certificate once and pick their in-game name. Three ways to talk:
  * hold the big button (or the space bar) on the page;
  * press a key **in the game** (see below): the page starts listening and sends when you
    stop talking;
  * *Always listening* (open mic): every phrase is sent when you pause.
* What they say is transcribed, shown in-game as their **team** chat, and answered by a bot
  on their team, in character. Bot lines appear as `[BOT]Name:` team chat and are spoken
  only on the pages of that team (spectators hear everyone; `[voice]` in `config.toml`).
* Bots also comment on their own on kills, revives, new players and map starts, and chip in
  at random when it is quiet (`idle_chatter_interval_s`). Probabilities, cooldowns and the
  NSFW `rating` are in `[policy]`.
* `personas.toml` gives each bot name a personality and a Piper voice.

## In-game keys

Keep the voice page open in the background with your name picked, then in the game
console (`~`) once:

```
bind v "cmd vh_talk"      // press: listen until you stop talking; press again: send now
bind b "cmd vh_mic"       // toggle always-listening on the page
```

`vh_stop` turns the mic off and `vh_status` shows how many pages are joined as you.
Feedback appears as a popup in the game's chat area (`voice: listening...`). If no page has
picked your name, the game tells you the page URL.

## Settings

`config.toml` sections `[policy]` and `[voice]` are re-read a few seconds after saving; no
restart needed. Highlights:

| key | meaning |
|---|---|
| `policy.rating` | `family`, `pg13`, `r` or `explicit` |
| `policy.idle_chatter_interval_s` | average seconds between spontaneous bot lines, `0` = off |
| `policy.idle_chatter_jitter` | randomness of that interval (0.5 = 50%..150%) |
| `policy.open_mic_reply_prob` | chance a bot answers open-mic speech that did not name a bot |
| `policy.bot_reply_prob`, `bot_thread_max` | bots answering each other about what they are doing |
| `tts.speed`, `tts.emotion` | playback rate (1.15 = 15% faster) and whether emotion tags shape delivery |
| `voice.team_only` | keep speech, replies and audio within a team |
| `voice.spectators_hear_all` | spectators and name-less pages hear every team |
| `voice.relay_human_voice` | also play a player's recording to teammates' pages |
| `voice.silence_ms`, `max_utterance_s`, `ptt_timeout_s` | when an utterance ends |

## Voices

`[tts] engine = "qwen"` uses **Qwen3-TTS 1.7B CustomVoice** on the GPU: two English male presets
(Ryan, Aiden), a per-bot `style` sentence (age, attitude) and an `[emotion]` tag the LLM puts in
front of every line ("[excited] Yes! Got him!"), all passed as the instruct text. `engine = "piper"`
is the old CPU path (instant, flat delivery; emotion only nudges rate and loudness).

Qwen3-TTS through stock HuggingFace runs at ~3.8x slower than real time on this box (Pascal GPUs,
2.4 GHz Xeon: the decode loop is launch-bound). `voicehub/fast_qwen.py` re-implements the decode
step with a static KV cache, fused kernels and one CUDA graph per audio frame, which brings it to
~0.7x real time (a 4 s sentence renders in under 3 s). Bot lines are streamed sentence by sentence,
so the first sentence plays while the rest renders. Set `qwen_fast = false` to compare with stock.

Notes: PyTorch is pinned to the CUDA 12.6 wheels (`[tool.uv.sources]` in `pyproject.toml`), the
last builds with kernels for sm_60/61. The 0.6B model overflows in float16; use float32 for it.
`bench/` holds the equivalence and speed scripts used to validate the fast path;
`samples/` has example WAVs.

## Running

```
cd voicehub
uv sync                                   # once; installs Python 3.12 venv
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice              # ~4.3 GB, once
uv run python -m piper.download_voices --data-dir voices en_US-lessac-medium   # only for engine = "piper"
uv run python -m voicehub                 # or: systemctl --user enable --now voicehub
```

The systemd unit is `voicehub.service` (copy or link it into `~/.config/systemd/user/`).
Copy `luascripts/voicehub/main.lua` into the server's `legacy/luascripts/voicehub/` and list
`luascripts/voicehub/main.lua` in `lua_modules` (e.g. in `etmain/legacy.cfg`).

## Endpoints

* `GET /api/health` – models, Ollama, rcon, roster, recent timings
* `GET /api/state` – map and players
* `POST /api/utterance` – multipart audio + `speaker_slot` + `mode` (`hold`/`ptt`/`open`)
* `POST /api/ask?text=...&bot=Name&speaker=Name` – text-only test of the LLM path
* `POST /api/test_speak?bot=Name&text=...` – speak a fixed line (TTS + in-game chat)

## Notes

* The mic needs a secure context, hence HTTPS with a self-signed certificate
  (`certs/`, regenerated if deleted; SANs in `config.toml`). `http://localhost:8081`
  also works on the server itself.
* Whisper runs on the GPU given by `[stt] device_index` and falls back to the other
  GPU, then CPU int8, if it cannot allocate memory.
* Ollama loading takes about a minute; the hub warms the model at start and keeps it
  resident (`keep_alive = -1`).

## License

GPL-3.0-or-later. See `LICENSE`.
