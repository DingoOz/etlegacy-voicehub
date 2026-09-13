(() => {
  const $ = (id) => document.getElementById(id);
  const feed = $("feed"), ptt = $("ptt"), dot = $("dot"), statusEl = $("status"), levelEl = $("level");
  let ws, state = { players: [] }, voice = {};
  let queue = [], playing = false, muted = false, volume = 1, playingUntil = 0;

  const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                  set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };

  function setStatus(ok, text) { dot.classList.toggle("ok", ok); statusEl.textContent = text; }
  function mySlot() { const v = $("who2").value; return v === "" ? null : parseInt(v, 10); }
  function me() { const s = mySlot(); return s === null ? null : state.players.find((p) => p.slot === s) || null; }
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
  function report(st, detail) { send({ type: "status", state: st, detail: detail || "" }); }

  function addLine(cls, who, text) {
    const div = document.createElement("div");
    div.className = "line " + cls;
    div.innerHTML = `<span class="who"></span><span class="text"></span>`;
    div.querySelector(".who").textContent = who + ":";
    div.querySelector(".text").textContent = text;
    feed.appendChild(div);
    while (feed.children.length > 60) feed.removeChild(feed.firstChild);
    feed.scrollTop = feed.scrollHeight;
    return div;
  }

  function fillWho() {
    const humans = state.players.filter((p) => !p.bot);
    const saved = store.get("who") || "";
    for (const sel of [$("who"), $("who2")]) {
      const cur = sel.value || saved;
      sel.innerHTML = '<option value="">(just listening)</option>' +
        humans.map((p) => `<option value="${p.slot}">${p.name} (${p.team})</option>`).join("");
      if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
    }
    const m = me(), mine = m ? m.team_id : 0;
    const bots = state.players.filter((p) => p.bot && (!voice.team_only || !mine || mine === 3 || p.team_id === mine));
    $("bots").innerHTML = bots.map((p) => `<li>${p.name} — ${p.team}, ${p.class}</li>`).join("") || "<li>no bots on your team</li>";
    $("map").textContent = state.map ? "· " + state.map : "";
    const t = $("team");
    t.textContent = m ? (voice.team_only ? `${m.team} channel` : "all teams") : "listening to everyone";
    t.className = "pill " + (m ? m.team.toLowerCase() : "");
  }

  // ------------------------------------------------------------ playback
  async function playNext() {
    if (playing || queue.length === 0) return;
    playing = true;
    const { url, line } = queue.shift();
    try {
      if (!muted) {
        const a = new Audio(url); a.volume = volume;
        line.classList.add("speaking");
        playingUntil = Infinity;
        await a.play();
        await new Promise((res) => { a.onended = res; a.onerror = res; });
        line.classList.remove("speaking");
      }
    } catch (e) { console.warn("play failed", e); }
    playingUntil = performance.now() + 400;  // ignore the mic briefly: speakers may still ring
    playing = false;
    playNext();
  }

  // ------------------------------------------------------------ capture
  // Raw PCM from the mic -> 16 kHz WAV. Three ways to delimit an utterance:
  //   hold : button / space held down
  //   ptt  : in-game key pressed; ends after `silence_ms` of quiet (or max length, or second press)
  //   open : always listening; every spoken phrase is sent when the speaker pauses
  const cap = {
    ctx: null, ready: false, mode: "idle",   // idle | hold | ptt | open
    pre: [], preLen: 0, seg: [], segLen: 0, speaking: false, lastVoice: 0, started: 0,
    floor: 0.004, rate: 16000, holdBuf: null,
  };
  const SR = 16000;

  function downsample(f32, from) {
    if (from === SR) return f32;
    const ratio = from / SR, n = Math.floor(f32.length / ratio), out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * ratio, j = Math.floor(p), t = p - j;
      out[i] = f32[j] * (1 - t) + (f32[Math.min(j + 1, f32.length - 1)] || 0) * t;
    }
    return out;
  }

  function wav(chunks, total) {
    const buf = new ArrayBuffer(44 + total * 2), dv = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); dv.setUint32(4, 36 + total * 2, true); str(8, "WAVE"); str(12, "fmt ");
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, SR, true); dv.setUint32(28, SR * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    str(36, "data"); dv.setUint32(40, total * 2, true);
    let o = 44;
    for (const c of chunks) for (let i = 0; i < c.length; i++, o += 2) dv.setInt16(o, Math.max(-1, Math.min(1, c[i])) * 32767, true);
    return new Blob([buf], { type: "audio/wav" });
  }

  function rms(f32) { let s = 0; for (let i = 0; i < f32.length; i++) s += f32[i] * f32[i]; return Math.sqrt(s / f32.length); }

  function onAudio(f32) {
    const now = performance.now();
    const level = rms(f32);
    const thr = Math.max(cap.floor * 3.5, 0.012);
    const loud = level > thr;
    if (!loud) cap.floor = cap.floor * 0.98 + level * 0.02;    // slow noise-floor tracker
    levelEl.classList.toggle("hot", loud);
    levelEl.firstElementChild.style.width = Math.min(100, level * 800) + "%";

    if (cap.mode === "idle") return;
    if (cap.mode === "hold") { cap.seg.push(f32); cap.segLen += f32.length; return; }

    // ptt / open: keep 0.4 s of pre-roll so the first syllable is not lost
    cap.pre.push(f32); cap.preLen += f32.length;
    while (cap.preLen > SR * 0.4) cap.preLen -= cap.pre.shift().length;

    const silenceMs = +voice.silence_ms || 900, maxS = +voice.max_utterance_s || 15;
    const ducked = now < playingUntil;              // a bot is talking out of the speakers
    if (!cap.speaking) {
      if (loud && !ducked) {
        cap.speaking = true; cap.lastVoice = now; cap.started = now;
        cap.seg = cap.pre.slice(); cap.segLen = cap.preLen;
        ptt.classList.add("rec"); ptt.textContent = "Listening…";
      } else if (cap.mode === "ptt" && now - cap.started > (+voice.ptt_timeout_s || 5) * 1000) {
        setMode("idle"); report("timeout");
      }
      return;
    }
    cap.seg.push(f32); cap.segLen += f32.length;
    if (loud) cap.lastVoice = now;
    if (now - cap.lastVoice > silenceMs || now - cap.started > maxS * 1000) finishSegment();
  }

  function finishSegment() {
    const chunks = cap.seg, total = cap.segLen, mode = cap.mode;
    cap.seg = []; cap.segLen = 0; cap.speaking = false;
    if (mode !== "open") setMode("idle"); else refreshButton();
    if (total < SR * 0.3) { if (mode === "ptt") report("timeout"); return; }
    upload(wav(chunks, total), mode);
  }

  async function upload(blob, mode) {
    ptt.classList.add("busy"); if (cap.mode !== "open") ptt.textContent = "Sending…";
    const fd = new FormData(); fd.append("audio", blob, "clip.wav"); fd.append("speaker_slot", $("who2").value); fd.append("mode", mode);
    try {
      const r = await fetch("/api/utterance", { method: "POST", body: fd });
      const j = await r.json();
      if (!j.text && mode !== "open") addLine("me", "you", "(nothing heard)");
    } catch (e) { addLine("me", "error", e.message); }
    ptt.classList.remove("busy"); refreshButton();
  }

  function refreshButton() {
    ptt.classList.remove("rec", "armed", "open");
    if (cap.mode === "open") { ptt.classList.add("open"); ptt.textContent = cap.speaking ? "Listening…" : "Open mic — talk any time"; }
    else if (cap.mode === "ptt") { ptt.classList.add("armed"); ptt.textContent = "Listening… (stops when you stop talking)"; }
    else if (cap.mode === "hold") { ptt.classList.add("rec"); ptt.textContent = "Listening… release to send"; }
    else ptt.textContent = "Hold to talk";
  }

  function setMode(m) {
    if (!cap.ready) { if (m !== "idle") report("nomic"); return; }
    if (cap.speaking && m !== cap.mode) finishSegment();
    cap.mode = m; cap.speaking = false; cap.seg = []; cap.segLen = 0; cap.started = performance.now();
    $("openmic").checked = (m === "open");
    refreshButton();
  }

  // In-game key (via server): talk = start listening / send now; mic = toggle open mic.
  function gameCmd(cmd, arg) {
    if (cmd === "talk") {
      if (cap.mode === "ptt" || cap.mode === "hold") { if (cap.speaking || cap.mode === "hold") finishSegment(); else { setMode("idle"); } }
      else if (cap.mode === "open") { if (cap.speaking) finishSegment(); }
      else { setMode("ptt"); report("listening"); }
    } else if (cmd === "mic") {
      const on = arg === "on" ? true : arg === "off" ? false : cap.mode !== "open";
      setMode(on ? "open" : "idle"); store.set("openmic", on ? "1" : "0"); report(on ? "mic_on" : "mic_off");
    } else if (cmd === "stop") { setMode("idle"); report("mic_off"); }
  }

  async function initCapture() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    await ctx.resume();
    cap.ctx = ctx; cap.rate = ctx.sampleRate;
    const src = ctx.createMediaStreamSource(stream);
    const handle = (f32) => onAudio(downsample(f32, cap.rate));
    try {
      await ctx.audioWorklet.addModule("/static/capture-worklet.js");
      const node = new AudioWorkletNode(ctx, "capture", { numberOfOutputs: 0 });
      node.port.onmessage = (e) => handle(e.data);
      src.connect(node);
    } catch (e) {
      console.warn("worklet unavailable, using ScriptProcessor", e);
      const node = ctx.createScriptProcessor(2048, 1, 1);
      node.onaudioprocess = (e) => handle(e.inputBuffer.getChannelData(0).slice());
      src.connect(node); node.connect(ctx.destination);
    }
    cap.ready = true;
  }

  // ------------------------------------------------------------ network
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { setStatus(true, "connected"); send({ type: "hello", slot: mySlot() }); };
    ws.onclose = () => { setStatus(false, "reconnecting"); setTimeout(connect, 2000); };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "state") { state = msg.state; if (msg.voice) voice = msg.voice; fillWho(); }
      else if (msg.type === "settings") { voice = msg.voice; fillWho(); }
      else if (msg.type === "speech") {
        const line = addLine("bot", `[BOT] ${msg.bot}`, msg.text);
        queue.push({ url: msg.audio_url, line }); playNext();
      } else if (msg.type === "voice") {
        const line = addLine("voice", msg.speaker, msg.text);
        queue.push({ url: msg.audio_url, line }); playNext();
      } else if (msg.type === "transcript") addLine("me", msg.speaker, msg.text);
      else if (msg.type === "cmd") gameCmd(msg.cmd, msg.arg);
    };
    setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 20000);
  }

  async function join() {
    $("setup").hidden = true; $("main").hidden = false;
    $("who2").value = $("who").value; store.set("who", $("who").value);
    send({ type: "hello", slot: mySlot() }); fillWho();
    try {
      await initCapture();
      $("hint").textContent = "Hold the button or the space bar while you speak. Say a bot's name to talk to that bot.";
      const wantOpen = store.get("openmic") === null ? !!voice.open_mic_default : store.get("openmic") === "1";
      if (wantOpen) setMode("open");
    } catch (e) {
      $("hint").textContent = "Microphone unavailable (" + e.message + "). You can still listen.";
      ptt.disabled = true; $("openmic").disabled = true;
    }
  }

  function holdStart(e) { e.preventDefault(); if (!cap.ready || cap.mode !== "idle") return; setMode("hold"); }
  function holdStop(e) { if (e) e.preventDefault(); if (cap.mode === "hold") finishSegment(); }

  $("join").onclick = join;
  ptt.addEventListener("pointerdown", (e) => { if (cap.mode === "ptt") finishSegment(); else if (cap.mode === "open") { if (cap.speaking) finishSegment(); } else holdStart(e); });
  ptt.addEventListener("pointerup", holdStop);
  ptt.addEventListener("pointercancel", holdStop);
  ptt.addEventListener("pointerleave", (e) => holdStop(e));
  document.addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat && !$("main").hidden && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT") holdStart(e); });
  document.addEventListener("keyup", (e) => { if (e.code === "Space" && !$("main").hidden) holdStop(e); });
  $("openmic").onchange = (e) => { setMode(e.target.checked ? "open" : "idle"); store.set("openmic", e.target.checked ? "1" : "0"); };
  $("who2").onchange = () => { store.set("who", $("who2").value); send({ type: "hello", slot: mySlot() }); fillWho(); };
  $("vol").oninput = (e) => { volume = parseFloat(e.target.value); };
  $("mute").onchange = (e) => { muted = e.target.checked; };

  connect();
  fetch("/api/state").then((r) => r.json()).then((s) => { state = s; voice = s.voice || {}; fillWho(); }).catch(() => {});
})();
