(() => {
  const $ = (id) => document.getElementById(id);
  let snap = null;
  const dirty = new Set();          // bot names with an unsaved (debounced) edit; not overwritten by refreshes
  const timers = {};
  let audio = null;

  function setStatus(ok, text) { $("dot").classList.toggle("ok", ok); $("status").textContent = text; }

  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: body ? { "content-type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  function options(sel, values, current) {
    const vals = [...values];
    if (current && !vals.includes(current)) vals.unshift(current);
    sel.innerHTML = vals.map((v) => `<option value="${v}">${v}</option>`).join("") || `<option value="">(none)</option>`;
    sel.value = current || "";
  }

  // ----------------------------------------------------------------- bot cards
  function fillCard(card, bot) {
    card.querySelector(".f-voiced").checked = !!bot.voiced;
    options(card.querySelector(".f-speaker"), snap.speakers, bot.speaker);
    options(card.querySelector(".f-voice"), snap.piper_voices, bot.voice);
    card.querySelector(".f-speed").value = bot.speed; card.querySelector(".o-speed").textContent = bot.speed.toFixed(2) + "×";
    card.querySelector(".f-volume").value = bot.volume; card.querySelector(".o-volume").textContent = bot.volume.toFixed(1) + "×";
    card.querySelector(".f-style").value = bot.style || "";
    card.querySelector(".f-persona").value = bot.persona || "";
    card.querySelector(".f-life").value = bot.life || "";
  }

  function readCard(card) {
    return {
      voiced: card.querySelector(".f-voiced").checked,
      speaker: card.querySelector(".f-speaker").value,
      voice: card.querySelector(".f-voice").value,
      speed: parseFloat(card.querySelector(".f-speed").value),
      volume: parseFloat(card.querySelector(".f-volume").value),
      style: card.querySelector(".f-style").value,
      persona: card.querySelector(".f-persona").value,
      life: card.querySelector(".f-life").value,
    };
  }

  function decorate(card, bot) {
    card.dataset.name = bot.name;
    card.querySelector(".name").textContent = bot.name;
    const where = card.querySelector(".where");
    if (bot.name === "default") { where.hidden = true; card.querySelector(".reset").hidden = true; }
    else {
      where.textContent = bot.online ? `${bot.team} · ${bot.class}` : "offline";
      where.className = "where pill " + (bot.online ? (bot.team || "").toLowerCase() : "off");
      card.querySelector(".reset").hidden = !bot.override;
      card.classList.toggle("inherits", !bot.override);
    }
    card.classList.toggle("offline", bot.name !== "default" && !bot.online);
    card.classList.toggle("unvoiced", !bot.voiced);
  }

  function makeCard(bot) {
    const card = $("bot-card").content.firstElementChild.cloneNode(true);
    decorate(card, bot); fillCard(card, bot);
    const name = bot.name;
    const save = () => {
      clearTimeout(timers[name]);
      timers[name] = setTimeout(async () => {
        const saved = card.querySelector(".saved");
        try {
          saved.textContent = "saving…";
          const res = name === "default" ? await api("PUT", "/api/settings/default", readCard(card))
                                         : await api("PUT", `/api/settings/bot/${encodeURIComponent(name)}`, readCard(card));
          saved.textContent = "saved"; setTimeout(() => { if (saved.textContent === "saved") saved.textContent = ""; }, 1500);
          dirty.delete(name);
          const b = res.bot || res.default;
          decorate(card, { ...bot, ...b, override: true, name });
          card.classList.toggle("unvoiced", !b.voiced);
          if (name === "default") refresh();   // other cards may inherit from it
        } catch (e) { saved.textContent = "error: " + e.message; }
      }, 400);
    };
    card.addEventListener("input", (e) => {
      dirty.add(name);
      if (e.target.classList.contains("f-speed")) card.querySelector(".o-speed").textContent = (+e.target.value).toFixed(2) + "×";
      if (e.target.classList.contains("f-volume")) card.querySelector(".o-volume").textContent = (+e.target.value).toFixed(1) + "×";
      save();
    });
    card.querySelector(".test").onclick = async () => {
      const btn = card.querySelector(".test");
      // flush a pending edit first so the preview uses what is on screen
      clearTimeout(timers[name]);
      btn.disabled = true; btn.textContent = "…";
      try {
        const body = readCard(card);
        if (name === "default") await api("PUT", "/api/settings/default", body);
        else await api("PUT", `/api/settings/bot/${encodeURIComponent(name)}`, body);
        dirty.delete(name);
        const res = await api("POST", "/api/settings/preview", { bot: name, text: $("testtext").value, emotion: $("testemotion").value });
        if (audio) { audio.pause(); }
        audio = new Audio(res.audio_url); audio.play();
        btn.textContent = "▶ Test";
        decorate(card, { ...bot, override: true, name, voiced: body.voiced });
      } catch (e) { btn.textContent = "✗ " + e.message; setTimeout(() => (btn.textContent = "▶ Test"), 3000); }
      btn.disabled = false;
    };
    card.querySelector(".reset").onclick = async () => {
      await api("DELETE", `/api/settings/bot/${encodeURIComponent(name)}`);
      dirty.delete(name); refresh(true);
    };
    return card;
  }

  function render(force) {
    const host = $("bots"), dhost = $("default");
    const existing = new Map([...host.querySelectorAll(".bot")].map((c) => [c.dataset.name, c]));
    const keep = new Set();
    for (const bot of snap.bots) {
      keep.add(bot.name);
      let card = existing.get(bot.name);
      if (!card) { card = makeCard(bot); host.appendChild(card); }
      else { decorate(card, bot); if (force || !dirty.has(bot.name)) fillCard(card, bot); }
      host.appendChild(card);   // reorder to match the snapshot
    }
    for (const [n, c] of existing) if (!keep.has(n)) c.remove();
    if (!snap.bots.length) host.innerHTML = '<p class="small">No bots on the server and none configured. Add one by name above.</p>';
    const d = { ...snap.default, name: "default", online: true, override: true };
    let dcard = dhost.querySelector(".bot");
    if (!dcard) dhost.appendChild(makeCard(d));
    else if (force || !dirty.has("default")) fillCard(dcard, d);
    document.body.classList.toggle("engine-qwen", snap.engine === "qwen");
    document.body.classList.toggle("engine-piper", snap.engine === "piper");
    $("engine").textContent = snap.engine === "qwen" ? "engine: Qwen3-TTS" + (snap.tts_ready ? "" : " (loading)") : "engine: Piper";
    $("map").textContent = snap.map ? "· " + snap.map : "";
  }

  // ----------------------------------------------------------------- global
  function renderGlobal() {
    for (const sec of ["tts", "voice", "policy"]) {
      const host = $("g-" + sec);
      if (host.children.length) continue;            // build once; values are not refreshed under the user's hands
      for (const [key, spec] of Object.entries(snap.global_fields[sec])) {
        const val = snap.global[sec][key];
        const lab = document.createElement("label");
        let input;
        if (spec.type === "bool") { input = document.createElement("input"); input.type = "checkbox"; input.checked = !!val; }
        else if (spec.type === "enum") { input = document.createElement("select"); options(input, spec.choices, val); }
        else { input = document.createElement("input"); input.type = "number"; input.value = val ?? ""; input.step = spec.type === "int" ? 1 : 0.05; if (spec.min != null) input.min = spec.min; if (spec.max != null) input.max = spec.max; }
        input.dataset.sec = sec; input.dataset.key = key;
        lab.append(input, " ", key.replaceAll("_", " "));
        host.appendChild(lab);
        input.addEventListener("change", async () => {
          const v = spec.type === "bool" ? input.checked : input.value;
          lab.classList.remove("ok", "err");
          try { await api("PUT", "/api/settings/global", { [sec]: { [key]: v } }); lab.classList.add("ok"); }
          catch (e) { lab.classList.add("err"); lab.title = e.message; }
        });
      }
    }
  }

  async function refresh(force) {
    try {
      snap = await api("GET", "/api/settings");
      setStatus(true, `${snap.bots.filter((b) => b.online).length} bots online`);
      if (!$("testemotion").children.length) options($("testemotion"), snap.emotions, "neutral");
      render(force); renderGlobal();
    } catch (e) { setStatus(false, "hub unreachable"); }
  }

  $("addbot").onclick = async () => {
    const n = $("newbot").value.trim();
    if (!n) return;
    await api("PUT", `/api/settings/bot/${encodeURIComponent(n)}`, {});
    $("newbot").value = ""; refresh(true);
  };
  $("newbot").addEventListener("keydown", (e) => { if (e.key === "Enter") $("addbot").click(); });

  refresh(true);
  setInterval(() => refresh(false), 5000);
})();
