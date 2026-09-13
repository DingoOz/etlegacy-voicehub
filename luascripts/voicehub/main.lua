-- voicehub: bridge between the ET: Legacy game server and the voicehub daemon.
--
-- Exports game events (roster, chat, voice macros, kills, revives, spawns, map
-- changes) as JSON lines to  <fs_homepath>/<fs_game>/voicehub/events.jsonl  and
-- executes commands appended by the daemon to  .../voicehub/inbox.jsonl
-- (chat on behalf of a client slot, allow-listed console commands).
--
-- The only client commands this module swallows are its own "vh_*" voice
-- commands (bound in-game, e.g.  bind v "cmd vh_talk"); everything else falls
-- through, so it can be listed before WolfAdmin in lua_modules.

local VERSION = "0.5"
local POLL_MS = 100
local MAX_TEXT = 150
local CONSOLE_ALLOW = { "playsound ", "bot " }

local HOME, EVENTS_PATH, INBOX_PATH
local events_fh
local inbox_offset = 0
local last_announce, last_announce_lt = nil, 0
local last_poll = 0
local level_time = 0

-------------------------------------------------------------------------------
-- Minimal JSON (encode + decode), enough for flat command/event records.
-------------------------------------------------------------------------------
local json = {}

local escapes = { ['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b', ['\f'] = '\\f',
                  ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }

local function encode_string(s)
    return '"' .. s:gsub('[%c"\\]', function(c)
        return escapes[c] or string.format('\\u%04x', c:byte())
    end) .. '"'
end

local function is_array(t)
    local n = 0
    for _ in pairs(t) do n = n + 1 end
    return n == #t
end

function json.encode(v)
    local tv = type(v)
    if tv == "nil" then return "null"
    elseif tv == "boolean" then return v and "true" or "false"
    elseif tv == "number" then
        if v ~= v or v == math.huge or v == -math.huge then return "null" end
        if math.type(v) == "integer" then return tostring(v) end
        return string.format("%.14g", v)
    elseif tv == "string" then return encode_string(v)
    elseif tv == "table" then
        local out = {}
        if is_array(v) and #v > 0 then
            for i = 1, #v do out[i] = json.encode(v[i]) end
            return "[" .. table.concat(out, ",") .. "]"
        end
        local keys = {}
        for k in pairs(v) do keys[#keys + 1] = tostring(k) end
        table.sort(keys)
        for _, k in ipairs(keys) do
            out[#out + 1] = encode_string(k) .. ":" .. json.encode(v[k])
        end
        return "{" .. table.concat(out, ",") .. "}"
    end
    return "null"
end

local decode_value

local function skip_ws(s, i)
    return s:find("[^ \t\r\n]", i) or (#s + 1)
end

local function decode_string(s, i)
    -- s[i] == '"'
    local buf = {}
    i = i + 1
    while i <= #s do
        local c = s:sub(i, i)
        if c == '"' then return table.concat(buf), i + 1 end
        if c == "\\" then
            local n = s:sub(i + 1, i + 1)
            local map = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t", ['"'] = '"', ["\\"] = "\\", ["/"] = "/" }
            if n == "u" then
                local hex = s:sub(i + 2, i + 5)
                local cp = tonumber(hex, 16) or 63
                buf[#buf + 1] = utf8.char(cp)
                i = i + 6
            else
                buf[#buf + 1] = map[n] or n
                i = i + 2
            end
        else
            buf[#buf + 1] = c
            i = i + 1
        end
    end
    error("unterminated string")
end

function decode_value(s, i)
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == "{" then
        local obj = {}
        i = skip_ws(s, i + 1)
        if s:sub(i, i) == "}" then return obj, i + 1 end
        while true do
            i = skip_ws(s, i)
            if s:sub(i, i) ~= '"' then error("expected key at " .. i) end
            local k; k, i = decode_string(s, i)
            i = skip_ws(s, i)
            if s:sub(i, i) ~= ":" then error("expected ':' at " .. i) end
            local v; v, i = decode_value(s, i + 1)
            obj[k] = v
            i = skip_ws(s, i)
            local d = s:sub(i, i)
            if d == "}" then return obj, i + 1 end
            if d ~= "," then error("expected ',' at " .. i) end
            i = i + 1
        end
    elseif c == "[" then
        local arr = {}
        i = skip_ws(s, i + 1)
        if s:sub(i, i) == "]" then return arr, i + 1 end
        while true do
            local v; v, i = decode_value(s, i)
            arr[#arr + 1] = v
            i = skip_ws(s, i)
            local d = s:sub(i, i)
            if d == "]" then return arr, i + 1 end
            if d ~= "," then error("expected ',' at " .. i) end
            i = i + 1
        end
    elseif c == '"' then
        return decode_string(s, i)
    elseif s:sub(i, i + 3) == "true" then return true, i + 4
    elseif s:sub(i, i + 4) == "false" then return false, i + 5
    elseif s:sub(i, i + 3) == "null" then return nil, i + 4
    else
        local num = s:match("^-?%d+%.?%d*[eE]?[-+]?%d*", i)
        if not num or num == "" then error("unexpected char at " .. i) end
        return tonumber(num), i + #num
    end
end

function json.decode(s)
    local ok, v = pcall(function() return (decode_value(s, 1)) end)
    if ok then return v end
    return nil, v
end

-------------------------------------------------------------------------------
-- Helpers
-------------------------------------------------------------------------------
local function log(msg)
    et.G_Print("voicehub: " .. msg .. "\n")
end

local function emit(ev, fields)
    if not events_fh then return end
    fields = fields or {}
    fields.ev = ev
    fields.t = os.time()
    fields.lt = level_time
    events_fh:write(json.encode(fields), "\n")
    events_fh:flush()
end

local function max_clients()
    return tonumber(et.trap_Cvar_Get("sv_maxclients")) or 64
end

local function gfield(i, name)
    local ok, v = pcall(et.gentity_get, i, name)
    if ok then return tonumber(v) end
    return nil
end

local function origin_of(i)
    local ok, o = pcall(et.gentity_get, i, "r.currentOrigin")
    if ok and type(o) == "table" and o[1] then
        return { math.floor(o[1]), math.floor(o[2]), math.floor(o[3]) }
    end
    return nil
end

local function slot_record(i)
    local ui = et.trap_GetUserinfo(i)
    if not ui or ui == "" then return nil end
    local name = et.gentity_get(i, "pers.netname") or et.Info_ValueForKey(ui, "name") or ""
    local guid = et.Info_ValueForKey(ui, "cl_guid") or ""
    local clean = et.Q_CleanStr(name) or ""
    clean = clean:gsub("^%[BOT%]", "")
    return {
        slot = i,
        name = name,
        clean = clean,
        guid = guid,
        team = tonumber(et.gentity_get(i, "sess.sessionTeam")),
        class = tonumber(et.gentity_get(i, "sess.playerType")),
        bot = guid:match("^OMNIBOT") ~= nil,
        health = gfield(i, "health"),
        kills = gfield(i, "sess.kills"),
        deaths = gfield(i, "sess.deaths"),
        origin = origin_of(i),
    }
end

local function emit_roster()
    local list = {}
    for i = 0, max_clients() - 1 do
        local r = slot_record(i)
        if r then list[#list + 1] = r end
    end
    emit("roster", { players = list, map = et.trap_Cvar_Get("mapname") })
end

local function emit_slot(ev, i, extra)
    local r = slot_record(i) or { slot = i }
    if extra then for k, v in pairs(extra) do r[k] = v end end
    emit(ev, r)
end

local function sanitize_text(text)
    text = tostring(text or "")
    text = text:gsub("[\r\n;]", " "):gsub('"', "'")
    text = text:gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
    if #text > MAX_TEXT then text = text:sub(1, MAX_TEXT) end
    return text
end

local function say_mode(mode)
    if mode == "team" then return et.SAY_TEAM or 1 end
    if mode == "buddy" then return et.SAY_BUDDY or 2 end
    return et.SAY_ALL or 0
end

local function console_allowed(text)
    for _, p in ipairs(CONSOLE_ALLOW) do
        if text:sub(1, #p) == p then return true end
    end
    return false
end

-------------------------------------------------------------------------------
-- Inbox commands from the daemon
-------------------------------------------------------------------------------
local function handle_command(cmd)
    local kind = cmd.cmd
    if kind == "say" then
        local slot = tonumber(cmd.slot)
        local text = sanitize_text(cmd.text)
        if slot and text ~= "" and slot_record(slot) then
            et.G_Say(slot, say_mode(cmd.mode), text)
        else
            log("say dropped (slot " .. tostring(cmd.slot) .. ")")
        end
    elseif kind == "console" then
        local text = sanitize_text(cmd.text)
        if console_allowed(text) then
            et.trap_SendConsoleCommand(et.EXEC_APPEND, text .. "\n")
        else
            log("console command not allowed: " .. text)
        end
    elseif kind == "cpm" then
        -- popup message in one player's chat area (in-game feedback for voice state)
        local slot = tonumber(cmd.slot)
        local text = sanitize_text(cmd.text)
        if slot and text ~= "" and slot_record(slot) then
            et.trap_SendServerCommand(slot, 'cpm "' .. text .. '"')
        end
    elseif kind == "roster" then
        emit_roster()
    elseif kind == "ping" then
        emit("pong", { id = cmd.id })
    elseif kind == "probe" then
        -- diagnostics: which et.* API names exist, and the fireteam configstrings
        local names = {}
        for k, _ in pairs(et) do names[#names + 1] = k end
        table.sort(names)
        local fts = {}
        for i = 600, 1023 do   -- MAX_CONFIGSTRINGS is 1024; a higher index is a fatal engine error
            local ok, cs = pcall(et.trap_GetConfigstring, i)
            if ok and type(cs) == "string" and cs:find("\\id\\", 1, true) then
                fts[#fts + 1] = { cs = i, value = cs }
            end
        end
        emit("probe", { api = names, fireteams = fts, cs_fireteams = et.CS_FIRETEAMS })

    else
        log("unknown command: " .. tostring(kind))
    end
end

local function poll_inbox()
    local f = io.open(INBOX_PATH, "rb")
    if not f then return end
    local size = f:seek("end") or 0
    if size < inbox_offset then inbox_offset = 0 end -- daemon truncated the file
    if size == inbox_offset then f:close(); return end
    f:seek("set", inbox_offset)
    local data = f:read("a") or ""
    f:close()
    local consumed = 0
    for line in data:gmatch("([^\n]*)\n") do
        consumed = consumed + #line + 1
        if line:match("%S") then
            local cmd, err = json.decode(line)
            if cmd then
                local ok, e = pcall(handle_command, cmd)
                if not ok then log("command failed: " .. tostring(e)) end
            else
                log("bad inbox line: " .. tostring(err))
            end
        end
    end
    inbox_offset = inbox_offset + consumed
end

-------------------------------------------------------------------------------
-- Engine hooks
-------------------------------------------------------------------------------
function et_InitGame(levelTime, randomSeed, restartMap)
    level_time = levelTime
    et.RegisterModname("voicehub " .. VERSION)

    HOME = string.gsub(et.trap_Cvar_Get("fs_homepath"), "\\", "/") .. "/"
        .. et.trap_Cvar_Get("fs_game") .. "/voicehub/"
    EVENTS_PATH = HOME .. "events.jsonl"
    INBOX_PATH = HOME .. "inbox.jsonl"
    os.execute('mkdir -p "' .. HOME .. '"')

    events_fh = io.open(EVENTS_PATH, "ab")
    if not events_fh then
        log("cannot open " .. EVENTS_PATH)
        return
    end

    -- Ignore anything the daemon queued before this map started.
    local f = io.open(INBOX_PATH, "rb")
    if f then inbox_offset = f:seek("end") or 0; f:close() else inbox_offset = 0 end

    emit("mapstart", {
        map = et.trap_Cvar_Get("mapname"),
        gametype = tonumber(et.trap_Cvar_Get("g_gametype")),
        timelimit = tonumber(et.trap_Cvar_Get("timelimit")),
        restart = (restartMap == 1),
    })
    emit_roster()
    log("loaded, exchange dir " .. HOME)
end

function et_ShutdownGame(restartMap)
    emit("mapend", { restart = (restartMap == 1) })
    if events_fh then events_fh:close(); events_fh = nil end
end

function et_RunFrame(levelTime)
    level_time = levelTime
    if levelTime - last_poll >= POLL_MS then
        last_poll = levelTime
        local ok, e = pcall(poll_inbox)
        if not ok then log("inbox poll failed: " .. tostring(e)) end
    end
end

function et_ClientConnect(clientId, firstTime, isBot)
    emit("connect", { slot = clientId, first = (firstTime == 1), bot = (isBot == 1) })
end

function et_ClientBegin(clientId)
    emit_slot("begin", clientId)
end

function et_ClientDisconnect(clientId)
    emit_slot("disconnect", clientId)
end

function et_ClientUserinfoChanged(clientId)
    emit_slot("userinfo", clientId)
end

-- In-game voice commands. Players bind them once, e.g.
--   bind v "cmd vh_talk"     press: start listening (ends when you stop talking); press again: send now
--   bind b "cmd vh_mic"      toggle always-listening (open mic) on the voice page
-- They are forwarded to the voicehub daemon, which drives that player's browser page.
local VOICE_CMDS = { vh_talk = "talk", vh_ptt = "talk", vh_mic = "mic", vh_stop = "stop", vh_status = "status" }

function et_ClientCommand(clientId, cmdText)
    local c = string.lower(et.trap_Argv(0) or "")
    local vc = VOICE_CMDS[c]
    if vc then
        emit("voice_cmd", { slot = clientId, cmd = vc, arg = string.lower(et.trap_Argv(1) or "") })
        return 1 -- handled here; keeps "Unknown command" off the player's console
    end
    if c == "say" or c == "say_team" or c == "say_buddy" or c == "say_teamnl" then
        local mode = ({ say = "all", say_team = "team", say_buddy = "buddy", say_teamnl = "spec" })[c]
        emit("chat", { slot = clientId, mode = mode, text = et.ConcatArgs(1) })
    elseif c == "vsay" or c == "vsay_team" or c == "vsay_buddy" then
        local mode = ({ vsay = "all", vsay_team = "team", vsay_buddy = "buddy" })[c]
        emit("vsay", { slot = clientId, mode = mode, macro = et.trap_Argv(1) })
    end
    return 0
end

function et_Obituary(victimId, killerId, mod)
    emit("kill", { victim = victimId, killer = killerId, mod = mod })
end

function et_ClientSpawn(clientId, revived)
    emit("spawn", { slot = clientId, revived = (revived == 1) })
end

function et_Print(consoleText)
    local medic, victim = string.match(consoleText, "^Medic_Revive:%s+(%d+)%s+(%d+)\n$")
    if medic then
        emit("revive", { medic = tonumber(medic), victim = tonumber(victim) })
        return
    end
    -- Objective announcements (map script wm_announce) reach the console as plain lines such as
    -- "Allies have breached the Old City wall". Chat is "say: ..." so it never matches.
    -- ET: Legacy logs them as  legacy announce: "^7Allies have ..."  (plain lines on older builds).
    local s = consoleText:gsub("%^.", ""):gsub("%s+$", "")
    local quoted = s:match('^legacy announce: "(.*)"$')
    if quoted then s = quoted end
    if #s > 0 and #s < 160 and not s:find("\n")
       and (s:match("^Axis ") or s:match("^Allies ") or s:match("^Allied ") or s:match("^The Axis ") or s:match("^The Allies ")) then
        if s ~= last_announce or (level_time - last_announce_lt) > 2000 then
            last_announce, last_announce_lt = s, level_time
            emit("announce", { text = s })
        end
    end
end
