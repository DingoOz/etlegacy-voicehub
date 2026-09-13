import asyncio, json, ssl, time, httpx, websockets
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
async def main():
    async with websockets.connect("wss://localhost:8443/ws", ssl=ctx) as ws:
        await ws.recv()
        t0 = time.time()
        async def drain(limit_s):
            out = []
            end = time.time() + limit_s
            while time.time() < end:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), max(0.1, end - time.time())))
                    if m["type"] in ("speech", "transcript"): out.append((round(time.time() - t0, 2), m["type"], m.get("bot"), m.get("emotion"), m.get("part"), m.get("parts"), m.get("text")))
                except asyncio.TimeoutError: break
            return out
        async with httpx.AsyncClient(verify=False, timeout=60) as c:
            bot = (await c.get("https://localhost:8443/api/state")).json()["voiced_bots"][0]
            r = await c.post("https://localhost:8443/api/test_speak", params={"bot": bot, "text": "Yes! Got him, Dingo, did you see that? Nobody gets past me on this map."})
            print("test_speak:", r.status_code, {k: r.json().get(k) for k in ("tts_s", "parts")})
            for row in await drain(3): print("  ws", row)
            t0 = time.time()
            r = await c.post("https://localhost:8443/api/ask", params={"bot": bot, "text": bot + ", what are you doing right now?", "speaker": "Dingo"})
            print("ask:", r.status_code, r.json())
            for row in await drain(25): print("  ws", row)
asyncio.run(main())
