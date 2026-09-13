import time, torch, sys
sys.path.insert(0, ".")
torch.cuda.init()
from qwen_tts import Qwen3TTSModel
from voicehub.fast_qwen import FastCodePredictor, patch_model
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
talker = m.model.talker if hasattr(m, "model") else m.talker
cp = talker.code_predictor
# 1) record real inputs from one original generation
rec = []
orig = cp.generate
def spy(inputs_embeds=None, **kw):
    out = orig(inputs_embeds=inputs_embeds, **kw); rec.append((inputs_embeds.detach().clone(), kw)); return out
cp.generate = spy
m.generate_custom_voice(text="Walter! Build the bridge already.", language="English", speaker="Ryan", instruct="")
cp.generate = orig
print("recorded frames:", len(rec), "kw:", {k: v for k, v in rec[0][1].items() if not torch.is_tensor(v)})
# 2) greedy equivalence on recorded inputs
fast = FastCodePredictor(talker, greedy=True)
mism = 0
for x, kw in rec[:20]:
    ref = orig(inputs_embeds=x, max_new_tokens=15, do_sample=False, output_hidden_states=True, return_dict_in_generate=True).sequences
    got = fast.generate_eager(x)
    mism += int((ref != got).any())
print("greedy mismatching frames:", mism, "of", min(20, len(rec)))
# per-code agreement over all frames
tot=agree=0
for x, kw in rec:
    ref = orig(inputs_embeds=x, max_new_tokens=15, do_sample=False, output_hidden_states=True, return_dict_in_generate=True).sequences
    got = fast.generate_eager(x); tot += 15; agree += int((ref == got).sum())
print(f"per-code agreement {agree}/{tot}")
# 3) speed: eager vs graph vs original
x = rec[0][0]
fast = FastCodePredictor(talker)
for name, fn in [("orig", lambda: orig(inputs_embeds=x, max_new_tokens=15, do_sample=True, top_k=50, temperature=0.9, output_hidden_states=True, return_dict_in_generate=True).sequences),
                 ("eager", lambda: fast.generate_eager(x)), ("graph", lambda: fast.generate(x))]:
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(10): fn()
    torch.cuda.synchronize(); print(f"{name}: {(time.time()-t)/10*1000:.1f} ms per frame")
