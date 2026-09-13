import torch, sys
sys.path.insert(0, ".")
torch.cuda.init()
from qwen_tts import Qwen3TTSModel
from voicehub.fast_qwen import FastTalker
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
inner = m.model; tk = inner.talker
cap = {}; ref = []
orig = tk.generate
def spy(**kw):
    cap.update(kw); out = orig(**kw); ref.append(torch.stack([h[-1] for h in out.hidden_states if h[-1] is not None], 1)); return out
tk.generate = spy
m.generate_custom_voice(text="Walter! Build the bridge already.", language="English", speaker="Ryan", instruct="", do_sample=False, subtalker_dosample=False)
tk.generate = orig
E, T, PAD = cap["inputs_embeds"], cap["trailing_text_hidden"], cap["tts_pad_embed"]
r = ref[0][0, :, 0].tolist(); print("ref frames", len(r), "first tokens", r[:12])
ft = FastTalker(inner, greedy=True)
for mode in (False, True):
    fr = ft.generate(E, T, PAD, max_new_tokens=200, min_new_tokens=2, use_graph=mode)
    t = [f[0, 0].item() for f in fr]
    same = sum(1 for a, b in zip(r, t) if a == b)
    print(f"graph={mode}: frames {len(t)} first tokens {t[:12]} matching-prefix {same}/{min(len(r),len(t))}")
from voicehub.fast_qwen import patch_model_full
fast = patch_model_full(inner)
seen = []
real = fast.generate
def wrap(*a, **k):
    out = real(*a, **k); seen.append([f[0,0].item() for f in out]); return out
fast.generate = wrap
try:
    m.generate_custom_voice(text="Walter! Build the bridge already.", language="English", speaker="Ryan", instruct="", do_sample=False, subtalker_dosample=False, max_new_tokens=100)
except Exception as e: print("patched path error:", type(e).__name__, str(e)[:100])
t = seen[-1]; print("patched greedy: frames", len(t), "first", t[:12], "prefix", sum(1 for a,b in zip(r,t) if a==b))
with torch.inference_mode():
    fr = real(E, T, PAD, max_new_tokens=100, min_new_tokens=2); t=[f[0,0].item() for f in fr]; print("inference_mode direct: frames", len(t), "first", t[:12])
