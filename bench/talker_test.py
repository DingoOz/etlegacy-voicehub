import time, torch, sys, soundfile as sf
sys.path.insert(0, ".")
torch.cuda.init()
from qwen_tts import Qwen3TTSModel
from voicehub.fast_qwen import patch_model_full
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
inner = m.model
txt = "Walter! Build the bridge already, we're waiting on you!"
# greedy reference with the original code
orig_generate = inner.talker.generate
codes_ref = []
def spy(**kw):
    out = orig_generate(**kw); codes_ref.append(torch.stack([h[-1] for h in out.hidden_states if h[-1] is not None], 1)); return out
inner.talker.generate = spy
t = time.time(); wav_ref, sr = m.generate_custom_voice(text=txt, language="English", speaker="Ryan", instruct="", do_sample=False, subtalker_dosample=False); t_ref = time.time() - t
inner.talker.generate = orig_generate
ref = codes_ref[0][0]  # [T,16]
fast = patch_model_full(inner)
t = time.time(); wav_fast, sr = m.generate_custom_voice(text=txt, language="English", speaker="Ryan", instruct="", do_sample=False, subtalker_dosample=False); t_fast = time.time() - t
got = torch.cat(inner._fast_talker_last if hasattr(inner, "_fast_talker_last") else [], 0) if False else None
print(f"greedy: original {t_ref:.2f}s ({len(wav_ref[0])/sr:.2f}s audio) vs fast {t_fast:.2f}s ({len(wav_fast[0])/sr:.2f}s audio)")
# frame-level comparison: rerun fast greedy and capture the frames
fr = fast.generate(*fast._last_inputs) if hasattr(fast, "_last_inputs") else None
sf.write("samples/greedy_ref.wav", wav_ref[0], sr); sf.write("samples/greedy_fast.wav", wav_fast[0], sr)
# sampled timing
m.generate_custom_voice(text="warm up", language="English", speaker="Ryan", instruct="")
tot = aud = 0
for i, (txt, emo) in enumerate([("Yes! Got him, Dingo, did you see that?", "excited"), ("Just hiding here, Dingo... planning to die soon probably.", "tired"), ("Oh great... Allies running away again. Shocking.", "sarcastic")]):
    t = time.time(); wavs, sr = m.generate_custom_voice(text=txt, language="English", speaker=["Ryan","Aiden"][i%2], instruct=f"Sound {emo}. A young man in his early twenties, gamer at a LAN party."); dt = time.time() - t; d = len(wavs[0]) / sr
    tot += dt; aud += d; print(f"fast sampled: {dt:.2f}s for {d:.2f}s audio (RTF {dt/d:.2f})")
    sf.write(f"samples/fast_full_{i}.wav", wavs[0], sr)
print(f"mean RTF {tot/aud:.2f}; peak vram {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
