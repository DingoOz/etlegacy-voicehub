import time, torch, sys, soundfile as sf
sys.path.insert(0, ".")
torch.cuda.init()
from qwen_tts import Qwen3TTSModel
from voicehub.fast_qwen import patch_model_full
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
fast = patch_model_full(m.model)
m.generate_custom_voice(text="warm up", language="English", speaker="Ryan", instruct="")
tot = aud = 0
lines = [("Yes! Got him, Dingo, did you see that?", "excited"), ("Just hiding here, Dingo... planning to die soon probably.", "tired"),
         ("Oh great... Allies running away again. Shocking.", "sarcastic"), ("Walter! Build the bridge already, we're waiting on you!", "angry")]
for i, (txt, emo) in enumerate(lines):
    t = time.time(); wavs, sr = m.generate_custom_voice(text=txt, language="English", speaker=["Ryan","Aiden"][i%2], instruct=f"Sound {emo}. A young man in his early twenties, gamer at a LAN party."); dt = time.time() - t; d = len(wavs[0]) / sr
    tot += dt; aud += d; print(f"fast sampled: {dt:.2f}s for {d:.2f}s audio (RTF {dt/d:.2f})")
    sf.write(f"samples/fast_full_{i}.wav", wavs[0], sr)
print(f"mean RTF {tot/aud:.2f}; peak vram {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
# where does the remaining time go? time one frame replay and the prefill+decode split
import statistics
x = torch.cuda.Event(enable_timing=True); y = torch.cuda.Event(enable_timing=True)
ts=[]
for _ in range(20):
    x.record(); fast.graph.replay(); y.record(); torch.cuda.synchronize(); ts.append(x.elapsed_time(y))
print(f"graph replay per frame: {statistics.median(ts):.1f} ms  (=> {1000/12.5/statistics.median(ts):.2f}x realtime for the frame loop alone)")
