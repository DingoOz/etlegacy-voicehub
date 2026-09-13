import torch, sys
sys.path.insert(0, ".")
torch.cuda.init()
from qwen_tts import Qwen3TTSModel
from voicehub.fast_qwen import patch_model_full
from torch.profiler import profile, ProfilerActivity
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
fast = patch_model_full(m.model)
m.generate_custom_voice(text="warm up", language="English", speaker="Ryan", instruct="")
with torch.no_grad():
    for _ in range(2): fast._frame()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(5): fast._frame()
        torch.cuda.synchronize()
ev = [e for e in p.key_averages() if e.device_type.name == "CUDA" or e.self_device_time_total > 0]
rows = sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)[:22]
tot = sum(e.self_device_time_total for e in p.key_averages())
print(f"total device time per frame: {tot/5/1000:.1f} ms")
for e in rows:
    if e.self_device_time_total > 0:
        print(f"{e.self_device_time_total/5/1000:7.2f} ms  {e.count//5:5d}x  {e.key[:90]}")
