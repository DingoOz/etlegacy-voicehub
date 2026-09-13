import torch, sys
sys.path.insert(0, ".")
torch.cuda.init()
from qwen_tts import Qwen3TTSModel
from voicehub.fast_qwen import FastTalker
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.float16, attn_implementation="sdpa")
inner = m.model; tk = inner.talker
cap = {}
orig = tk.generate
def spy(**kw): cap.update(kw); raise RuntimeError("stop")
tk.generate = spy
try: m.generate_custom_voice(text="Walter! Build the bridge already.", language="English", speaker="Ryan", instruct="")
except RuntimeError: pass
tk.generate = orig
E, T, PAD, AM = cap["inputs_embeds"], cap["trailing_text_hidden"], cap["tts_pad_embed"], cap["attention_mask"]
P = E.shape[1]; print("prefill len", P, "trailing", T.shape, "mask all ones:", bool(AM.all()))
with torch.no_grad():
    out = tk(inputs_embeds=E, attention_mask=AM, trailing_text_hidden=T, tts_pad_embed=PAD, use_cache=True, return_dict=True, output_hidden_states=True)
    ref_h = out.past_hidden; ref_logit = out.logits[:, -1].float()
    print("rope_deltas", tk.rope_deltas)
    ft = FastTalker(inner, greedy=True)
    h = ft._layers(E.to(ft.dtype), torch.arange(P, device=ft.device))
    my_h = h[:, -1:]; my_logit = tk.codec_head(my_h[:, -1]).float()
    print("prefill hidden max abs diff", (ref_h.float() - my_h.float()).abs().max().item(), "ref norm", ref_h.float().abs().mean().item())
    print("argmax ref/mine", ref_logit.argmax().item(), my_logit.argmax().item(), "logit diff", (ref_logit - my_logit).abs().max().item())
    # one decode step in the original (greedy sub-talker)
    tok = ref_logit.argmax(-1, keepdim=True)
    am2 = torch.ones(1, P + 1, device=E.device, dtype=AM.dtype)
    out2 = tk(input_ids=tok, attention_mask=am2, past_key_values=out.past_key_values, cache_position=torch.tensor([P], device=E.device),
              past_hidden=ref_h, generation_step=0, trailing_text_hidden=T, tts_pad_embed=PAD, use_cache=True, return_dict=True,
              subtalker_dosample=False, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
    ref_codes = out2.hidden_states[1]; ref_logit2 = out2.logits[:, -1].float()
    # mine
    ft.past_hidden.copy_(my_h); ft.tok_in.copy_(tok.view(1)); ft.pos.fill_(P); ft.text_add.copy_(T[:, 0:1]); ft.eos_bias.fill_(float("-inf")); ft.hist.fill_(ft.filler); ft.hist[0, 0] = tok[0, 0]
    ft._frame()
    print("codes ref ", ref_codes[0].tolist()); print("codes mine", ft.codes_out[0].tolist())
    my_logit2 = tk.codec_head(ft.past_hidden[:, -1]).float()
    print("step1 hidden diff", (out2.past_hidden.float() - ft.past_hidden.float()).abs().max().item(), "argmax ref/mine", ref_logit2.argmax().item(), ft.tok_out.item())
