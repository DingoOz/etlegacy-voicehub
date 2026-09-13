"""Speed-ups for Qwen3-TTS on old GPUs (no Triton / torch.compile on Pascal).

The talker emits one audio frame per step; for every frame a small 5-layer "code predictor"
runs 15 autoregressive sub-steps through HuggingFace `generate`. That machinery costs far
more than the maths on a 2.4 GHz Xeon. `FastCodePredictor` re-implements the sub-loop with
plain tensor ops, no KV cache (the sequence is at most 17 tokens) and captures all 15 steps
in a single CUDA graph, so one frame costs one graph replay instead of ~500 kernel launches
driven from Python.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
import torch.nn.functional as F

log = logging.getLogger("voicehub.fastqwen")


def _rope(q, k, cos, sin):
    def rot(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def _fuse_linear(mods: list[torch.nn.Module]) -> torch.Tensor:
    """One weight matrix for several bias-free Linear layers; the modules keep views into it."""
    W = torch.cat([m.weight.data for m in mods], dim=0).contiguous()
    o = 0
    for m in mods:
        n = m.weight.shape[0]
        m.weight = torch.nn.Parameter(W[o:o + n], requires_grad=False)
        o += n
    return W


class _Layer:
    """Fused-weight view of one decoder layer (talker and code predictor share the structure)."""

    def __init__(self, layer, head_dim: int):
        sa, mlp = layer.self_attn, layer.mlp
        assert sa.q_proj.bias is None and mlp.gate_proj.bias is None
        self.D = head_dim
        self.nq, self.nk = sa.q_proj.weight.shape[0] // head_dim, sa.k_proj.weight.shape[0] // head_dim
        self.w_qkv = _fuse_linear([sa.q_proj, sa.k_proj, sa.v_proj])
        self.w_o = sa.o_proj.weight
        self.w_gu = _fuse_linear([mlp.gate_proj, mlp.up_proj])
        self.w_d = mlp.down_proj.weight
        self.inter = mlp.gate_proj.weight.shape[0]
        self.ln1, self.ln2 = layer.input_layernorm, layer.post_attention_layernorm
        self.qn, self.kn = sa.q_norm, sa.k_norm
        self.scale = sa.scaling

    def qkv(self, x: torch.Tensor):
        """x [1,L,H] -> q [1,nq,L,D], k [1,nk,L,D], v [1,nk,L,D] (normed, un-roped)."""
        L = x.shape[1]
        h = _norm(x, self.ln1)
        qkv = F.linear(h, self.w_qkv)
        q, k, v = qkv.split([self.nq * self.D, self.nk * self.D, self.nk * self.D], dim=-1)
        q = _norm(q.view(1, L, self.nq, self.D), self.qn).transpose(1, 2)
        k = _norm(k.view(1, L, self.nk, self.D), self.kn).transpose(1, 2)
        v = v.view(1, L, self.nk, self.D).transpose(1, 2)
        return q, k, v

    def rope(self, q, k, cos, sin):
        qk = torch.cat((q, k), dim=1)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        x1, x2 = qk[..., : self.D // 2], qk[..., self.D // 2:]
        qk = qk * cos + torch.cat((-x2, x1), dim=-1) * sin
        return qk[:, : self.nq], qk[:, self.nq:]

    def out(self, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        x = x + F.linear(o.transpose(1, 2).reshape(1, L, -1), self.w_o)
        gu = F.linear(_norm(x, self.ln2), self.w_gu)
        g, u = gu.split([self.inter, self.inter], dim=-1)
        return x + F.linear(F.silu(g) * u, self.w_d)


def _norm(x: torch.Tensor, mod) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), mod.weight, mod.variance_epsilon)


class FastCodePredictor:
    def __init__(self, talker, top_k: int = 50, temperature: float = 0.9, greedy: bool = False):
        self.cp = talker.code_predictor
        self.model = self.cp.model
        self.layers = list(self.model.layers)
        self.head_dim = self.layers[0].self_attn.head_dim
        self.fl = [_Layer(l, self.head_dim) for l in self.layers]
        self.n_groups = self.cp.config.num_code_groups          # 16 -> 15 predicted codes
        self.top_k, self.temperature, self.greedy = top_k, temperature, greedy
        self.device = next(self.cp.parameters()).device
        self.dtype = next(self.cp.parameters()).dtype
        self.graph: torch.cuda.CUDAGraph | None = None
        self.in_buf = torch.zeros(1, 2, talker.config.hidden_size, device=self.device, dtype=self.dtype)
        pos = torch.arange(self.n_groups + 1, device=self.device).unsqueeze(0)
        cos, sin = self.model.rotary_emb(self.in_buf, pos)
        self.cos_tab, self.sin_tab = cos[0], sin[0]                       # [17, D]
        nl, nk, T = len(self.layers), self.fl[0].nk, self.n_groups + 1
        self.k_cache = torch.zeros(nl, 1, nk, T, self.head_dim, device=self.device, dtype=self.dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.out_buf = torch.zeros(1, self.n_groups - 1, device=self.device, dtype=torch.long)

    # ---- forward for the tokens at [start, start+L) using the per-step cache; returns last hidden [1, H]
    def _forward(self, x: torch.Tensor, start: int) -> torch.Tensor:
        L = x.shape[1]
        end = start + L
        cos, sin = self.cos_tab[start:end].unsqueeze(0), self.sin_tab[start:end].unsqueeze(0)
        for i, fl in enumerate(self.fl):
            q, k, v = fl.qkv(x)
            q, k = fl.rope(q, k, cos, sin)
            self.k_cache[i, :, :, start:end] = k
            self.v_cache[i, :, :, start:end] = v
            o = F.scaled_dot_product_attention(q, self.k_cache[i, :, :, :end], self.v_cache[i, :, :, :end],
                                               is_causal=(L > 1), scale=fl.scale, enable_gqa=True)
            x = fl.out(x, o)
        return _norm(x[:, -1], self.model.norm)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.greedy:
            return logits.argmax(-1)
        if self.top_k and self.top_k < logits.shape[-1]:
            kth = logits.topk(self.top_k, dim=-1).values[..., -1:]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        u = torch.rand_like(logits).clamp_(1e-10, 1.0)
        return (logits / self.temperature - torch.log(-torch.log(u))).argmax(-1)   # Gumbel-max == softmax sampling

    def _run(self, prefill: torch.Tensor) -> torch.Tensor:
        x = self.cp.small_to_mtp_projection(prefill)
        toks = []
        start = 0
        for step in range(self.n_groups - 1):
            h = self._forward(x, start)
            start += x.shape[1]
            t = self._sample(self.cp.lm_head[step](h))
            toks.append(t)
            if step < self.n_groups - 2:
                x = self.cp.small_to_mtp_projection(self.model.codec_embedding[step](t).unsqueeze(1))
        return torch.stack(toks, dim=-1)

    def capture(self) -> None:
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self.out_buf.copy_(self._run(self.in_buf))
        torch.cuda.current_stream(self.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            self.out_buf.copy_(self._run(self.in_buf))
        self.graph = g
        log.info("code predictor CUDA graph captured")

    @torch.no_grad()
    def generate(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            self.capture()
        self.in_buf.copy_(inputs_embeds)
        self.graph.replay()
        return self.out_buf.clone()

    @torch.no_grad()
    def generate_eager(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self._run(inputs_embeds.to(self.dtype))


def patch_model(model, top_k: int = 50, temperature: float = 0.9, use_graph: bool = True):
    """Replace talker.code_predictor.generate with the fast path (batch size 1 only)."""
    talker = model.talker
    fast = FastCodePredictor(talker, top_k=top_k, temperature=temperature)
    original = talker.code_predictor.generate

    def generate(inputs_embeds=None, **kw):
        if inputs_embeds is None or inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] != 2:
            return original(inputs_embeds=inputs_embeds, **kw)
        seq = fast.generate(inputs_embeds) if use_graph else fast.generate_eager(inputs_embeds)
        return SimpleNamespace(sequences=seq)

    talker.code_predictor.generate = generate
    model._fast_code_predictor = fast
    return fast


class FastTalker:
    """Whole-frame CUDA graph for the talker: one replay = embed previous frame's codes, run the
    28-layer decode step against a static KV cache, sample the next first-codebook token (with
    repetition penalty / suppressed tokens / min-new-tokens), and run the code predictor for
    the remaining 15 codebooks. The prefill runs eagerly through the same tensor code."""

    def __init__(self, model, max_len: int = 1024, max_new: int = 512, top_k: int = 50, temperature: float = 0.9,
                 repetition_penalty: float = 1.05, cp_top_k: int = 50, cp_temperature: float = 0.9,
                 greedy: bool = False):
        self.tk = model.talker
        self.tm = self.tk.model
        self.layers = list(self.tm.layers)
        self.fl = [_Layer(l, self.layers[0].self_attn.head_dim) for l in self.layers]
        cfg = self.tk.config
        self.cfg = cfg
        self.device = next(self.tk.parameters()).device
        self.dtype = next(self.tk.parameters()).dtype
        self.H, self.D = cfg.hidden_size, self.layers[0].self_attn.head_dim
        self.n_kv = cfg.num_key_value_heads
        self.eos = cfg.codec_eos_token_id
        V = cfg.vocab_size
        self.top_k, self.temperature, self.rep, self.greedy = top_k, temperature, repetition_penalty, greedy
        self.max_len, self.max_new = max_len, max_new
        self.cp = FastCodePredictor(self.tk, cp_top_k, cp_temperature, greedy)

        d, dt = self.device, self.dtype
        bias = torch.zeros(V, device=d)
        for i in range(V - 1024, V):
            if i != self.eos:
                bias[i] = float("-inf")
        self.suppress_bias = bias
        self.filler = V - 1                      # a suppressed token: harmless in the penalty history
        nl = len(self.layers)
        self.k_cache = torch.zeros(nl, 1, self.n_kv, max_len, self.D, device=d, dtype=dt)
        self.v_cache = torch.zeros_like(self.k_cache)
        pos_all = torch.arange(max_len, device=d).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.tm.rotary_emb(torch.zeros(1, device=d, dtype=dt), pos_all)
        self.cos_tab, self.sin_tab = cos[0, 0], sin[0, 0]          # [max_len, D]; text-only => 3 axes equal
        self.arange = torch.arange(max_len, device=d)
        # static graph state
        self.pos = torch.zeros((), device=d, dtype=torch.long)
        self.tok_in = torch.zeros(1, device=d, dtype=torch.long)
        self.past_hidden = torch.zeros(1, 1, self.H, device=d, dtype=dt)
        self.text_add = torch.zeros(1, 1, self.H, device=d, dtype=dt)
        self.hist = torch.full((1, max_new), self.filler, device=d, dtype=torch.long)
        self.eos_bias = torch.zeros((), device=d)
        self.codes_out = torch.zeros(1, cfg.num_code_groups, device=d, dtype=torch.long)
        self.tok_out = torch.zeros(1, device=d, dtype=torch.long)
        self.graph: torch.cuda.CUDAGraph | None = None

    # ---- transformer over `x` [1, L, H] whose tokens sit at `positions` [L] (writes the cache)
    def _layers(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        cos = self.cos_tab[positions].unsqueeze(0)
        sin = self.sin_tab[positions].unsqueeze(0)
        mask = (self.arange.view(1, -1) <= positions.view(-1, 1)).view(1, 1, L, self.max_len)
        for i, fl in enumerate(self.fl):
            q, k, v = fl.qkv(x)
            q, k = fl.rope(q, k, cos, sin)
            self.k_cache[i].index_copy_(2, positions, k)
            self.v_cache[i].index_copy_(2, positions, v)
            o = F.scaled_dot_product_attention(q, self.k_cache[i], self.v_cache[i], attn_mask=mask,
                                               scale=fl.scale, enable_gqa=True)
            x = fl.out(x, o)
        return _norm(x, self.tm.norm)

    def _sample_first(self, h_last: torch.Tensor) -> torch.Tensor:
        logits = self.tk.codec_head(h_last).float()                       # [1, V]
        score = logits.gather(1, self.hist)
        score = torch.where(score < 0, score * self.rep, score / self.rep)
        logits = logits.scatter(1, self.hist, score)
        logits = logits + self.suppress_bias
        logits[:, self.eos] += self.eos_bias
        if self.greedy:
            return logits.argmax(-1)
        kth = logits.topk(self.top_k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        u = torch.rand_like(logits).clamp_(1e-10, 1.0)
        return (logits / self.temperature - torch.log(-torch.log(u))).argmax(-1)

    # ---- one frame, graph body: reads tok_in/past_hidden/pos/text_add/hist/eos_bias, writes outputs
    def _frame(self) -> None:
        last_id_hidden = self.tm.codec_embedding(self.tok_in).unsqueeze(1)          # [1,1,H]
        codes15 = self.cp._run(torch.cat((self.past_hidden, last_id_hidden), dim=1))  # [1,15]
        emb = last_id_hidden
        for i in range(codes15.shape[1]):
            emb = emb + self.cp.model.codec_embedding[i](codes15[:, i]).unsqueeze(1)
        x = emb + self.text_add
        h = self._layers(x, self.pos.view(1))
        tok = self._sample_first(h[:, -1])
        self.codes_out.copy_(torch.cat((self.tok_in.view(1, 1), codes15), dim=1))
        self.past_hidden.copy_(h)
        self.tok_out.copy_(tok)

    def capture(self) -> None:
        saved = (self.tok_in.clone(), self.past_hidden.clone(), self.pos.clone(), self.hist.clone())
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._frame()
        torch.cuda.current_stream(self.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            self._frame()
        self.graph = g
        self.graph_flags = (self.greedy, self.cp.greedy)
        self.tok_in.copy_(saved[0]); self.past_hidden.copy_(saved[1]); self.pos.copy_(saved[2]); self.hist.copy_(saved[3])
        log.info("talker frame CUDA graph captured (greedy=%s/%s)", *self.graph_flags)

    @torch.no_grad()
    def generate(self, inputs_embeds: torch.Tensor, trailing_text_hidden: torch.Tensor, tts_pad_embed: torch.Tensor,
                 max_new_tokens: int = 2048, min_new_tokens: int = 2, use_graph: bool = True) -> list[torch.Tensor]:
        """Returns the list of per-frame code tensors [1, 16] (EOS frame excluded)."""
        P = inputs_embeds.shape[1]
        max_new = min(max_new_tokens, self.max_new - 1, self.max_len - P - 1)
        self.hist.fill_(self.filler)
        self.k_cache.zero_(); self.v_cache.zero_()
        self.eos_bias.fill_(float("-inf") if min_new_tokens > 0 else 0.0)
        h = self._layers(inputs_embeds.to(self.dtype), torch.arange(P, device=self.device))
        tok = self._sample_first(h[:, -1])
        self.past_hidden.copy_(h[:, -1:])
        self.hist[0, 0] = tok[0]
        self.pos.fill_(P)
        if use_graph and (self.graph is None or self.graph_flags != (self.greedy, self.cp.greedy)):
            self.capture()   # sampling mode is baked into the graph
        frames: list[torch.Tensor] = []
        n_text = trailing_text_hidden.shape[1]
        for g in range(max_new):
            if tok.item() == self.eos:
                break
            self.tok_in.copy_(tok)
            self.text_add.copy_(trailing_text_hidden[:, g:g + 1] if g < n_text else tts_pad_embed)
            self.eos_bias.fill_(float("-inf") if g + 1 < min_new_tokens else 0.0)
            if use_graph:
                self.graph.replay()
            else:
                self._frame()
            frames.append(self.codes_out.clone())
            tok = self.tok_out.clone()
            self.hist[0, g + 1] = tok[0]
            self.pos.add_(1)
        return frames


def patch_model_full(model, **kw) -> FastTalker:
    """Replace talker.generate with the graph-captured loop (batch size 1; other cases fall back)."""
    fast = FastTalker(model, **kw)
    talker = model.talker
    original = talker.generate

    def generate(inputs_embeds=None, attention_mask=None, trailing_text_hidden=None, tts_pad_embed=None, **gk):
        if inputs_embeds is None or inputs_embeds.shape[0] != 1 or (attention_mask is not None and not bool(attention_mask.all())):
            return original(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            trailing_text_hidden=trailing_text_hidden, tts_pad_embed=tts_pad_embed, **gk)
        fast.greedy = not gk.get("do_sample", True)
        fast.cp.greedy = not gk.get("subtalker_dosample", True)
        frames = fast.generate(inputs_embeds, trailing_text_hidden, tts_pad_embed,
                               max_new_tokens=int(gk.get("max_new_tokens", 2048)), min_new_tokens=int(gk.get("min_new_tokens", 2)))
        dummy = fast.past_hidden
        hidden_states = [((dummy,), None)] + [((dummy,), c) for c in frames]
        return SimpleNamespace(hidden_states=hidden_states)

    talker.generate = generate
    model._fast_talker = fast
    return fast
