"""Router tracing and unique-group metrics for held-out Mixtral PG19 replay."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import torch
import torch.nn.functional as F

TRACE_VERSION = 1
PHASE_TO_ID = {"prefill": 0, "decode": 1}
ID_TO_PHASE = {v: k for k, v in PHASE_TO_ID.items()}

def load_torch(path: str | Path) -> Any:
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")

def layer_name(layer: int) -> str: return f"model.layers.{layer}.block_sparse_moe"

def load_group_state(path: str | Path) -> dict[str, torch.Tensor]:
    value = load_torch(path)
    if not isinstance(value, Mapping): raise TypeError("group state must be a mapping")
    result = {str(k): v.detach().cpu().long().flatten() for k, v in value.items()}
    if any(v.numel() != 8 for v in result.values()): raise ValueError("all Mixtral mappings require 8 labels")
    return result

class RouterTraceCollector:
    def __init__(self, model):
        self.current, self.rows = None, defaultdict(list)
        self.handles = [layer.block_sparse_moe.gate.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(model.model.layers)]
    def _hook(self, layer):
        def capture(_module, _inputs, output):
            if self.current is None: raise RuntimeError("gate invoked outside trace scope")
            logits = output.detach().cpu().to(torch.float16)
            if logits.ndim != 2 or logits.shape[0] != len(self.current["positions"]):
                raise RuntimeError(f"layer {layer} router rows do not match replay positions")
            top2 = torch.topk(logits.float(), 2, -1).indices.to(torch.uint8); n = logits.shape[0]
            for name, value in {
                "sample_id": torch.full((n,), self.current["sample_id"], dtype=torch.int32),
                "phase": torch.full((n,), PHASE_TO_ID[self.current["phase"]], dtype=torch.uint8),
                "position": torch.tensor(self.current["positions"], dtype=torch.int32),
                "decode_step": torch.full((n,), self.current["decode_step"], dtype=torch.int32),
                "layer": torch.full((n,), layer, dtype=torch.int16),
                "top1_expert": top2[:, 0], "top2_expert": top2[:, 1], "router_logits": logits,
            }.items(): self.rows[name].append(value)
        return capture
    def begin(self, sample_id, phase, positions, decode_step=-1):
        if self.current is not None or phase not in PHASE_TO_ID: raise RuntimeError("invalid trace scope")
        self.current = {"sample_id": int(sample_id), "phase": phase, "positions": list(positions), "decode_step": int(decode_step)}
    def end(self): self.current = None
    def close(self):
        for handle in self.handles: handle.remove()
    def payload(self, metadata):
        names = ("sample_id","phase","position","decode_step","layer","top1_expert","top2_expert","router_logits")
        result = {"version": TRACE_VERSION, "metadata": dict(metadata),
                  "records": {k: torch.cat(self.rows[k]) for k in names}}
        validate_trace_payload(result); return result

def _input_device(model):
    for p in model.get_input_embeddings().parameters():
        if p.device.type != "meta": return p.device
    for p in model.parameters():
        if p.device.type != "meta": return p.device
    raise RuntimeError("no model input device")

def trace_teacher_forced_replay(model, samples: Iterable[Mapping[str, Any]], *, prefill_length, decode_length, metadata):
    model.eval(); device = _input_device(model); collector = RouterTraceCollector(model)
    try:
        with torch.no_grad():
            for sample in samples:
                sid, ids = int(sample["sample_id"]), sample["input_ids"].cpu().long().flatten()
                if ids.numel() != prefill_length + decode_length: raise ValueError(f"invalid token count for {sid}")
                prompt = ids[:prefill_length].view(1,-1).to(device)
                collector.begin(sid, "prefill", range(prefill_length))
                try: out = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True, return_dict=True)
                finally: collector.end()
                cache = out.past_key_values
                for step in range(decode_length):
                    token = ids[prefill_length+step].view(1,1).to(device)
                    mask = torch.ones((1,prefill_length+step+1), device=device, dtype=torch.long)
                    collector.begin(sid, "decode", [prefill_length+step], step)
                    try: out = model(input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
                    finally: collector.end()
                    cache = out.past_key_values
    finally: collector.close()
    meta = dict(metadata); meta.update({"dataset_role":"test","used_for_grouping":False,"used_for_selection":False,
        "trace_kind":"teacher_forced_router_trace","prefill_length":prefill_length,"decode_length":decode_length,"top_k":2,"num_experts":8})
    return collector.payload(meta)

def validate_trace_payload(trace):
    r = trace.get("records", {})
    names = ("sample_id","phase","position","decode_step","layer","top1_expert","top2_expert","router_logits")
    if trace.get("version") != TRACE_VERSION or any(k not in r for k in names): raise ValueError("invalid trace payload")
    if len({int(r[k].shape[0]) for k in names}) != 1 or r["router_logits"].ndim != 2 or r["router_logits"].shape[1] != 8:
        raise ValueError("invalid trace tensor shapes")
    if not torch.isfinite(r["router_logits"].float()).all(): raise ValueError("non-finite router logits")

def trace_keys(trace):
    validate_trace_payload(trace); r=trace["records"]
    return torch.stack((r["sample_id"].long(),r["phase"].long(),r["position"].long(),r["decode_step"].long(),r["layer"].long()),1)

def require_trace_alignment(base, candidate):
    if not torch.equal(trace_keys(base), trace_keys(candidate)): raise ValueError("base/candidate trace keys differ")

def _metric(a,b,labels):
    same = labels[a.long()] == labels[b.long()]; one,total=int(same.sum()),int(same.numel()); rate=one/total if total else 0.
    return {"num_tokens":total,"unique_group_count":{"1":one,"2":total-one},"unique_group_rate":{"1":rate,"2":1-rate},
            "rate_1":rate,"rate_2":1-rate,"mean_unique_groups":2-rate,"same_group_routing_rate":rate}

def summarize_unique_groups(trace, group_state):
    validate_trace_payload(trace); r=trace["records"]; result={"by_phase_layer":{},"by_phase":{},"all":None}
    for pid,phase in ID_TO_PHASE.items():
        ms=[]
        for layer in range(32):
            mask=(r["phase"]==pid)&(r["layer"]==layer); m=_metric(r["top1_expert"][mask],r["top2_expert"][mask],group_state[layer_name(layer)])
            result["by_phase_layer"][f"{phase}/layer_{layer}"]=m; ms.append(m)
        one,total=sum(x["unique_group_count"]["1"] for x in ms),sum(x["num_tokens"] for x in ms); rate=one/total if total else 0.
        result["by_phase"][phase]={"num_tokens":total,"unique_group_count":{"1":one,"2":total-one},"unique_group_rate":{"1":rate,"2":1-rate},"rate_1":rate,"rate_2":1-rate,"mean_unique_groups":2-rate,"same_group_routing_rate":rate}
    ms=[_metric(r["top1_expert"][r["layer"]==l],r["top2_expert"][r["layer"]==l],group_state[layer_name(l)]) for l in range(32)]
    one,total=sum(x["unique_group_count"]["1"] for x in ms),sum(x["num_tokens"] for x in ms); rate=one/total if total else 0.
    result["all"]={"num_tokens":total,"unique_group_count":{"1":one,"2":total-one},"unique_group_rate":{"1":rate,"2":1-rate},"rate_1":rate,"rate_2":1-rate,"mean_unique_groups":2-rate,"same_group_routing_rate":rate}
    return result

def routing_drift_metrics(base, candidate, group_state):
    require_trace_alignment(base,candidate); a,b=base["records"],candidate["records"]; out={}
    for pid,phase in ID_TO_PHASE.items():
        for layer in range(32):
            mask=(a["phase"]==pid)&(a["layer"]==layer); a1,a2=a["top1_expert"][mask].long(),a["top2_expert"][mask].long(); b1,b2=b["top1_expert"][mask].long(),b["top2_expert"][mask].long(); labels=group_state[layer_name(layer)]
            aset,bset=torch.sort(torch.stack((a1,a2),1),1).values,torch.sort(torch.stack((b1,b2),1),1).values
            ga,gb=torch.sort(torch.stack((labels[a1],labels[a2]),1),1).values,torch.sort(torch.stack((labels[b1],labels[b2]),1),1).values
            la,lb=a["router_logits"][mask].float(),b["router_logits"][mask].float(); pa,pb=F.softmax(la,-1),F.softmax(lb,-1); mid=(pa+pb)/2
            js=((pa*(pa.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(-1)+(pb*(pb.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(-1))/2
            mean=lambda x:float(x.float().mean()) if x.numel() else 0.
            out[f"{phase}/layer_{layer}"]={"num_tokens":int(mask.sum()),"top1_expert_agreement":mean(a1==b1),"top2_expert_set_exact_agreement":mean(torch.all(aset==bset,1)),"top2_expert_overlap":mean(((a1[:,None]==b1[:,None])|(a1[:,None]==b2[:,None])).sum(1)/2),"group_set_exact_agreement":mean(torch.all(ga==gb,1)),"unique_group_count_agreement":mean((ga[:,0]==ga[:,1])==(gb[:,0]==gb[:,1])),"router_logit_cosine":mean(F.cosine_similarity(la,lb,dim=-1,eps=1e-8)),"router_logit_relative_l2":float(torch.linalg.vector_norm(la-lb)/torch.linalg.vector_norm(la).clamp_min(1e-8)) if la.numel() else 0.,"router_probability_js_divergence":mean(js)}
    return out

def layer0_sanity(drift, max_logit_relative_l2=1e-3):
    warnings=[]
    for phase in ID_TO_PHASE.values():
        x=drift[f"{phase}/layer_0"]
        if x["top1_expert_agreement"]<.999 or x["top2_expert_set_exact_agreement"]<.999 or x["router_logit_relative_l2"]>max_logit_relative_l2: warnings.append(f"{phase} layer 0 sanity failure: {x}")
    return warnings

