#!/usr/bin/env python3
"""Trace base Mixtral on fixed PG19 tokens, then project every candidate mapping."""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoTokenizer, MixtralForCausalLM
from hcsmoe.merging.pg19_routing import load_group_state, load_torch, summarize_unique_groups, trace_teacher_forced_replay
p=argparse.ArgumentParser(); p.add_argument("--model_name",default="mistralai/Mixtral-8x7B-v0.1"); p.add_argument("--tokens_path",type=Path,required=True); p.add_argument("--grouping_dir",type=Path,required=True); p.add_argument("--output_dir",type=Path,required=True)
a=p.parse_args(); tokens=load_torch(a.tokens_path); meta=tokens["metadata"]; print("WARNING: PG19 is held-out test data; do not select methods/alpha from these results.")
model=MixtralForCausalLM.from_pretrained(a.model_name,torch_dtype=torch.bfloat16,device_map="auto"); trace=trace_teacher_forced_replay(model,tokens["samples"],prefill_length=meta["prefill_length"],decode_length=meta["decode_length"],metadata={"model_name":a.model_name,"tokens_path":str(a.tokens_path),"dataset_role":"test","used_for_grouping":False,"used_for_selection":False})
a.output_dir.mkdir(parents=True,exist_ok=True); torch.save(trace,a.output_dir/"base_router_trace.pt")
methods={}
for path in sorted(a.grouping_dir.glob("*/group_state_dict.pt")):
    methods[path.parent.name]=summarize_unique_groups(trace,load_group_state(path))
(a.output_dir/"frozen_projection_summary.json").write_text(json.dumps({"metadata":{"dataset_role":"test","used_for_grouping":False,"used_for_selection":False},"methods":methods},indent=2))
print(f"Wrote base trace and frozen projection summaries under {a.output_dir}")

