#!/usr/bin/env python3
"""Teacher-forced PG19 router replay, loading one merged checkpoint at a time."""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from hcsmoe.merging.mixtral_checkpoint import load_compressed_model_for_evaluation
from hcsmoe.merging.pg19_routing import load_torch, trace_teacher_forced_replay
p=argparse.ArgumentParser(); p.add_argument("--tokens_path",type=Path,required=True); p.add_argument("--models_manifest",type=Path,required=True); p.add_argument("--output_dir",type=Path,required=True)
a=p.parse_args(); tokens=load_torch(a.tokens_path); manifest=json.loads(a.models_manifest.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True)
for entry in manifest["models"]:
    model_path=Path(entry["model_path"]); model_path=model_path/"model.pth" if model_path.is_dir() else model_path
    group_path=Path(entry["group_state_path"]); model_name=entry.get("model_name","mistralai/Mixtral-8x7B-v0.1")
    model,_=load_compressed_model_for_evaluation(model_name,str(model_path),str(group_path),False,None)
    trace=trace_teacher_forced_replay(model,tokens["samples"],prefill_length=tokens["metadata"]["prefill_length"],decode_length=tokens["metadata"]["decode_length"],metadata={"method_id":entry["method_id"],"model_path":str(model_path),"group_state_path":str(group_path),"dataset_role":"test","used_for_grouping":False,"used_for_selection":False})
    torch.save(trace,a.output_dir/f"{entry['method_id']}_router_trace.pt"); del model; torch.cuda.empty_cache()
print(f"Wrote merged replay traces under {a.output_dir}")

