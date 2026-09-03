#!/usr/bin/env python3
"""Compare frozen projection with actual merged-model PG19 router replay."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcsmoe.merging.pg19_routing import load_group_state,load_torch,summarize_unique_groups,routing_drift_metrics,layer0_sanity
p=argparse.ArgumentParser(); p.add_argument("--grouping_dir",type=Path,required=True); p.add_argument("--frozen_dir",type=Path,required=True); p.add_argument("--actual_dir",type=Path,required=True); p.add_argument("--output_dir",type=Path,required=True); p.add_argument("--baseline_method",default="hc_legacy")
a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); base=load_torch(a.frozen_dir/"base_router_trace.pt"); rows=[]; methods={}
for group_path in sorted(a.grouping_dir.glob("*/group_state_dict.pt")):
    method=group_path.parent.name; actual_path=a.actual_dir/f"{method}_router_trace.pt"
    if not actual_path.exists(): continue
    groups=load_group_state(group_path); actual=load_torch(actual_path); projected=summarize_unique_groups(base,groups); realized=summarize_unique_groups(actual,groups); drift=routing_drift_metrics(base,actual,groups)
    methods[method]={"projected":projected,"realized":realized,"router_drift":drift,"layer0_sanity_warnings":layer0_sanity(drift)}
baseline=methods.get(a.baseline_method)
if baseline is None: raise KeyError(f"baseline {a.baseline_method!r} has no candidate directory and replay trace")
for method,value in methods.items():
    for phase in ("prefill","decode"):
        for layer in range(32):
            key=f"{phase}/layer_{layer}"; x=value["projected"]["by_phase_layer"][key]; y=value["realized"]["by_phase_layer"][key]; bpx=baseline["projected"]["by_phase_layer"][key]; bry=baseline["realized"]["by_phase_layer"][key]; d=value["router_drift"][key]
            rows.append({"method_id":method,"phase":phase,"layer":layer,"projected_rate_1":x["rate_1"],"projected_rate_2":x["rate_2"],"projected_mean_unique":x["mean_unique_groups"],"realized_rate_1":y["rate_1"],"realized_rate_2":y["rate_2"],"realized_mean_unique":y["mean_unique_groups"],"grouping_only_gain":bpx["mean_unique_groups"]-x["mean_unique_groups"],"drift_effect":y["mean_unique_groups"]-x["mean_unique_groups"],"realized_gain":bry["mean_unique_groups"]-y["mean_unique_groups"],**d})
(a.output_dir/"grouping_drift_decomposition.json").write_text(json.dumps({"metadata":{"dataset_role":"test","used_for_grouping":False,"used_for_selection":False},"methods":methods},indent=2))
with (a.output_dir/"per_layer_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("WARNING: selecting a method after observing PG19 invalidates its unseen-test role.")
print(f"Wrote comparison artifacts under {a.output_dir}")

