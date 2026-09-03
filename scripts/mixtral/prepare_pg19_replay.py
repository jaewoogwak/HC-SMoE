#!/usr/bin/env python3
"""Create a deterministic, held-out PG19 fixed-token replay artifact."""
import argparse, random
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

p=argparse.ArgumentParser(); p.add_argument("--model_name",default="mistralai/Mixtral-8x7B-v0.1"); p.add_argument("--split",default="test"); p.add_argument("--num_samples",type=int,default=8); p.add_argument("--prefill_length",type=int,default=4096); p.add_argument("--decode_length",type=int,default=256); p.add_argument("--seed",type=int,default=42); p.add_argument("--output_path",type=Path,required=True)
a=p.parse_args(); total=a.prefill_length+a.decode_length; tok=AutoTokenizer.from_pretrained(a.model_name); ds=load_dataset("emozilla/pg19",split=a.split); rng=random.Random(a.seed); indices=list(range(len(ds))); rng.shuffle(indices); samples=[]
for doc_id in indices:
    text=ds[doc_id]["text"]; ids=tok(text,add_special_tokens=False)["input_ids"]
    if len(ids)<total: continue
    start=rng.randrange(len(ids)-total+1); samples.append({"sample_id":len(samples),"document_id":int(doc_id),"start_offset":start,"input_ids":torch.tensor(ids[start:start+total],dtype=torch.long)})
    if len(samples)==a.num_samples: break
if len(samples)!=a.num_samples: raise RuntimeError(f"only found {len(samples)} PG19 documents with {total} tokens")
a.output_path.parent.mkdir(parents=True,exist_ok=True)
torch.save({"version":1,"metadata":{"dataset":"pg19","dataset_role":"test","tokenizer":a.model_name,"split":a.split,"seed":a.seed,"num_samples":a.num_samples,"prefill_length":a.prefill_length,"decode_length":a.decode_length,"selection":"contiguous","used_for_grouping":False,"used_for_selection":False},"samples":samples},a.output_path)
print(f"Wrote held-out PG19 replay tokens: {a.output_path}")

