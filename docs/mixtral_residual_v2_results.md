# Mixtral HC-SMoE residual results (v2)

## Setup

- Model: `mistralai/Mixtral-8x7B-v0.1`
- Static merge: frequency-weighted (`merge=freq`), hierarchical/average grouping,
  `num_average_groups=4`
- Residual calibration: C4, 4,096 selected tokens per expert; deterministic seed 0
- Residual training: FP32, AdamW, learning rate 0.001, batch size 64, 3 epochs,
  validation ratio 0.1, patience 2
- MC tasks: `winogrande`, `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`,
  `mmlu`, `openbookqa`, and `rte`

## Held-out expert reconstruction

The static checkpoint's held-out reconstruction is relative L2 `0.4014` and cosine
`0.9178`. Width 256 improves both metrics; the fixed training hyperparameters are
not stable for widths 512 and 1024.

| Residual width | Relative L2 (lower is better) | Cosine (higher is better) | Residual parameters | Residual / original experts | Logical total expert parameter ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0.4014 | 0.9178 | 0 | 0.00% | 50.00% |
| 256 | **0.3313** | **0.9447** | 585,105,408 | 1.30% | 51.30% |
| 512 | 0.4645 | 0.8972 | 1,170,210,816 | 2.59% | 52.59% |
| 1024 | 0.9271 | 0.7099 | 2,340,421,632 | 5.19% | 55.19% |

Each residual run used 76,260 held-out routed tokens. The 256-width residual
reduces aggregate relative L2 by about 17.5% compared with static HC-SMoE.

## Multiple-choice evaluation

All values are accuracy. The `Mean` column is an unweighted descriptive average
across the eight tasks, not an lm-eval aggregate metric.

| Model | WinoGrande | ARC-C | ARC-E | BoolQ | HellaSwag | MMLU | OpenBookQA | RTE | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0.7324 | 0.4531 | 0.7361 | 0.8419 | 0.5717 | 0.4943 | 0.2940 | 0.6029 | 0.5908 |
| Residual 256 | 0.7222 | 0.4215 | 0.7273 | 0.8379 | 0.5405 | 0.4776 | 0.2820 | 0.5523 | 0.5702 |
| Residual 512 | 0.7182 | 0.4326 | 0.7315 | 0.8352 | 0.5396 | 0.4712 | 0.2900 | 0.5560 | 0.5718 |
| Residual 1024 | 0.6914 | 0.4121 | 0.7138 | 0.8226 | 0.5313 | 0.4566 | 0.2960 | 0.5487 | 0.5591 |

The raw MC outputs are intentionally left untracked under `results/*/mc_results.txt`.

## Completion status

- Static, width 256, 512, and 1024: static/residual checkpoints, reconstruction
  metrics, and MC evaluation completed.
- Generation suite (`humaneval`, `humaneval_plus`, `mbpp`, `mbpp_plus`, `gsm8k`,
  `hendrycks_math500`): no completed `generation_results.json` yet.
- Width 2048: only grouping metadata was written; its interrupted partial checkpoint
  was removed and must be re-run.

For width 512 and 1024, tune residual learning rate and/or schedule before treating
