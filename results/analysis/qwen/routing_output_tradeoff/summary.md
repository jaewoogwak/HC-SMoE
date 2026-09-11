# Routing/output trade-off (held-out C4)

## A. Final trade-off

| alpha | raw-acc avg | held-out Mean U | U=1 | local relL2 | cosine |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.5218 | 2.8617 | 0.0929 | 0.3299 | 0.8985 |
| 0.5 | 0.5454 | 2.7519 | 0.1680 | 0.3518 | 0.8802 |
| 1.0 | 0.3290 | 2.7203 | 0.1946 | 0.3993 | 0.8553 |

## B. alpha=0.5 vs alpha=1 pair mechanism

| category | pairs | hc mean L2 | coactivation | Jaccard | active-union relL2 | corouted relL2 |
|---|---:|---:|---:|---:|---:|---:|
| routing_only | 6847 | 4.0575 | 0.0041 | 0.0335 | 1.4614 | 1.4586 |
| hybrid_only_vs_routing | 6878 | 2.2896 | 0.0037 | 0.0292 | 1.4512 | 1.4494 |
| shared_05_1 | 4004 | 2.4078 | 0.0056 | 0.0460 | 1.4472 | 1.4416 |

## C. alpha=0 vs alpha=0.5 pair mechanism

| category | pairs | hc mean L2 | coactivation | Jaccard | active-union relL2 | corouted relL2 |
|---|---:|---:|---:|---:|---:|---:|
| hc_only | 6190 | 1.9218 | 0.0031 | 0.0246 | 1.4498 | 1.4462 |
| hybrid_only_vs_hc | 6055 | 2.6428 | 0.0045 | 0.0368 | 1.4537 | 1.4500 |
| shared_0_05 | 4827 | 1.9447 | 0.0043 | 0.0337 | 1.4447 | 1.4426 |

## Representative pairs

### mean output false friends
- L0 E0/E16: hc=0.1518, coact=0.0033, union-relL2=1.4129, corouted-relL2=1.4053
- L0 E0/E29: hc=0.1539, coact=0.0045, union-relL2=1.4080, corouted-relL2=1.3979
- L0 E0/E13: hc=0.1541, coact=0.0043, union-relL2=1.4120, corouted-relL2=1.4002

### hybrid compatible pairs
- L16 E30/E53: hc=6.8280, coact=0.0675, union-relL2=1.3660, corouted-relL2=1.3523
- L13 E9/E38: hc=2.2310, coact=0.0609, union-relL2=1.3958, corouted-relL2=1.3879
- L16 E2/E8: hc=3.1994, coact=0.0574, union-relL2=1.3727, corouted-relL2=1.3622

### routing only harmful pairs
- L6 E23/E40: hc=2.6012, coact=0.0600, union-relL2=1.3628, corouted-relL2=1.3488
- L8 E37/E49: hc=2.2516, coact=0.0582, union-relL2=1.3927, corouted-relL2=1.3874
- L15 E17/E50: hc=7.6963, coact=0.0579, union-relL2=1.3566, corouted-relL2=1.3266
