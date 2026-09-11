# Routing/output trade-off (held-out C4)

## A. Final trade-off

| alpha | raw-acc avg | held-out Mean U | U=1 | local relL2 | cosine |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.5978 | 1.6750 | 0.3250 | 0.4875 | 0.8343 |
| 0.5 | 0.6071 | 1.5832 | 0.4168 | 0.4717 | 0.8338 |
| 1.0 | 0.3645 | 1.5732 | 0.4268 | 0.4745 | 0.8291 |

## B. alpha=0.5 vs alpha=1 pair mechanism

| category | pairs | hc mean L2 | coactivation | Jaccard | active-union relL2 | corouted relL2 |
|---|---:|---:|---:|---:|---:|---:|
| routing_only | 162 | 7.6951 | 0.0394 | 0.0872 | 1.1306 | 1.0881 |
| hybrid_only_vs_routing | 153 | 5.9295 | 0.0397 | 0.0864 | 1.1057 | 1.0708 |
| shared_05_1 | 144 | 4.4258 | 0.0505 | 0.1151 | 1.1114 | 1.0733 |

## C. alpha=0 vs alpha=0.5 pair mechanism

| category | pairs | hc mean L2 | coactivation | Jaccard | active-union relL2 | corouted relL2 |
|---|---:|---:|---:|---:|---:|---:|
| hc_only | 101 | 5.8450 | 0.0357 | 0.0778 | 1.1099 | 1.0723 |
| hybrid_only_vs_hc | 155 | 5.9580 | 0.0422 | 0.0929 | 1.1067 | 1.0691 |
| shared_0_05 | 142 | 4.3736 | 0.0479 | 0.1085 | 1.1105 | 1.0751 |

## Representative pairs

### mean output false friends
- L1 E5/E7: hc=0.1442, coact=0.0500, union-relL2=1.0040, corouted-relL2=0.9615
- L1 E2/E7: hc=0.1557, coact=0.0259, union-relL2=1.0128, corouted-relL2=0.9192
- L1 E1/E7: hc=0.1640, coact=0.0319, union-relL2=1.0443, corouted-relL2=0.9580

### hybrid compatible pairs
- L31 E2/E7: hc=81.9356, coact=0.1545, union-relL2=0.3225, corouted-relL2=0.4243
- L28 E0/E4: hc=11.7651, coact=0.1105, union-relL2=0.9669, corouted-relL2=0.9078
- L15 E4/E7: hc=2.4509, coact=0.1082, union-relL2=1.1878, corouted-relL2=1.1498

### routing only harmful pairs
- L29 E4/E5: hc=18.3182, coact=0.1351, union-relL2=0.8424, corouted-relL2=0.8179
- L15 E2/E6: hc=2.8238, coact=0.1097, union-relL2=1.1633, corouted-relL2=1.1439
- L14 E0/E3: hc=2.5500, coact=0.1092, union-relL2=1.1930, corouted-relL2=1.1572
