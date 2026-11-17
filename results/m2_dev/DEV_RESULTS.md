# Real dev metrics: M2

> Generated strictly from REAL model checkpoints/run.json, not from synthetic smoke or test.

| Model | Dev macro-F1 mean ± SD | Dev accuracy mean ± SD | Trainable params |
|---|---:|---:|---:|
| image | 0.5746 ± 0.0068 | 0.6930 ± 0.0077 | 66,309 |
| text | 0.6811 ± 0.0016 | 0.7762 ± 0.0064 | 99,077 |
| concat | 0.8116 ± 0.0041 | 0.8540 ± 0.0044 | 197,637 |

**Limits:** dev-only; group-multimodal leakage controlled by locked manifests; rare class has only 9 dev examples. Do not make out-of-event generalization claims.
