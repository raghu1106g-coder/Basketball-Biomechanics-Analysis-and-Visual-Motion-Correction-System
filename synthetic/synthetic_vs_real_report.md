# Synthetic vs real feature comparison

**All 'synthetic' rows are SYNTHETIC (simulated). They test the pipeline and feedback logic and are NOT evidence of real-world accuracy.**

Real file `features.csv` not found, so only synthetic statistics are shown.

| feature | synth mean | synth SD |
|---|---|---|
| elbow_angle_load | 88.00 | 12.79 |
| elbow_angle_release | 162.58 | 10.07 |
| knee_flex_min | 129.04 | 11.16 |
| knee_angle_release | 171.27 | 4.60 |
| followthrough_frames | 14.47 | 5.32 |
| knee_reliable_pct | 92.60 | 10.07 |
| elbow_reliable_pct | 92.60 | 10.69 |

## Where they differ (plain language)

No real data supplied, so no differences can be described.

Differences are expected: synthetic values come from assumed ranges, style/fault profiles and normal noise, not from real athletes. Tune the assumptions if the real data shows a different spread.
