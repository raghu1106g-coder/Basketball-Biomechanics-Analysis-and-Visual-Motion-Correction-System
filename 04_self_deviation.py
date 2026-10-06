"""
Step 5: Self-Comparison Deviation Report
--------------------------------------------
With a single player's shots, this computes each shot's deviation from
THIS PLAYER'S OWN average -- a same-player consistency check. This stands
in for the multi-shooter expert-template comparison until more shooters'
data is collected (noted as future work in the PPT).
"""
import pandas as pd

df = pd.read_csv("features.csv")

feature_cols = [
    "elbow_angle_load", "elbow_angle_release",
    "knee_flex_min", "knee_angle_release",
    "followthrough_frames",
]

mean = df[feature_cols].mean()
std = df[feature_cols].std().replace(0, 1e-6)

print("Player's own average (across 6 shots):")
print(mean.round(2))
print("\nPer-shot deviation (z-score from own average):\n")

report_rows = []
for _, row in df.iterrows():
    z = {col: round((row[col] - mean[col]) / std[col], 2) for col in feature_cols}
    report_rows.append({"shot_num": row["shot_num"], **z})
    flagged = [c for c, v in z.items() if abs(v) > 1.0]
    flag_str = f"  <-- inconsistent: {', '.join(flagged)}" if flagged else ""
    print(f"Shot {row['shot_num']}: {z}{flag_str}")

pd.DataFrame(report_rows).to_csv("self_deviation_report.csv", index=False)
print("\nSaved self_deviation_report.csv")
print("\nNote: this is a same-player consistency check, not a multi-shooter")
print("expert-template comparison -- that requires more players' data (future work).")