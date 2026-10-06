#!/usr/bin/env python3
"""Generate a SYNTHETIC basketball free-throw biomechanics dataset (demo only).

The data exercises the pipeline and the feedback logic. It is NOT evidence of
real-world accuracy: every distribution here is an assumption, not a fit to data.

Usage:
    python generate_synthetic.py --n_shooters 12 --shots 10 --seed 42 --out synthetic/
Optional: --real features.csv   (your real file; only READ, never written)

Design in one paragraph: each shooter has a STYLE (shifts feature means, stays
inside functional ranges) and ~1 in 3 shooters also get ONE persistent FAULT
(pushes one feature outside its range). Per-shot values = shooter mean + MVN noise
+ mild fatigue drift. Curves are built first from those values, and the final
features are MEASURED from the curves (so curves and features always agree).
Labels come from the TRUE (pre-tracking-noise) values vs. functional ranges.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 60
SOURCE = "synthetic"
LABELS = ["good", "moderate", "poor"]
RANK = {l: i for i, l in enumerate(LABELS)}

FEATURES = ["elbow_angle_load", "elbow_angle_release", "knee_flex_min",
            "knee_angle_release", "followthrough_frames"]
FEATURE_COLS = FEATURES + ["knee_reliable_pct", "elbow_reliable_pct",
                           "shooter_id", "shot_num", "source"]
META_COLS = ["shooter_id", "view", "shot_num", "filepath", "made_shot", "arc_quality",
             "followthrough_quality", "elbow_quality", "knee_flex_quality", "notes",
             "start_sec", "end_sec", "source", "angle_deg", "fps", "wrist_habit",
             "style_profile"]
CURVE_COLS = ["shooter_id", "shot_num", "frame", "elbow_angle", "knee_angle"]

# Functional ranges = band where NO correction is needed (all ASSUMPTIONS).
RANGES = {"elbow_angle_load": (65, 105), "elbow_angle_release": (155, 178),
          "knee_flex_min": (105, 140), "knee_angle_release": (165, 180),
          "followthrough_frames": (8, 25)}
ARC_RANGE = RANGES["elbow_angle_release"]   # arc label uses an arc proxy vs this band
OUT_TOL = 0.15      # outside range by <= 15% of band width -> "moderate", else "poor"
RATER_FLIP = 0.08   # share of trait labels flipped at random (rater noise)

# Style = windows for the shooter MEAN of each feature (all inside functional ranges).
# ldur = typical frames of the load dip, extdur = frames of the extension phase.
STYLES = {
    "compact_deep_knee":         dict(kmin=(112, 122), erel=(160, 168), eload=(68, 82),
                                      krel=(166, 178), ft=(12, 17), ldur=26, extdur=14),
    "high_release_shallow_knee": dict(kmin=(130, 137), erel=(168, 174), eload=(90, 102),
                                      krel=(166, 178), ft=(14, 20), ldur=20, extdur=14),
    "smooth_balanced":           dict(kmin=(120, 130), erel=(164, 171), eload=(80, 92),
                                      krel=(166, 178), ft=(14, 18), ldur=22, extdur=14),
    "quick_release":             dict(kmin=(122, 132), erel=(162, 170), eload=(70, 84),
                                      krel=(166, 178), ft=(11, 14), ldur=15, extdur=10),
}
# Persistent faults: which variable, and the window the shooter mean is drawn from.
FAULTS = {"elbow_flare": ("erel", (145, 152)),          # release < 155
          "shallow_knee": ("kmin", (144, 152)),         # knee min > 142
          "short_followthrough": ("ft", (4.5, 6.0))}    # < 8 frames

# MVN variable order and correlations (within-shooter, shot to shot).
VARS = ["eload", "erel", "kmin", "krel", "ft", "ldur"]
CORR = np.eye(6)
for (a, b), r in {("eload", "erel"): .10, ("eload", "kmin"): .15, ("kmin", "krel"): .20,
                  ("kmin", "ldur"): -.45,   # deeper bend (lower angle) -> longer load
                  ("erel", "ft"): .20, ("erel", "krel"): .15, ("eload", "ldur"): -.15}.items():
    CORR[VARS.index(a), VARS.index(b)] = CORR[VARS.index(b), VARS.index(a)] = r
assert np.linalg.eigvalsh(CORR).min() > 0, "correlation matrix must be positive definite"


# ----------------------------------------------------------------- shooters
def draw_shooters(rng, n):
    """Draw per-shooter parameters. Re-draw (seeded) until between-shooter SD of the
    means exceeds within-shooter SD for every feature (needs a few shooters)."""
    names = list(STYLES)
    for attempt in range(1000):
        styles = [names[i % len(names)] for i in range(n)]
        rng.shuffle(styles)
        n_fault = int(round(n / 3))
        faulty = sorted(rng.choice(n, size=n_fault, replace=False).tolist()) if n_fault else []
        ftypes = [list(FAULTS)[i % len(FAULTS)] for i in range(n_fault)]
        rng.shuffle(ftypes)
        fault_of = dict(zip(faulty, ftypes))
        shooters = []
        for i in range(n):
            st = STYLES[styles[i]]
            mu = {k: rng.uniform(*st[k]) for k in ("eload", "erel", "kmin", "krel", "ft")}
            if i in fault_of:
                key, win = FAULTS[fault_of[i]]
                mu[key] = rng.uniform(*win)
            mu["ldur"] = st["ldur"] + rng.normal(0, 1.5)
            shooters.append(dict(
                shooter_id=f"shooter{101 + i}", style=styles[i], fault=fault_of.get(i, "none"),
                mu=mu, extdur=st["extdur"],
                sd=dict(eload=rng.uniform(6, 10), erel=rng.uniform(4, 6), kmin=rng.uniform(3, 5),
                        krel=3.0, ft=3.0, ldur=3.0),
                fatigue=rng.uniform(0.5, 1.5), skill=rng.normal(0, 0.6),
                arc_skill=rng.normal(0, 1.5),
                wrist=str(rng.choice(["neutral", "outward", "inward"], p=[.6, .2, .2]))))
        if n < 6:
            return shooters
        ok = True
        for k in ("eload", "erel", "kmin", "krel", "ft"):
            between = np.std([s["mu"][k] for s in shooters], ddof=1)
            within = np.sqrt(np.mean([s["sd"][k] ** 2 for s in shooters]))
            ok &= between >= 1.15 * within
        if ok:
            return shooters
    print("WARNING: could not satisfy between>within SD design; using last draw")
    return shooters


# ------------------------------------------------------------------ curves
def ease(a, b, n, p):
    """Cosine ease from a to b over n frames; p warps the timing (time-warp)."""
    s = np.linspace(0, 1, n) ** p
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * s))


def build_curves(rng, v, N, ldur, extdur, low_elbow, low_knee):
    """Elbow/knee curves: set pose -> load dip -> extension -> follow-through hold.
    Returns (elbow, knee, release_frame R). The elbow stays within 10 deg of its
    release angle for exactly v['ft'] frames (before jitter), then relaxes away."""
    ft = int(v["ft"])
    R = int(round(N * rng.uniform(0.48, 0.56)))
    R = min(R, N - 1 - ft - 14)                      # keep room for hold + tail
    Lm = R - extdur                                  # frame of deepest dip
    ds = max(Lm - ldur, 6)                           # dip start
    Lm = max(Lm, ds + 6)
    p1, p2 = rng.uniform(0.8, 1.25, 2)               # per-shot time warps

    e = np.empty(N)
    e_set = min(v["eload"] + rng.uniform(12, 25), v["erel"] - 20)
    e[:ds + 1] = e_set
    e[ds:Lm + 1] = ease(e_set, v["eload"], Lm - ds + 1, p1)
    e[Lm:R + 1] = ease(v["eload"], v["erel"], R - Lm + 1, p2)
    k = np.arange(N - R)
    d = np.where(k < ft, 4 * np.sin(np.pi * k / ft) - 7 * (k / ft) ** 2,   # |d| < 8 inside band
                 -12 - 1.2 * (k - ft))                                    # then leaves band
    e[R:] = np.minimum(v["erel"] + np.maximum(d, -45), 180)

    kn = np.empty(N)
    k_start = min(v["krel"] + rng.uniform(-3, 3), 180)
    dsk, Lmk = max(ds - 2, 2), Lm - 2                # knee bends a touch before the elbow
    kn[:dsk + 1] = k_start
    kn[dsk:Lmk + 1] = ease(k_start, v["kmin"], Lmk - dsk + 1, p1)
    kn[Lmk:R + 1] = ease(v["kmin"], v["krel"], R - Lmk + 1, p2)
    kn[R:] = v["krel"] - 2 * np.linspace(0, 1, N - R)

    for arr, low in ((e, low_elbow), (kn, low_knee)):  # tracking jitter (key frame R pinned)
        jit = np.clip(rng.normal(0, 2.5, N), -6, 6) if low else rng.uniform(-1, 1, N)
        jit[R] = 0
        arr += jit
    return np.clip(e, 0, 180), np.clip(kn, 0, 180), R


def measure(e, kn, R):
    """Extract the 5 features from the curves (release frame R is generator ground truth)."""
    erel = e[R]
    ft = 0
    while R + ft < len(e) and abs(e[R + ft] - erel) <= 10:
        ft += 1
    return dict(elbow_angle_load=e[:R + 1].min(), elbow_angle_release=erel,
                knee_flex_min=kn.min(), knee_angle_release=kn[R], followthrough_frames=ft)


# ------------------------------------------------------------------ labels
def band_label(x, lo, hi):
    """good = inside range; moderate = just outside; poor = well outside."""
    if lo <= x <= hi:
        return "good"
    dist = (lo - x) if x < lo else (x - hi)
    return "moderate" if dist <= OUT_TOL * (hi - lo) else "poor"


def worst(*labels):
    return max(labels, key=RANK.get)


def flip_labels(rng, labs):
    """Rater noise: ~8% of labels replaced by one of the other two."""
    labs = list(labs)
    for i in np.where(rng.random(len(labs)) < RATER_FLIP)[0]:
        labs[i] = str(rng.choice([l for l in LABELS if l != labs[i]]))
    return labs


def outside_penalty(true):
    """Sum of normalised distance outside each functional range (0 if inside)."""
    pen = 0.0
    for f, (lo, hi) in RANGES.items():
        x = true[f]
        pen += max(lo - x, x - hi, 0) / (hi - lo)
    return pen


def sample_made(rng, shooter_ids, skill, trues, arc_dev, target):
    """made_shot: weak logistic link to form + shooter skill, intercept calibrated to
    `target` make rate; resampled until the realised overall rate is in 45-65%."""
    rest = np.array([skill[s] - 1.2 * outside_penalty(t) + 0.04 * a
                     for s, t, a in zip(shooter_ids, trues, arc_dev)])
    lo, hi = -8.0, 8.0
    for _ in range(50):
        b0 = (lo + hi) / 2
        if (1 / (1 + np.exp(-(b0 + rest)))).mean() < target:
            lo = b0
        else:
            hi = b0
    p = 1 / (1 + np.exp(-(b0 + rest)))
    for _ in range(500):
        made = rng.random(len(p)) < p
        if 0.45 <= made.mean() <= 0.65:
            break
    return np.where(made, "made", "miss")


# -------------------------------------------------------------- generation
def generate(n_shooters, shots, seed):
    rng = np.random.default_rng(seed)
    shooters = draw_shooters(rng, n_shooters)
    feat_rows, meta_rows, curve_frames, trues, arc_devs, sids = [], [], [], [], [], []
    skill = {s["shooter_id"]: s["skill"] for s in shooters}
    tfrac = np.arange(shots) / max(shots - 1, 1)

    for s in shooters:
        mu = np.array([s["mu"][k] for k in VARS])
        sd = np.array([s["sd"][k] for k in VARS])
        draws = rng.multivariate_normal(mu, np.outer(sd, sd) * CORR, size=shots)
        f = s["fatigue"]   # mild fatigue drift: lower elbow release, shallower knee, shorter hold
        draws[:, 1] -= f * 2.5 * tfrac
        draws[:, 2] += f * 2.0 * tfrac
        draws[:, 4] -= f * 1.0 * tfrac
        for t in range(shots):
            true = dict(zip(VARS, draws[t]))
            true["eload"] = np.clip(true["eload"], 55, 115)
            true["erel"] = np.clip(true["erel"], 135, 179)
            true["krel"] = np.clip(true["krel"], 160, 180)
            true["kmin"] = np.clip(true["kmin"], 95, true["krel"] - 12)
            true["ft"] = int(np.clip(round(true["ft"]), 2, 30))
            ldur = int(np.clip(round(true["ldur"]), 8, 36))
            N = int(rng.integers(int(2.0 * FPS), int(2.6 * FPS) + 1))   # 120-156 frames

            # tracking reliability: ~10% of shots are poorly tracked (40-70%, noisier angles)
            low_e = low_k = False
            if rng.random() < 0.10:
                which = rng.choice(3, p=[.4, .4, .2])
                low_e, low_k = which in (0, 2), which in (1, 2)
            rel_e = rng.uniform(40, 70) if low_e else rng.uniform(90, 100)
            rel_k = rng.uniform(40, 70) if low_k else rng.uniform(90, 100)

            e, kn, R = build_curves(rng, true, N, ldur, s["extdur"], low_e, low_k)
            m = measure(e, kn, R)
            feat_rows.append({**{k: round(float(m[k]), 2) for k in FEATURES[:4]},
                              "followthrough_frames": int(m["followthrough_frames"]),
                              "knee_reliable_pct": round(rel_k, 1),
                              "elbow_reliable_pct": round(rel_e, 1),
                              "shooter_id": s["shooter_id"], "shot_num": t + 1, "source": SOURCE})

            # labels from TRUE values (what a rater sees), not from tracking-noisy measures
            tv = dict(elbow_angle_load=true["eload"], elbow_angle_release=true["erel"],
                      knee_flex_min=true["kmin"], knee_angle_release=true["krel"],
                      followthrough_frames=true["ft"])
            arc_proxy = true["erel"] + 0.25 * (true["krel"] - 172) + s["arc_skill"] + rng.normal(0, 2)
            L = lambda f: band_label(tv[f], *RANGES[f])
            labs = flip_labels(rng, [band_label(arc_proxy, *ARC_RANGE), L("followthrough_frames"),
                                     worst(L("elbow_angle_release"), L("elbow_angle_load")),
                                     worst(L("knee_flex_min"), L("knee_angle_release"))])
            trues.append(tv); sids.append(s["shooter_id"]); arc_devs.append(arc_proxy - 165)

            notes = []
            if s["wrist"] != "neutral" and rng.random() < 0.7:
                notes.append("wrist rotate " + s["wrist"] + "s")
            if low_e or low_k:
                notes.append("tracking dropout: " + ("elbow" if low_e and not low_k else
                             "knee" if low_k and not low_e else "elbow+knee"))
            start = round(t * 5 + rng.uniform(0.2, 2.0), 3)
            meta_rows.append(dict(
                shooter_id=s["shooter_id"], view="side", shot_num=t + 1, filepath="synthetic/none",
                made_shot=None, arc_quality=labs[0], followthrough_quality=labs[1],
                elbow_quality=labs[2], knee_flex_quality=labs[3], notes="; ".join(notes),
                start_sec=start, end_sec=round(start + N / FPS, 3), source=SOURCE, angle_deg=90,
                fps=FPS, wrist_habit=s["wrist"], style_profile=s["style"]))
            curve_frames.append(pd.DataFrame({
                "shooter_id": s["shooter_id"], "shot_num": t + 1, "frame": np.arange(N),
                "elbow_angle": np.round(e, 2), "knee_angle": np.round(kn, 2)}))

    made = sample_made(rng, sids, skill, trues, arc_devs, target=rng.uniform(0.50, 0.60))
    meta = pd.DataFrame(meta_rows)
    meta["made_shot"] = made
    feats = pd.DataFrame(feat_rows)[FEATURE_COLS]
    return shooters, feats, meta[META_COLS], pd.concat(curve_frames, ignore_index=True)[CURVE_COLS]


def reference_table():
    note = "ASSUMPTION - replace with cited literature value"
    return pd.DataFrame([{"feature": f, "functional_min": lo, "functional_max": hi, "note": note}
                         for f, (lo, hi) in RANGES.items()])


def personal_best(feats, meta):
    """Per shooter: mean of each feature over made AND all-four-labels-good shots.
    Falls back (made only -> all-good only -> all shots) if none qualify; see `basis`."""
    labs = ["arc_quality", "followthrough_quality", "elbow_quality", "knee_flex_quality"]
    d = feats.merge(meta[["shooter_id", "shot_num", "made_shot"] + labs], on=["shooter_id", "shot_num"])
    d["all_good"] = (d[labs] == "good").all(axis=1)
    d["made"] = d["made_shot"] == "made"
    rows = []
    for sid, g in d.groupby("shooter_id", sort=False):
        for basis, mask in (("made+all_good", g.made & g.all_good), ("made_only", g.made),
                            ("all_good_only", g.all_good), ("all_shots", g.made | ~g.made)):
            if mask.sum() > 0:
                break
        rows.append({"shooter_id": sid, **g.loc[mask, FEATURES].mean().round(2).to_dict(),
                     "n_template_shots": int(mask.sum()), "template_basis": basis, "source": SOURCE})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- self-checks
def self_checks(shooters, feats, meta, curves, tmpl):
    print("\n=== SELF-CHECKS (synthetic data) ===")
    print("\nPer-feature mean / SD:")
    print(feats[FEATURES + ["knee_reliable_pct", "elbow_reliable_pct"]].agg(["mean", "std"]).T.round(2).to_string())

    print("\nWithin vs between-shooter variation (ratio = between SD / within SD, want > 1):")
    rows = []
    for f in FEATURES:
        g = feats.groupby("shooter_id")[f]
        within, between = np.sqrt(g.var().mean()), g.mean().std()
        rows.append((f, round(within, 2), round(between, 2), round(between / within, 2),
                     round(between ** 2 / within ** 2, 2)))
    print(pd.DataFrame(rows, columns=["feature", "within_SD", "between_SD", "SD_ratio", "var_ratio"]).to_string(index=False))

    print("\nLabel distribution per trait (after ~8% rater noise):")
    for c in ["arc_quality", "followthrough_quality", "elbow_quality", "knee_flex_quality"]:
        print(f"  {c:22s}", meta[c].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict())
    print("\nmade_shot:", meta.made_shot.value_counts().to_dict(),
          f"(make rate {(meta.made_shot == 'made').mean():.1%})")
    print("\nPer-style shot counts:", meta.style_profile.value_counts().to_dict())
    print("Shooters per style:   ", pd.Series([s["style"] for s in shooters]).value_counts().to_dict())

    print("\nShooter ground truth (style / fault / measured mean of the faulted feature):")
    fm = feats.groupby("shooter_id")[FEATURES].mean().round(1)
    for s in shooters:
        extra = ""
        if s["fault"] != "none":
            var = {"elbow_flare": "elbow_angle_release", "shallow_knee": "knee_flex_min",
                   "short_followthrough": "followthrough_frames"}[s["fault"]]
            extra = f"  -> {var} mean {fm.loc[s['shooter_id'], var]} (range {RANGES[var]})"
        print(f"  {s['shooter_id']}  {s['style']:26s} fault={s['fault']:20s} wrist={s['wrist']}{extra}")

    dur = (meta.end_sec - meta.start_sec)
    checks = {
        "feature column order exact": list(feats.columns) == FEATURE_COLS,
        "metadata column order exact": list(meta.columns) == META_COLS,
        "no duplicate (shooter_id, shot_num) in features": not feats.duplicated(["shooter_id", "shot_num"]).any(),
        "no duplicate (shooter_id, shot_num) in metadata": not meta.duplicated(["shooter_id", "shot_num"]).any(),
        "no duplicate curve frames": not curves.duplicated(["shooter_id", "shot_num", "frame"]).any(),
        "angles within 0-180": bool(curves[["elbow_angle", "knee_angle"]].stack().between(0, 180).all()
                                    and feats[FEATURES[:4]].stack().between(0, 180).all()),
        "followthrough_frames integer >= 0": bool((feats.followthrough_frames >= 0).all()),
        "reliability within 0-100": bool(feats[["knee_reliable_pct", "elbow_reliable_pct"]].stack().between(0, 100).all()),
        "clip length 2.0-2.6 s (+/-0.002 rounding)": bool(dur.between(2.0 - 0.002, 2.6 + 0.002).all()),
        "clip frames match start/end": bool((curves.groupby(["shooter_id", "shot_num"]).size().values
                                             - (dur * FPS).round().values).__abs__().max() <= 1),
        "make rate 45-65%": 0.45 <= (meta.made_shot == "made").mean() <= 0.65,
        "labels valid": bool(meta[["arc_quality", "followthrough_quality", "elbow_quality",
                                   "knee_flex_quality"]].stack().isin(LABELS).all()),
        "ids start at shooter101 (never shooter01)": feats.shooter_id.str.match(r"^shooter1\d\d+$").all(),
        "personal-best template for every shooter": len(tmpl) == feats.shooter_id.nunique(),
    }
    print("\nChecks:")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    low = ((feats.knee_reliable_pct < 80) | (feats.elbow_reliable_pct < 80)).mean()
    print(f"  (info) shots with a poorly tracked joint: {low:.1%}")
    return all(checks.values())


# ------------------------------------------------------------------ report
def write_report(feats, shooters, out, real_path):
    real = None
    if real_path and os.path.exists(real_path):
        real = pd.read_csv(real_path)
    cols = [f for f in FEATURES if real is None or f in real.columns]
    rows, lines = [], []
    for f in cols:
        syn = feats[f]
        r = {"feature": f, "syn_mean": syn.mean(), "syn_sd": syn.std()}
        if real is not None:
            rr = pd.to_numeric(real[f], errors="coerce").dropna()
            r.update(real_mean=rr.mean(), real_sd=rr.std(), real_n=len(rr))
            z = (syn.mean() - rr.mean()) / rr.std() if rr.std() > 0 else np.nan
            ratio = syn.std() / rr.std() if rr.std() > 0 else np.nan
            where = "similar average" if abs(z) < 0.25 else (
                f"synthetic average is {'higher' if z > 0 else 'lower'} ({abs(z):.1f} real SDs)")
            spread = "similar spread" if 0.77 <= ratio <= 1.3 else (
                "synthetic is more spread out" if ratio > 1.3 else "synthetic is tighter")
            lines.append(f"- **{f}**: {where}; {spread} (SD ratio {ratio:.2f}).")
        rows.append(r)
    tbl = pd.DataFrame(rows).round(2)

    # figure: histograms synthetic vs real
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), cols):
        ax.hist(feats[f], bins=20, density=True, alpha=.55, label="synthetic")
        if real is not None:
            ax.hist(pd.to_numeric(real[f], errors="coerce").dropna(), bins=20, density=True,
                    alpha=.55, label="real")
        ax.set_title(f); ax.legend(fontsize=8)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("SYNTHETIC vs REAL feature distributions (synthetic data - demo only)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "synthetic_vs_real.png"), dpi=120)
    plt.close(fig)

    md = ["# Synthetic vs real features", "",
          "> **This dataset is SYNTHETIC.** It tests the pipeline and demonstrates the feedback logic. "
          "It is generated from assumed ranges and is **not** evidence of real-world accuracy.", ""]
    if real is None:
        md += [f"Real features file not found (`{real_path}`), so only synthetic statistics are shown.", ""]
    else:
        md += [f"Real data read from `{real_path}` ({len(real)} rows, read-only). "
               f"Synthetic: {len(feats)} rows, {feats.shooter_id.nunique()} shooters.", ""]
    md += ["## Per-feature mean / SD", "", tbl.to_markdown(index=False) if hasattr(tbl, "to_markdown")
           and _has_tabulate() else "```\n" + tbl.to_string(index=False) + "\n```", ""]
    md += ["## Plain-language summary", ""]
    if real is None:
        md += ["No real data to compare against. Re-run with `--real path/to/features.csv`."]
    else:
        md += lines + ["", "Differences are expected: the synthetic generator uses assumed styles, faults and "
                       "ranges, and was not fitted to the real data. Large gaps are a prompt to adjust the "
                       "generator assumptions (or to check the real data), not a measure of accuracy."]
        overlap = set(feats.shooter_id) & set(real.get("shooter_id", []))
        if overlap:
            md += ["", f"**WARNING:** shooter_id overlap with real data: {sorted(overlap)}"]
    md += ["", "## Ground-truth design (synthetic only)", "",
           "| shooter_id | style_profile | persistent fault | wrist_habit |", "|---|---|---|---|"]
    md += [f"| {s['shooter_id']} | {s['style']} | {s['fault']} | {s['wrist']} |" for s in shooters]
    with open(os.path.join(out, "synthetic_vs_real_report.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_shooters", type=int, default=12)
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synthetic/")
    ap.add_argument("--real", default="features.csv", help="your real features.csv (read-only)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    outputs = ["synthetic_features.csv", "synthetic_metadata.csv", "synthetic_angle_curves.csv",
               "reference_ranges.csv", "personal_best_templates.csv",
               "synthetic_vs_real_report.md", "synthetic_vs_real.png"]
    real_abs = os.path.abspath(a.real)
    if any(os.path.abspath(os.path.join(a.out, o)) == real_abs for o in outputs):
        sys.exit("Refusing to run: an output path would overwrite your real data file.")

    shooters, feats, meta, curves = generate(a.n_shooters, a.shots, a.seed)
    tmpl = personal_best(feats, meta)
    feats.to_csv(os.path.join(a.out, outputs[0]), index=False)
    meta.to_csv(os.path.join(a.out, outputs[1]), index=False)
    curves.to_csv(os.path.join(a.out, outputs[2]), index=False)
    reference_table().to_csv(os.path.join(a.out, outputs[3]), index=False)
    tmpl.to_csv(os.path.join(a.out, outputs[4]), index=False)
    write_report(feats, shooters, a.out, a.real)
    ok = self_checks(shooters, feats, meta, curves, tmpl)
    print(f"\nWrote {len(outputs)} files to {a.out}  (all data SYNTHETIC; source='synthetic')")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
