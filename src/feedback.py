"""
Module 7 — Feedback Generator
===============================
Convert biomechanical fault list into coach-style natural-language feedback.

CLI usage:
    python -m src.feedback --faults faults.json
"""

import argparse
import json
import os
import sys


# ── Template dictionary: (joint, phase) → coaching advice ──────────────────

COACHING_TEMPLATES = {
    # ── Elbow ──
    ("elbow", "load", "low"): (
        "Your shooting elbow drops to {actual_value}° during the load phase — "
        "aim for {lo}°–{hi}° at the set point. Try keeping your elbow tucked "
        "under the ball and aligned with your shooting shoulder."
    ),
    ("elbow", "load", "high"): (
        "Your elbow is at {actual_value}° during the load phase, which is "
        "higher than the ideal {lo}°–{hi}°. A slightly lower elbow at the "
        "set point helps generate a smoother upward release."
    ),
    ("elbow", "release", "default"): (
        "At release, your elbow angle is {actual_value}° — work on fully "
        "extending your arm to get a clean follow-through."
    ),

    # ── Knee ──
    ("knee", "load", "low"): (
        "Your knee bend reaches {actual_value}°, deeper than the "
        "{lo}°–{hi}° target. Too much knee flexion can reduce shot "
        "consistency. Focus on a controlled half-squat dip."
    ),
    ("knee", "load", "high"): (
        "Your knees only flex to {actual_value}°, above the {lo}°–{hi}° "
        "range. Bend a bit more to generate power from your legs — your "
        "shot will feel less arm-dependent."
    ),

    # ── Release angle ──
    ("forearm", "release", "low"): (
        "Your forearm release angle is {actual_value}°, flatter than the "
        "{lo}°–{hi}° ideal. Try finishing with your hand higher above "
        "your head to get more arc on the ball."
    ),
    ("forearm", "release", "high"): (
        "Your forearm release angle is {actual_value}°, steeper than {lo}°–{hi}°. "
        "A slightly lower release point will flatten your trajectory and "
        "improve accuracy at distance."
    ),

    # ── Follow-through ──
    ("wrist", "followthrough", "default"): (
        "Your follow-through is only {actual_value} frames — hold your "
        "finish longer! A full follow-through (wrist snapped, fingers "
        "pointed at the rim) helps with touch and consistency."
    ),

    # ── Shoulder ──
    ("shoulder", "release", "default"): (
        "Your shooting shoulder flares to {actual_value}° of abduction — "
        "try keeping your elbow more in-line with your hip and shoulder. "
        "Less lateral motion means a straighter trajectory."
    ),
}

# Fallback template for any uncovered combination
FALLBACK_TEMPLATE = (
    "During the {phase} phase, your {joint} measured {actual_value}° "
    "which is outside the target range of {lo}–{hi}. Focus on keeping "
    "this joint within the recommended range for better consistency."
)


def generate_feedback(faults, include_summary=True):
    """
    Convert a list of fault dicts into coaching feedback text.

    Parameters
    ----------
    faults : list of dict  — each with keys:
        joint, phase, severity, actual_value, expected_range, description

    Returns
    -------
    str — multi-paragraph coaching feedback
    """
    if not faults:
        return ("Great shot! No significant biomechanical faults detected. "
                "Your form looks consistent with the template. Keep practising "
                "to lock in this muscle memory.")

    lines = []

    if include_summary:
        severe = sum(1 for f in faults if f.get("severity") == "severe")
        moderate = sum(1 for f in faults if f.get("severity") == "moderate")
        mild = sum(1 for f in faults if f.get("severity") == "mild")
        lines.append("=== Coaching Feedback ===\n")
        lines.append("Found {} area(s) to improve ({} severe, {} moderate, {} mild).\n".format(
            len(faults), severe, moderate, mild))

    for i, fault in enumerate(faults, 1):
        joint = fault.get("joint", "unknown")
        phase = fault.get("phase", "unknown")
        severity = fault.get("severity", "mild")
        actual = fault.get("actual_value", "?")
        expected = fault.get("expected_range", [None, None])
        lo = expected[0] if expected[0] is not None else "?"
        hi = expected[1] if expected[1] is not None else "?"

        # Determine direction (low or high relative to range)
        direction = "default"
        if isinstance(actual, (int, float)) and isinstance(lo, (int, float)):
            direction = "low" if actual < lo else "high"

        # Look up template
        key = (joint, phase, direction)
        template = COACHING_TEMPLATES.get(key)
        if template is None:
            key_fallback = (joint, phase, "default")
            template = COACHING_TEMPLATES.get(key_fallback, FALLBACK_TEMPLATE)

        text = template.format(
            joint=joint, phase=phase, severity=severity,
            actual_value=actual, lo=lo, hi=hi,
        )

        severity_emoji = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}.get(severity, "⚪")
        lines.append("{}  {}. [{}] {}".format(severity_emoji, i, severity.upper(), text))
        lines.append("")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate coaching feedback from faults.")
    parser.add_argument("--faults", type=str, required=True,
                        help="Path to faults JSON file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output text file (default: print to stdout).")
    args = parser.parse_args()

    with open(args.faults) as f:
        data = json.load(f)
    faults_list = data.get("faults", data if isinstance(data, list) else [])

    feedback = generate_feedback(faults_list)
    print(feedback)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(feedback)
        print("\nSaved to {}".format(args.output))


if __name__ == "__main__":
    main()
