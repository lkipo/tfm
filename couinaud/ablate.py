#!/usr/bin/env python
"""
Measure what each change to the assignment rule actually buys.

Three things were added to the nearest-seed rule -- a geodesic metric, Murray flow
weighting, and a hepatic-vein barrier -- and all three default to off, so the
question is not whether they work but whether they are worth switching on. This runs
each of them over the cohort and puts the answers side by side.

    python ablate.py                    # the standard set of arms
    python ablate.py --arms base geo    # just those two
    python ablate.py --cases 6          # a quick subset while iterating

What the columns mean:

  sector err   median |log((VI+VII)/(V+VIII) / 0.83)|, against Leelaudomlipi's
               published means. This is the headline: it is the quantity the
               coverage bias distorts, and the one the flow weighting targets.
  rho          the coverage bias itself -- how strongly the volume a pedicle wins
               tracks how many skeleton points it happens to have. Lower is better;
               0 would mean anatomy alone decides.
  Dice RHV     independent cross-validation. The vein-barrier arms hold the RHV out
               of the growth cost precisely so this column stays honest for them;
               MHV and LHV are spent there and are reported as such.
  8 segs       cases where all eight segments are non-empty.
  1 blob       cases where every segment is a single connected component. A monotone
               front inside a connected mask cannot produce a disconnected basin, so
               the geodesic arms should be perfect here by construction -- if they
               are not, the fault is in the seed labelling, not the assignment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from checks import MIN_LARGEST_CC, _largest_cc_fraction, run_checks
from couinaud import SEG_IDS, Config, load_case, missing_labels, segment_liver
from run_couinaud import DATA_ROOT, OUT, usable_cases

REFERENCE_SECTOR_RATIO = (8.4 + 14.9) / (10.6 + 17.6)   # Leelaudomlipi (VI+VII)/(V+VIII)

# Each arm is one configuration. They are cumulative on purpose: every row differs
# from the one above it by a single change, so a column that moves can be attributed.
ARMS: dict[str, tuple[str, Config]] = {
    "base": ("straight-line nearest seed (today's default)", Config()),
    "geo": ("geodesic: the front stays inside the liver",
            Config(assign="geodesic")),
    "geo+flow50": ("geodesic + Murray weighting at alpha=0.50 (half-strength)",
                   Config(assign="geodesic", flow_alpha=0.50)),
    "geo+flow100": ("geodesic + Murray weighting at alpha=1.00 (volume ~ r^3)",
                    Config(assign="geodesic", flow_alpha=1.00)),
    "geo+flow150": ("geodesic + Murray weighting at alpha=1.50 (overshoot, as a control)",
                    Config(assign="geodesic", flow_alpha=1.50)),
    "geo+vein2": ("geodesic + vein barrier x2, RHV held out",
                  Config(assign="geodesic", vein_barrier=2.0, vein_barrier_holdout="rhv")),
    "geo+vein5": ("geodesic + vein barrier x5, RHV held out",
                  Config(assign="geodesic", vein_barrier=5.0, vein_barrier_holdout="rhv")),
    "all": ("geodesic + alpha=0.50 + vein barrier x2, RHV held out",
            Config(assign="geodesic", flow_alpha=0.50, vein_barrier=2.0,
                   vein_barrier_holdout="rhv")),
}
DEFAULT_ARMS = list(ARMS)


def sector_ratio(pct: dict[int, float]) -> float | None:
    ant, post = pct[5] + pct[8], pct[6] + pct[7]
    return post / ant if ant > 0 and post > 0 else None


def seed_ratio(notes: list[str]) -> float | None:
    note = next((n for n in notes if n.strip().startswith("pedicle skeletons:")), None)
    if note is None:
        return None
    counts = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", note)}
    ant, post = counts.get("right_ant", 0), counts.get("right_post", 0)
    return post / ant if ant >= 5 and post >= 5 else None


def run_arm(name: str, cfg: Config, cases: list[Path]) -> dict:
    """One configuration over every case. Failures are recorded, never swallowed."""
    per_case, failures = [], []
    t0 = time.time()
    for path in cases:
        try:
            case = load_case(path)
            res = segment_liver(case, cfg, verbose=False)
            checks = run_checks(case, res, reference=res)
            pct = res.volume_pct()
            per_case.append({
                "case": case.name,
                "volume_pct": {str(s): pct[s] for s in SEG_IDS},
                "sector_ratio": sector_ratio(pct),
                "seed_ratio": seed_ratio(res.notes),
                "metrics": checks["metrics"],
                "rows": {n: ok for n, ok, _ in checks["rows"]},
                "spent_veins": sorted(res.veins_used()),
                "empty_segments": [s for s in SEG_IDS if not (res.labels == s).any()],
                "fragmented": [s for s in SEG_IDS if (res.labels == s).any()
                               and _largest_cc_fraction(res.labels == s) < MIN_LARGEST_CC],
            })
        except Exception as exc:                            # noqa: BLE001
            failures.append((path.name, repr(exc)))
            print(f"    !! {path.name}: {exc}", file=sys.stderr)
    return {"arm": name, "config": cfg.__dict__.copy(), "cases": per_case,
            "failures": failures, "seconds": time.time() - t0}


def summarise(arm: dict) -> dict:
    cs = arm["cases"]
    if not cs:
        return {}
    ratios = [c["sector_ratio"] for c in cs if c["sector_ratio"]]
    err = [abs(np.log(r / REFERENCE_SECTOR_RATIO)) for r in ratios]

    paired = [(c["seed_ratio"], c["sector_ratio"]) for c in cs
              if c["seed_ratio"] and c["sector_ratio"]]
    rho = (float(np.corrcoef(np.log([p[0] for p in paired]),
                             np.log([p[1] for p in paired]))[0, 1])
           if len(paired) >= 5 else float("nan"))

    def med(key):
        v = [c["metrics"][key] for c in cs if key in c["metrics"]]
        return float(np.median(v)) if v else float("nan")

    spent = set().union(*[set(c["spent_veins"]) for c in cs]) if cs else set()
    return {
        "n": len(cs),
        "sector_err": float(np.median(err)) if err else float("nan"),
        "sector_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "rho": rho,
        "dice_rhv": med("dice_right_sector_vs_rhv"),
        "dice_mhv": med("dice_cantlie_vs_mhv"),
        "dice_lhv": med("dice_segment_iv_vs_lhv"),
        "all_eight": sum(1 for c in cs if not c["empty_segments"]),
        "one_blob": sum(1 for c in cs if not c["fragmented"]),
        # by prefix: the check names carry their thresholds, which are free to change
        "in_band": sum(1 for c in cs for name, ok in c["rows"].items()
                       if ok and name.startswith("segment volumes inside")),
        "spent": sorted(spent),
        "seconds_per_case": arm["seconds"] / len(cs),
    }


def report(arms: list[dict]) -> None:
    rows = [(a["arm"], summarise(a)) for a in arms]
    n = max(s["n"] for _, s in rows if s)

    def cell(s, key, fmt, spent_key=None):
        width = int(fmt.lstrip(">").split(".")[0])
        if spent_key and spent_key in s["spent"]:
            return f"{'spent':>{width}}"
        v = s[key]
        return f"{'-':>{width}}" if not np.isfinite(v) else format(v, fmt)

    print("\n" + "=" * 96)
    print(f"{'arm':<12}{'sector err':>11}{'ratio':>8}{'rho':>7}{'DiceRHV':>9}"
          f"{'DiceMHV':>9}{'DiceLHV':>9}{'8 segs':>8}{'1 blob':>8}{'band':>7}{'s/case':>8}")
    print("-" * 96)
    for name, s in rows:
        if not s:
            print(f"{name:<12}  (no cases completed)")
            continue
        print(f"{name:<12}{s['sector_err']:>11.3f}{s['sector_ratio']:>8.2f}{s['rho']:>7.2f}"
              f"{cell(s, 'dice_rhv', '>9.3f', 'rhv')}"
              f"{cell(s, 'dice_mhv', '>9.3f', 'mhv')}"
              f"{cell(s, 'dice_lhv', '>9.3f', 'lhv')}"
              f"{s['all_eight']:>5}/{s['n']:<2}{s['one_blob']:>5}/{s['n']:<2}"
              f"{s['in_band']:>4}/{s['n']:<2}{s['seconds_per_case']:>8.1f}")
    print("-" * 96)
    print(f"{n} cases. 'sector err' is the median |log| distance of "
          f"(VI+VII)/(V+VIII) from the published {REFERENCE_SECTOR_RATIO:.2f} -- "
          f"lower is better.")
    print("'spent' means that vein fed the assignment, so its Dice would be high by "
          "construction and is refused.")
    for name, _ in rows:
        print(f"  {name:<12} {ARMS[name][0]}")
    fails = [(a["arm"], f) for a in arms for f in a["failures"]]
    if fails:
        print("\nfailures:")
        for arm, (case, exc) in fails:
            print(f"  {arm:<12} {case}: {exc}")


def render_figure(case_name: str, arm_names: list[str]) -> int:
    """
    Draw one case under each arm, with the baseline's boundary overlaid.

    The table says how much moved; this says *where*, which is the only way to tell a
    boundary that improved from one that merely shifted.
    """
    import viz
    from couinaud import VEINS

    case = load_case(DATA_ROOT / case_name)
    results = {}
    for name in arm_names:
        results[name] = segment_liver(case, ARMS[name][1], verbose=False)
        print(f"  {name}: done")
    veins = {k: case.masks[cls] for k, cls in VEINS.items() if case.has(cls)}
    OUT.mkdir(exist_ok=True)
    path = viz.figure_assignment(case, results, OUT / f"{case_name}_assignment.png",
                                 veins=veins)
    print(f"figure -> {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--cases", type=int, default=0, help="use only the first N cases")
    ap.add_argument("--out", default=str(OUT / "ablation.json"))
    ap.add_argument("--figure", metavar="CASE",
                    help="also render the arms side by side for this one case")
    args = ap.parse_args()

    if args.figure:
        return render_figure(args.figure, args.arms)

    cases = usable_cases()
    if args.cases:
        cases = cases[:args.cases]
    print(f"{len(cases)} cases x {len(args.arms)} arms\n")

    arms = []
    for name in args.arms:
        desc, cfg = ARMS[name]
        print(f"--- {name}: {desc}")
        arm = run_arm(name, cfg, cases)
        arms.append(arm)
        s = summarise(arm)
        print(f"    {len(arm['cases'])} ok, {len(arm['failures'])} failed, "
              f"{arm['seconds']:.0f}s"
              + (f", sector err {s['sector_err']:.3f}, rho {s['rho']:+.2f}" if s else ""))

    report(arms)
    OUT.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"arms": arms, "summary": {a["arm"]: summarise(a) for a in arms}},
        indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
