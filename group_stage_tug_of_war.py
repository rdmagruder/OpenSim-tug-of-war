"""
Group-stage evaluator for the OpenSim Tug-of-War competition (Python, OpenSim 4.x)

Update: Control file handling
-----------------------------
Now supports student-submitted XML files that may already contain both Left and Right controls.
Logic:
 • If the XML has a control explicitly named "LeftMuscle.excitation", we take that for Left side.
 • If the XML has only one ControlLinear, we use it regardless of name.
 • If multiple controls exist but no LeftMuscle.excitation, we fall back to the first ControlLinear.
"""
from __future__ import annotations
import os
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import opensim as osim

# ------------------------- CONFIG -------------------------
BASE_MODEL_PATH = Path(r"C:/Users/<you>/Documents/OpenSim/4.4/Resources/Models/Tug_of_War/Tug_of_War_Millard.osim")
ENTRIES_DIR     = Path("entries")
WORK_DIR        = Path("group_stage_outputs")
SIM_DURATION    = 1.0
NUM_GROUPS      = 8
RESULTS_CSV     = WORK_DIR / "group_stage_results.csv"
# ----------------------------------------------------------

@dataclass
class Entry:
    name: str
    model_path: Path
    controls_path: Path

@dataclass
class MatchResult:
    left: str
    right: str
    final_tz: float
    winner: str | None
    out_dir: Path


def find_entries(entries_dir: Path) -> List[Entry]:
    entries: List[Entry] = []
    for student_dir in sorted(p for p in entries_dir.iterdir() if p.is_dir()):
        osims = list(student_dir.glob("*.osim"))
        ctrls = list(student_dir.glob("*.xml"))
        if len(osims) != 1 or len(ctrls) != 1:
            print(f"[WARN] '{student_dir.name}': expected 1 .osim and 1 .xml; found {len(osims)} and {len(ctrls)}. Skipping.")
            continue
        entries.append(Entry(name=student_dir.name, model_path=osims[0], controls_path=ctrls[0]))
    if not entries:
        raise RuntimeError(f"No valid entries found under {entries_dir.resolve()}.")
    return entries


def group_students(entries: List[Entry], k_groups: int) -> Dict[str, List[Entry]]:
    groups: Dict[str, List[Entry]] = {chr(ord('A')+i): [] for i in range(k_groups)}
    for idx, e in enumerate(sorted(entries, key=lambda x: x.name.lower())):
        g = chr(ord('A') + (idx % k_groups))
        groups[g].append(e)
    return groups


# ---------- OpenSim helpers ----------

def _copy_student_muscle_params(src_model: osim.Model, dst_model: osim.Model, dst_muscle_name: str, state: osim.State) -> None:
    src_muscle_base = src_model.getMuscles().get('LeftMuscle')
    Millard = osim.Millard2012EquilibriumMuscle
    src_m = Millard.safeDownCast(src_muscle_base)
    dst_muscle_base = dst_model.getMuscles().get(dst_muscle_name)
    dst_m = Millard.safeDownCast(dst_muscle_base)
    # Copy key properties
    dst_m.set_max_isometric_force(src_m.get_max_isometric_force())
    dst_m.set_optimal_fiber_length(src_m.get_optimal_fiber_length())
    dst_m.set_tendon_slack_length(src_m.get_tendon_slack_length())
    dst_m.set_pennation_angle_at_optimal(src_m.get_pennation_angle_at_optimal())
    dst_m.set_max_contraction_velocity(src_m.get_max_contraction_velocity())
    dst_m.set_activation_time_constant(src_m.get_activation_time_constant())
    dst_m.set_deactivation_time_constant(src_m.get_deactivation_time_constant())
    _ = dst_m.getGeometryPath().getLength(state)


def _extract_control(cs: osim.ControlSet, prefer_left: bool) -> osim.ControlLinear:
    candidates = [osim.ControlLinear.safeDownCast(cs.get(i)) for i in range(cs.getSize())]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        raise RuntimeError("No ControlLinear found in ControlSet.")
    if prefer_left:
        for c in candidates:
            if c.getName() == "LeftMuscle.excitation":
                return c
    return candidates[0]


def _merge_controls(left_ctrl_path: Path, right_ctrl_path: Path, out_path: Path) -> None:
    left_cs  = osim.ControlSet(str(left_ctrl_path))
    right_cs = osim.ControlSet(str(right_ctrl_path))

    cl = _extract_control(left_cs, prefer_left=True)
    cr = _extract_control(right_cs, prefer_left=False)

    merged = osim.ControlSet()
    cl_new = osim.ControlLinear(cl)
    cl_new.setName("LeftMuscle.excitation")
    merged.adoptAndAppend(cl_new)

    cr_new = osim.ControlLinear(cr)
    cr_new.setName("RightMuscle.excitation")
    merged.adoptAndAppend(cr_new)

    doc = osim.XMLDocument()
    doc.setRootElement(merged)
    doc.write(str(out_path))


def _run_forward(model_path: Path, controls_xml: Path, out_dir: Path, t0: float = 0.0, tf: float = SIM_DURATION) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    tool = osim.ForwardTool()
    tool.setModelFilename(str(model_path))
    tool.setControlsFileName(str(controls_xml))
    tool.setResultsDir(str(out_dir))
    tool.setInitialTime(t0)
    tool.setFinalTime(tf)
    tool.setSolveForEquilibriumForAuxiliaryStates(True)
    tool.run()

    states_files = list(out_dir.glob(f"*states.sto"))
    if not states_files:
        raise RuntimeError(f"No states.sto found in {out_dir}")
    states_path = sorted(states_files)[-1]
    table = osim.TimeSeriesTable(str(states_path))
    labels = list(table.getColumnLabels())
    tz_cols = [i for i, s in enumerate(labels) if 'block_tz' in s]
    final_row = table.getNumRows() - 1
    final_tz = table.getMatrix().get(final_row, tz_cols[0])
    return float(final_tz)


def play_match(left: Entry, right: Entry, base_model: Path, arena_dir: Path) -> MatchResult:
    match_dir = arena_dir / f"{left.name}_vs_{right.name}"
    match_dir.mkdir(parents=True, exist_ok=True)
    model = osim.Model(str(base_model))
    state = model.initSystem()

    left_model  = osim.Model(str(left.model_path)); left_model.initSystem()
    right_model = osim.Model(str(right.model_path)); right_model.initSystem()

    _copy_student_muscle_params(left_model,  model, 'LeftMuscle',  state)
    _copy_student_muscle_params(right_model, model, 'RightMuscle', state)

    combined_model_path = match_dir / "Combined_Tug_of_War.osim"
    model.print(str(combined_model_path))

    merged_controls_path = match_dir / "Combined_controls.xml"
    _merge_controls(left.controls_path, right.controls_path, merged_controls_path)

    final_tz = _run_forward(combined_model_path, merged_controls_path, match_dir)
    winner = left.name if final_tz > 0 else (right.name if final_tz < 0 else None)

    return MatchResult(left=left.name, right=right.name, final_tz=final_tz, winner=winner, out_dir=match_dir)


def run_group_stage():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    entries = find_entries(ENTRIES_DIR)
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASE_MODEL_PATH}")
    groups = group_students(entries, NUM_GROUPS)

    wins: Dict[str, int] = {}
    margin: Dict[str, float] = {}

    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["group", "left", "right", "final_block_tz", "winner", "match_dir"])

        for gname, members in groups.items():
            print(f"\n=== Group {gname} ({len(members)} students) ===")
            group_dir = WORK_DIR / f"Group_{gname}"
            group_dir.mkdir(exist_ok=True)
            for m in members:
                wins.setdefault(m.name, 0)
                margin.setdefault(m.name, 0.0)

            for left, right in itertools.combinations(members, 2):
                res = play_match(left, right, BASE_MODEL_PATH, group_dir)
                print(f"{res.left} vs {res.right} -> tz = {res.final_tz:.4f} -> winner: {res.winner}")
                w.writerow([gname, res.left, res.right, f"{res.final_tz:.6f}", res.winner or "TIE", str(res.out_dir)])
                if res.winner: wins[res.winner] += 1
                margin[res.left]  +=  res.final_tz
                margin[res.right] += -res.final_tz

    group_winners: Dict[str, Tuple[str, int, float]] = {}
    for gname, members in groups.items():
        if not members: continue
        ranking = sorted(
            [(m.name, wins[m.name], margin[m.name]) for m in members],
            key=lambda t: (t[1], t[2]), reverse=True
        )
        group_winners[gname] = ranking[0]

    print("\n================= GROUP WINNERS =================")
    for g in sorted(group_winners.keys()):
        name, wcnt, msum = group_winners[g]
        print(f"Group {g}: {name}  (wins={wcnt}, margin={msum:.4f})")

    winners_csv = WORK_DIR / "group_winners.csv"
    with open(winners_csv, 'w', newline='') as wf:
        w = csv.writer(wf)
        w.writerow(["group", "winner", "wins", "aggregate_margin"])
        for g in sorted(group_winners.keys()):
            name, wcnt, msum = group_winners[g]
            w.writerow([g, name, wcnt, f"{msum:.6f}"])
    print(f"\nWrote: {RESULTS_CSV}")
    print(f"Wrote: {winners_csv}")


if __name__ == "__main__":
    run_group_stage()
