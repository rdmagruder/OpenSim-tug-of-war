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
import math

import opensim as osim

# ------------------------- CONFIG -------------------------
# Validation thresholds (same as your validator)
specificTension = 35  # N/cm^2
volumeMax = 100       # cm^3
powerMax = 175        # W
muscleTendonLengthMin, muscleTendonLengthMax = 0.15, 0.45
tendonSlackLengthMin  = 0.10
optimalFiberLengthMin, optimalFiberLengthMax = 0.05, 0.20
pennationAngleAtOptimalMin, pennationAngleAtOptimalMax = 0, 30  # deg
maxContractionVelocityMin, maxContractionVelocityMax = 2, 10    # lM0/s
excitationIntegralMax = 0.5
activationTimeConstantMin, activationTimeConstantMax = 0.010, 0.020
deactivationTimeConstantMin, deactivationTimeConstantMax = 0.040, 0.060
deactMinusActivTimeConstMin, deactMinusActivTimeConstMax = 0.030, 0.040

BASE_MODEL_PATH = Path(r"C:/Users/<you>/Documents/OpenSim/4.4/Resources/Models/Tug_of_War/Tug_of_War_Millard.osim")
ENTRIES_DIR     = Path("entries")
WORK_DIR        = Path("group_stage_outputs")
SIM_DURATION    = 1.0
NUM_GROUPS      = 8
RESULTS_CSV     = WORK_DIR / "group_stage_results.csv"
INVALID_CSV     = WORK_DIR / "invalid_entries.csv"
STRICT_VALIDATE = True  # if True, skip invalid entries entirely; if False, include with warning
# ----------------------------------------------------------
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
    """Discover student entries. This raw discovery does not filter by validity.
    Validation happens in run_group_stage() so we can emit a single invalid_entries.csv file.
    """
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


# ---------- Validation helpers ----------

def _check(cond: bool, msg: str) -> str:
    return "" if cond else ("    " + msg + "")

def _check_ineq(lo, val, hi, name, units) -> str:
    return "" if (lo <= val <= hi) else \
        f"For {name}, the inequality {lo} {units} <= {val} {units} <= {hi} {units} does not hold."

def _validate_student_model(model_path: Path) -> str:
    model = osim.Model(str(model_path))
    state = model.initSystem()

    # Expect a Millard muscle named LeftMuscle (what students tune)
    base = model.getMuscles().get('LeftMuscle')
    m = osim.Millard2012EquilibriumMuscle.safeDownCast(base)
    if m is None:
        return "LeftMuscle is not a Millard2012EquilibriumMuscle or missing."

    msg = ""
    area_cm2 = m.get_max_isometric_force() / specificTension
    volume_cm3 = area_cm2 * (m.get_optimal_fiber_length() * 100.0)
    msg += _check(volume_cm3 <= volumeMax, f"Volume {volume_cm3:.4g} cm^3 exceeds {volumeMax} cm^3.")

    power_W = m.get_max_isometric_force() * m.get_max_contraction_velocity() * m.get_optimal_fiber_length()
    msg += _check(power_W <= powerMax, f"Power {power_W:.4g} W exceeds {powerMax} W.")

    msg += _check(m.get_tendon_slack_length() >= tendonSlackLengthMin,
                  f"Tendon slack length {m.get_tendon_slack_length():.4g} m < {tendonSlackLengthMin} m.")

    optL = m.get_optimal_fiber_length()
    msg += _check_ineq(optimalFiberLengthMin, optL, optimalFiberLengthMax, "optimal fiber length", "m")

    penn_deg = m.get_pennation_angle_at_optimal() * 180.0 / math.pi
    msg += _check_ineq(pennationAngleAtOptimalMin, penn_deg, pennationAngleAtOptimalMax,
                       "pennation angle at optimal fiber length", "deg")

    vmax = m.get_max_contraction_velocity()
    msg += _check_ineq(maxContractionVelocityMin, vmax, maxContractionVelocityMax, "max contraction velocity", "lM0/s")

    activ = m.get_activation_time_constant()
    deact = m.get_deactivation_time_constant()
    msg += _check_ineq(activationTimeConstantMin, activ, activationTimeConstantMax, "activation time constant", "s")
    msg += _check_ineq(deactivationTimeConstantMin, deact, deactivationTimeConstantMax, "deactivation time constant", "s")
    msg += _check_ineq(deactMinusActivTimeConstMin, deact - activ, deactMinusActivTimeConstMax,
                       "[deact-activ] time constant", "s")

    init_len = m.getGeometryPath().getLength(state)
    expected = optL + m.get_tendon_slack_length()
    msg += _check(abs(init_len - expected) < 1e-10,
                  f"Initial MTU length {init_len:.6g} m, expected {optL:.6g} + {m.get_tendon_slack_length():.6g} = {expected:.6g} m.")
    msg += _check_ineq(muscleTendonLengthMin, init_len, muscleTendonLengthMax, "initial MTU length", "m")
    return msg


def _integrate_piecewise_linear(x: List[float], y: List[float]) -> float:
    total = 0.0
    for i in range(len(x)-1):
        total += (x[i+1]-x[i]) * 0.5*(y[i]+y[i+1])
    return total


def _clamp01(u: float) -> float:
    return 0.0 if u < 0 else (1.0 if u > 1.0 else u)


def _validate_controlset_for_left(cs: osim.ControlSet) -> str:
    msg = ""
    # Prefer a control named LeftMuscle.excitation if present
    candidates = [osim.ControlLinear.safeDownCast(cs.get(i)) for i in range(cs.getSize())]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return "No ControlLinear entries in control file."

    ctrl = None
    for c in candidates:
        if c.getName() == "LeftMuscle.excitation":
            ctrl = c; break
    if ctrl is None:
        # fallback to first
        ctrl = candidates[0]

    nodes = ctrl.getControlValues()
    t = [nodes.get(k).getTime()  for k in range(nodes.getSize())]
    u = [ _clamp01(nodes.get(k).getValue()) for k in range(nodes.getSize())]

    if not t:
        return "Left control has no nodes."

    msg += _check(min(t) >= 0 and max(t) <= 1, "Excitation times must be within [0,1] s for Left.")
    msg += _check(abs(t[0]) < 1e-10, "Left control first time must be 0s.")
    msg += _check(abs(t[-1]-1.0) < 1e-10, "Left control last time must be 1s.")
    msg += _check(t == sorted(t), "Left control times must be ascending.")

    integ = _integrate_piecewise_linear(t, u)
    msg += _check(integ <= excitationIntegralMax, f"Left control integral {integ:.6g} exceeds {excitationIntegralMax}.")

    return msg


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

    # Validate entries first; collect reasons
    invalid_rows = []
    valid_entries: List[Entry] = []
    for e in entries:
        err = _validate_student_model(e.model_path)
        try:
            cs = osim.ControlSet(str(e.controls_path))
            err += _validate_controlset_for_left(cs)
        except Exception as ex:
            err += f"Failed to read controls: {ex}"
        if err:
            print(f"[INVALID] {e.name}:{err}")
            invalid_rows.append([e.name, str(e.model_path), str(e.controls_path), err.strip()])
            if not STRICT_VALIDATE:
                print(f"[WARN] Including {e.name} despite validation issues (STRICT_VALIDATE=False).")
                valid_entries.append(e)
        else:
            valid_entries.append(e)

    # Save invalid report
    with open(INVALID_CSV, 'w', newline='') as inv:
        w = csv.writer(inv)
        w.writerow(["student", "model", "controls", "issues"])
        for row in invalid_rows:
            w.writerow(row)
    if invalid_rows:
        print(f"Saved invalid entry report: {INVALID_CSV}")

    if not valid_entries:
        raise RuntimeError("All entries were invalid; nothing to simulate.")

    groups = group_students(valid_entries, NUM_GROUPS)

    wins: Dict[str, int] = {}
    margin: Dict[str, float] = {}

    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["group", "left", "right", "final_block_tz", "winner", "match_dir"])

        for gname, members in groups.items():
            print(f"=== Group {gname} ({len(members)} students) ===")
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

    print("================= GROUP WINNERS =================")
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
    print(f"Wrote: {RESULTS_CSV}")
    print(f"Wrote: {winners_csv}")


if __name__ == "__main__":
    run_group_stage()
