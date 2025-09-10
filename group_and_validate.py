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
import argparse
import array as pyarray
try:
    import numpy as np
    _HAS_NP = True
except Exception:
    _HAS_NP = False

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

# Paths
BASE_MODEL_PATH = Path(r"C:\Users\MoBL2\Documents\OpenSim\4.5\Models\Tug_of_War\Tug_of_War_Millard.osim")
ENTRIES_DIR     = Path("entries")
WORK_DIR        = Path("group_stage_outputs")
DEFAULT_XML     = Path(r"C:\Users\MoBL2\PycharmProjects\OpenSim-tug-of-war\default_control.xml")  # e.g., in your OpenSim-tug-of-war repo root

# Tournament
SIM_DURATION    = 1.0
NUM_GROUPS      = 8

# Outputs
RESULTS_CSV     = WORK_DIR / "group_stage_results.csv"
INVALID_CSV     = WORK_DIR / "invalid_entries.csv"
STRICT_VALIDATE = True  # if True, skip invalid entries entirely; if False, include with warning
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

def _pick_preferred_xml(xml_paths: List[Path]) -> Path | None:
    """
    If any XML contains a ControlLinear named 'LeftMuscle.excitation', prefer that.
    Otherwise return the first readable ControlSet. If none load, return None.
    """
    # First pass: look for explicit LeftMuscle.excitation
    for p in xml_paths:
        try:
            cs = osim.ControlSet(str(p))
            for i in range(cs.getSize()):
                c = osim.ControlLinear.safeDownCast(cs.get(i))
                if c and c.getName() == "LeftMuscle.excitation":
                    return p
        except Exception:
            continue
    # Second pass: first readable
    for p in xml_paths:
        try:
            _ = osim.ControlSet(str(p))
            return p
        except Exception:
            continue
    return None

def find_entries(entries_dir: Path) -> List[Entry]:
    """Discover student entries for the group stage.
    Strict mode: requires exactly one .osim AND at least one .xml in each student folder.
    If multiple XMLs exist, prefer the one containing 'LeftMuscle.excitation'.
    Students without any XML are DISQUALIFIED (skipped).
    """
    entries: List[Entry] = []
    dq_no_xml: List[str] = []

    for student_dir in sorted(p for p in entries_dir.iterdir() if p.is_dir()):
        osims = list(student_dir.glob("*.osim"))
        xmls  = list(student_dir.glob("*.xml"))

        # Require exactly one .osim
        if len(osims) != 1:
            print(f"[DQ] '{student_dir.name}': expected exactly 1 .osim; found {len(osims)}. Disqualified.")
            continue
        model_path = osims[0]

        # Require at least one .xml (NO fallback)
        if len(xmls) == 0:
            print(f"[DQ] '{student_dir.name}': no .xml control file submitted. Disqualified.")
            dq_no_xml.append(student_dir.name)
            continue
        elif len(xmls) == 1:
            controls_path = xmls[0]
        else:
            # Multiple XMLs: pick the one with LeftMuscle.excitation if possible
            chosen = _pick_preferred_xml(xmls)
            if chosen is None:
                print(f"[DQ] '{student_dir.name}': multiple XMLs but none loadable. Disqualified.")
                continue
            controls_path = chosen
            if controls_path.name not in [p.name for p in xmls]:
                print(f"[INFO] '{student_dir.name}': picked {controls_path.name} from multiple XMLs.")

        entries.append(Entry(name=student_dir.name, model_path=model_path, controls_path=controls_path))

    if dq_no_xml:
        print(f"[SUMMARY] Disqualified (missing XML): {', '.join(dq_no_xml)}")

    if not entries:
        raise RuntimeError(f"No usable entries found under {entries_dir.resolve()}.")

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

def _copy_student_muscle_params(src_model: osim.Model, dst_model: osim.Model,
                                dst_muscle_name: str, state: osim.State) -> None:
    Millard = osim.Millard2012EquilibriumMuscle

    # pick student's tuned muscle (you may already have a helper for this)
    src_muscle_base = src_model.getMuscles().get('LeftMuscle')
    src_m = Millard.safeDownCast(src_muscle_base)

    dst_muscle_base = dst_model.getMuscles().get(dst_muscle_name)
    dst_m = Millard.safeDownCast(dst_muscle_base)

    # --- copy ALL key properties, including default_activation ---
    dst_m.set_max_isometric_force(src_m.get_max_isometric_force())
    dst_m.set_optimal_fiber_length(src_m.get_optimal_fiber_length())
    dst_m.set_tendon_slack_length(src_m.get_tendon_slack_length())
    dst_m.set_pennation_angle_at_optimal(src_m.get_pennation_angle_at_optimal())
    dst_m.set_max_contraction_velocity(src_m.get_max_contraction_velocity())
    dst_m.set_activation_time_constant(src_m.get_activation_time_constant())
    dst_m.set_deactivation_time_constant(src_m.get_deactivation_time_constant())
    dst_m.set_default_activation(src_m.get_default_activation())   # <-- important

    # touch geometry so the path is realized; not strictly necessary
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
    """
    Build a ControlSet XML containing two ControlLinear controls:
      - LeftMuscle.excitation  (prefer 'LeftMuscle.excitation' in left file, else first)
      - RightMuscle.excitation (prefer first control in right file)
    We hand-serialize minimal XML compatible with OpenSim 4.x.
    """
    left_cs  = osim.ControlSet(str(left_ctrl_path))
    right_cs = osim.ControlSet(str(right_ctrl_path))

    cl = _extract_control(left_cs,  prefer_left=True)
    cr = _extract_control(right_cs, prefer_left=False)

    def _serialize_controllinear_minimal(name: str, ctrl: osim.ControlLinear) -> str:
        nodes = ctrl.getControlValues()
        node_xml_parts = []
        for i in range(nodes.getSize()):
            t = nodes.get(i).getTime()
            v = nodes.get(i).getValue()
            node_xml_parts.append(
                "                    <ControlLinearNode>\n"
                f"                        <t>{t}</t>\n"
                f"                        <value>{v}</value>\n"
                "                    </ControlLinearNode>\n"
            )
        nodes_block = "".join(node_xml_parts)

        # Minimal, safe subset of tags
        return (
            f'            <ControlLinear name="{name}">\n'
            f"                <is_model_control>true</is_model_control>\n"
            f"                <extrapolate>true</extrapolate>\n"
            f"                <x_nodes>\n{nodes_block}                </x_nodes>\n"
            f"                <min_nodes />\n"
            f"                <max_nodes />\n"
            f"            </ControlLinear>\n"
        )

    left_xml  = _serialize_controllinear_minimal("LeftMuscle.excitation",  cl)
    right_xml = _serialize_controllinear_minimal("RightMuscle.excitation", cr)

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OpenSimDocument Version="40500">\n'
        '    <ControlSet name="Control Set">\n'
        '        <objects>\n'
        f"{left_xml}"
        f"{right_xml}"
        '        </objects>\n'
        '        <groups />\n'
        '    </ControlSet>\n'
        '</OpenSimDocument>\n'
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)

def _build_function_for(times, vals, name: str):
    """
    Build an OpenSim Function from knot vectors in a way that works across Python bindings.
    Try:
      - PiecewiseLinearFunction (NumPy then C-arrays then ArrayDouble)
      - GCVSpline (degree=1 then degree=3; NumPy then C-arrays)
      - SimmSpline (as a last resort)
    """
    # Defensive: ensure plain floats and strictly increasing times (you already dedupe/sort earlier)
    times = [float(t) for t in times]
    vals  = [float(v) for v in vals]

    # --- PiecewiseLinearFunction paths ---
    if _HAS_NP:
        t_np = np.ascontiguousarray(times, dtype=float)
        v_np = np.ascontiguousarray(vals,  dtype=float)
        for ctor in (
            lambda: osim.PiecewiseLinearFunction(len(times), t_np, v_np, name),
            lambda: osim.PiecewiseLinearFunction(len(times), t_np, v_np),
        ):
            try:
                return ctor()
            except TypeError:
                pass

    t_arr = pyarray.array('d', times)
    v_arr = pyarray.array('d', vals)
    for ctor in (
        lambda: osim.PiecewiseLinearFunction(len(times), t_arr, v_arr, name),
        lambda: osim.PiecewiseLinearFunction(len(times), t_arr, v_arr),
    ):
        try:
            return ctor()
        except TypeError:
            pass

    try:
        t_os = osim.ArrayDouble(); v_os = osim.ArrayDouble()
        for t in times: t_os.append(t)
        for v in vals:  v_os.append(v)
        for ctor in (
            lambda: osim.PiecewiseLinearFunction(len(times), t_os, v_os, name),
            lambda: osim.PiecewiseLinearFunction(len(times), t_os, v_os),
        ):
            try:
                return ctor()
            except TypeError:
                pass
    except Exception:
        pass

    # --- GCVSpline paths (robust across builds) ---
    # degree=1 ≈ piecewise-linear; degree=3 = cubic
    if _HAS_NP:
        t_np = np.ascontiguousarray(times, dtype=float)
        v_np = np.ascontiguousarray(vals,  dtype=float)
        for deg in (1, 3):
            for ctor in (
                lambda: osim.GCVSpline(deg, len(times), t_np, v_np),
            ):
                try:
                    return ctor()
                except TypeError:
                    pass

    for deg in (1, 3):
        for ctor in (
            lambda: osim.GCVSpline(deg, len(times), t_arr, v_arr),
        ):
            try:
                return ctor()
            except TypeError:
                pass

    # --- Last resort: SimmSpline ---
    for ctor in (
        lambda: osim.SimmSpline(len(times), t_arr, v_arr),
        lambda: osim.SimmSpline(len(times), t_os,  v_os) if 't_os' in locals() else None,
    ):
        try:
            f = ctor() if ctor else None
            if f is not None:
                return f
        except Exception:
            pass

    raise RuntimeError(
        f"Could not construct a Function. n={len(times)}, times[:3]={times[:3]}, vals[:3]={vals[:3]}"
    )



def _run_forward(model_path: Path, controls_xml: Path, out_dir: Path,
                 t0: float = 0.0, tf: float = SIM_DURATION) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Absolute, POSIX-style paths (OpenSim parses these most reliably)
    model_abs    = model_path.resolve()
    controls_abs = controls_xml.resolve()
    results_abs  = out_dir.resolve()
    model_posix    = model_abs.as_posix()
    controls_posix = controls_abs.as_posix()
    results_posix  = results_abs.as_posix()

    # Double-check files exist before we hand them to the tool
    if not model_abs.exists():
        raise FileNotFoundError(f"Model file missing: {model_abs}")
    if not controls_abs.exists():
        raise FileNotFoundError(f"Controls XML missing: {controls_abs}")

    # Load the model so we can also pass the object to the tool after parsing
    model = osim.Model(str(model_abs))
    _ = model.initSystem()

    # Write a minimal ForwardTool setup that includes a ControlSetController
    setup_path = out_dir / "setup_forward.xml"
    setup_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <ForwardTool name="{model.getName()}">
    <model_file>{model_posix}</model_file>
    <replace_force_set>false</replace_force_set>
    <force_set_files />
    <results_directory>{results_posix}</results_directory>
    <output_precision>8</output_precision>
    <initial_time>{t0}</initial_time>
    <final_time>{tf}</final_time>
    <solve_for_equilibrium_for_auxiliary_states>true</solve_for_equilibrium_for_auxiliary_states>
    <maximum_number_of_integrator_steps>20000</maximum_number_of_integrator_steps>
    <maximum_integrator_step_size>1</maximum_integrator_step_size>
    <minimum_integrator_step_size>1e-008</minimum_integrator_step_size>
    <integrator_error_tolerance>1e-005</integrator_error_tolerance>
    <AnalysisSet name="Analyses">
      <objects />
      <groups />
    </AnalysisSet>
    <ControllerSet name="Controllers">
      <objects>
        <ControlSetController>
          <controls_file>{controls_posix}</controls_file>
        </ControlSetController>
      </objects>
      <groups />
    </ControllerSet>
    <external_loads_file />
    <use_specified_dt>false</use_specified_dt>
  </ForwardTool>
</OpenSimDocument>
"""
    setup_path.write_text(setup_xml, encoding="utf-8")

    # Construct the tool FROM XML (so it parses the controller),
    # then set the already-loaded model object
    tool = osim.ForwardTool(str(setup_path))
    tool.setModel(model)
    tool.run()

    # Read final block_tz
    states_files = list(out_dir.glob("*states.sto"))
    if not states_files:
        raise RuntimeError(f"No states.sto found in {out_dir}")
    states_path = sorted(states_files)[-1]

    table = osim.TimeSeriesTable(str(states_path))
    labels = list(table.getColumnLabels())
    tz_cols = [i for i, s in enumerate(labels) if "block_tz" in s]
    if not tz_cols:
        raise RuntimeError("Could not locate 'block_tz' in states table labels.")
    last_idx = table.getNumRows() - 1
    row = table.getRowAtIndex(last_idx)
    return float(row[tz_cols[0]])



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
    model.printToXML(str(combined_model_path))

    merged_controls_path = match_dir / "Combined_controls.xml"
    _merge_controls(left.controls_path, right.controls_path, merged_controls_path)

    final_tz = _run_forward(combined_model_path, merged_controls_path, match_dir)
    winner = left.name if final_tz > 0 else (right.name if final_tz < 0 else None)

    return MatchResult(left=left.name, right=right.name, final_tz=final_tz, winner=winner, out_dir=match_dir)


def run_group_stage(open_division: bool = False):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    entries = find_entries(ENTRIES_DIR)
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASE_MODEL_PATH}")

    if open_division:
        print("[INFO] Open division: skipping all validation (muscle + controls).")
        valid_entries = entries
        invalid_rows = []
    else:
        # ---- existing validation block stays the same ----
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

        # Save invalid report (unchanged)
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
    parser = argparse.ArgumentParser(description="Tug-of-War group stage")
    parser.add_argument(
        "--open", action="store_true",
        help="Open division: skip all validation (muscle + activation checks)"
    )
    parser.add_argument(
        "--groups", type=int, default=NUM_GROUPS,
        help=f"Number of groups (default {NUM_GROUPS})"
    )
    args = parser.parse_args()

    # Override NUM_GROUPS if the user specified --groups
    NUM_GROUPS = args.groups

    run_group_stage(open_division=args.open)


