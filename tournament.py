"""
Tournament runner for the OpenSim Tug-of-War competition (Python, OpenSim 4.x)

Reads group winners from group_stage_outputs/group_winners.csv, seeds competitors by
aggregate_margin, assigns byes if needed, and runs a single-elimination bracket
until a champion is decided.

This file intentionally does NOT validate entries. It just uses the winners.
"""
from __future__ import annotations
import csv
import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import opensim as osim

# ------------------------- CONFIG -------------------------
BASE_MODEL_PATH = Path(r"C:\Users\MoBL2\Documents\OpenSim\4.5\Models\Tug_of_War\Tug_of_War_Millard.osim")
ENTRIES_DIR     = Path("entries")
WORK_DIR        = Path("group_stage_outputs")
DEFAULT_XML     = Path(r"C:\Users\MoBL2\PycharmProjects\OpenSim-tug-of-war\default_control.xml")

SIM_DURATION    = 1.0  # seconds

WINNERS_CSV     = WORK_DIR / "group_winners.csv"
TOURNEY_DIR     = WORK_DIR / "tournament"
BRACKET_CSV     = TOURNEY_DIR / "tournament_results.csv"
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
    winner: Optional[str]
    out_dir: Path

# ---------- Utility: find the best control XML in a folder ----------
def _pick_preferred_xml(xml_paths: List[Path]) -> Optional[Path]:
    """Prefer a ControlSet that contains 'LeftMuscle.excitation'; otherwise first loadable."""
    # pass 1: must contain LeftMuscle.excitation
    for p in xml_paths:
        try:
            cs = osim.ControlSet(str(p))
            for i in range(cs.getSize()):
                c = osim.ControlLinear.safeDownCast(cs.get(i))
                if c and c.getName() == "LeftMuscle.excitation":
                    return p
        except Exception:
            continue
    # pass 2: first loadable control set
    for p in xml_paths:
        try:
            _ = osim.ControlSet(str(p))
            return p
        except Exception:
            continue
    return None

# ---------- Load an Entry by winner name (folder name under entries/) ----------
def get_entry_for_name(name: str) -> Entry:
    student_dir = ENTRIES_DIR / name
    if not student_dir.exists() or not student_dir.is_dir():
        raise FileNotFoundError(f"Winner folder not found: {student_dir}")

    osims = list(student_dir.glob("*.osim"))
    xmls  = list(student_dir.glob("*.xml"))

    if len(osims) != 1:
        raise RuntimeError(f"{name}: expected exactly 1 .osim; found {len(osims)}")

    model_path = osims[0]

    if len(xmls) == 0:
        if DEFAULT_XML.exists():
            controls_path = DEFAULT_XML
            print(f"[INFO] {name}: no XML found; using default {DEFAULT_XML}")
        else:
            raise RuntimeError(f"{name}: no XML and default {DEFAULT_XML} missing.")
    elif len(xmls) == 1:
        controls_path = xmls[0]
    else:
        chosen = _pick_preferred_xml(xmls)
        if chosen is None:
            raise RuntimeError(f"{name}: multiple XMLs but none usable.")
        controls_path = chosen

    return Entry(name=name, model_path=model_path, controls_path=controls_path)

# ---------- Controls merge: write minimal ControlSet XML ----------
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

    cl = _extract_control(left_cs,  prefer_left=True)
    cr = _extract_control(right_cs, prefer_left=False)

    def _serialize_controllinear_minimal(name: str, ctrl: osim.ControlLinear) -> str:
        nodes = ctrl.getControlValues()
        bits = []
        for i in range(nodes.getSize()):
            t = nodes.get(i).getTime()
            v = nodes.get(i).getValue()
            bits.append(
                "                    <ControlLinearNode>\n"
                f"                        <t>{t}</t>\n"
                f"                        <value>{v}</value>\n"
                "                    </ControlLinearNode>\n"
            )
        nodes_block = "".join(bits)
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
        f"{left_xml}{right_xml}"
        '        </objects>\n'
        '        <groups />\n'
        '    </ControlSet>\n'
        '</OpenSimDocument>\n'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")

# ---------- Sim runner ----------
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

# ---------- Build combined model + play one match ----------
def _copy_student_muscle_params(src_model: osim.Model, dst_model: osim.Model,
                                dst_muscle_name: str, state: osim.State) -> None:
    Millard = osim.Millard2012EquilibriumMuscle
    src_muscle_base = src_model.getMuscles().get('LeftMuscle')
    src_m = Millard.safeDownCast(src_muscle_base)

    dst_muscle_base = dst_model.getMuscles().get(dst_muscle_name)
    dst_m = Millard.safeDownCast(dst_muscle_base)

    dst_m.set_max_isometric_force(src_m.get_max_isometric_force())
    dst_m.set_optimal_fiber_length(src_m.get_optimal_fiber_length())
    dst_m.set_tendon_slack_length(src_m.get_tendon_slack_length())
    dst_m.set_pennation_angle_at_optimal(src_m.get_pennation_angle_at_optimal())
    dst_m.set_max_contraction_velocity(src_m.get_max_contraction_velocity())
    dst_m.set_activation_time_constant(src_m.get_activation_time_constant())
    dst_m.set_deactivation_time_constant(src_m.get_deactivation_time_constant())
    _ = dst_m.getGeometryPath().getLength(state)

def play_match(left: Entry, right: Entry, base_model: Path, arena_dir: Path) -> MatchResult:
    match_dir = arena_dir / f"{left.name}_vs_{right.name}"
    match_dir.mkdir(parents=True, exist_ok=True)

    model = osim.Model(str(base_model))
    state = model.initSystem()

    left_model  = osim.Model(str(left.model_path));  left_model.initSystem()
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

# ---------- Seeding / bracket helpers ----------
def _read_winners(csv_path: Path) -> List[Tuple[str, float]]:
    """Returns list of (name, aggregate_margin)"""
    rows: List[Tuple[str, float]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["winner"].strip()
            margin = float(row["aggregate_margin"])
            rows.append((name, margin))
    return rows

def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def _seed_and_byes(winners: List[Tuple[str, float]]) -> Tuple[List[Tuple[str,float]], int]:
    """
    Sort winners by aggregate_margin (desc) → seeds.
    Returns (seed_list, num_byes).
    Top 'num_byes' seeds receive byes into Round 2.
    """
    seeds = sorted(winners, key=lambda t: t[1], reverse=True)
    bracket_size = _next_power_of_two(len(seeds))
    num_byes = bracket_size - len(seeds)
    return seeds, num_byes

def _round_name(size_after_advancers: int) -> str:
    names = {
        1: "Final",
        2: "Final",
        4: "Semifinals",
        8: "Quarterfinals",
        16: "Round of 16",
        32: "Round of 32",
    }
    # pick closest standard name; default to generic
    return names.get(size_after_advancers, f"Round of {size_after_advancers}")

# ---------- Tournament runner ----------
def run_tournament():
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASE_MODEL_PATH}")
    TOURNEY_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Read winners & seed
    winners = _read_winners(WINNERS_CSV)
    if not winners:
        raise RuntimeError(f"No winners found in {WINNERS_CSV}")
    seeds, num_byes = _seed_and_byes(winners)

    # 2) Build Entry objects for all winners (by folder name)
    entries_by_name: Dict[str, Entry] = {}
    for name, _ in seeds:
        entries_by_name[name] = get_entry_for_name(name)

    # 3) First round: give byes to top seeds; pair the rest in order
    advancers: List[str] = []
    log_rows = []
    round_idx = 1

    # Byes
    bye_names = [name for name, _ in seeds[:num_byes]]
    advancers.extend(bye_names)
    if num_byes > 0:
        print(f"[INFO] Assigning {num_byes} bye(s) to top seeds: {', '.join(bye_names)}")

    # Non-bye seeds to play in Round 1
    to_play = [name for name, _ in seeds[num_byes:]]
    round_size = len(advancers) + len(to_play)  # size before matches
    round_dir = TOURNEY_DIR / f"Round_{round_idx}_{_round_name(round_size)}"
    round_dir.mkdir(exist_ok=True)

    # Pair adjacent seeds among the non-bye list: (s1 vs s2), (s3 vs s4), ...
    it = iter(to_play)
    for left_name, right_name in itertools.zip_longest(it, it):
        if right_name is None:
            # odd count among non-bye? give an extra bye to this last one
            advancers.append(left_name)
            print(f"[INFO] Extra bye (unpaired): {left_name}")
            continue

        left = entries_by_name[left_name]
        right = entries_by_name[right_name]
        res = play_match(left, right, BASE_MODEL_PATH, round_dir)
        print(f"[R{round_idx}] {res.left} vs {res.right} → tz={res.final_tz:.4f} → winner={res.winner}")
        advancers.append(res.winner if res.winner else res.left)  # break tie in favor of left if needed
        log_rows.append([round_idx, _round_name(round_size), res.left, res.right,
                         f"{res.final_tz:.6f}", res.winner or "TIE", str(res.out_dir)])

    # 4) Subsequent rounds until champion
    while len(advancers) > 1:
        round_idx += 1
        names_this_round = advancers
        advancers = []

        round_dir = TOURNEY_DIR / f"Round_{round_idx}_{_round_name(len(names_this_round))}"
        round_dir.mkdir(exist_ok=True)

        it = iter(names_this_round)
        for left_name, right_name in itertools.zip_longest(it, it):
            if right_name is None:
                advancers.append(left_name)  # bye
                print(f"[INFO] Bye in R{round_idx}: {left_name}")
                continue

            left = entries_by_name[left_name]
            right = entries_by_name[right_name]
            res = play_match(left, right, BASE_MODEL_PATH, round_dir)
            print(f"[R{round_idx}] {res.left} vs {res.right} → tz={res.final_tz:.4f} → winner={res.winner}")
            advancers.append(res.winner if res.winner else res.left)
            log_rows.append([round_idx, _round_name(len(names_this_round)), res.left, res.right,
                             f"{res.final_tz:.6f}", res.winner or "TIE", str(res.out_dir)])

    champion = advancers[0]
    print(f"\n🏆 TOURNAMENT CHAMPION: {champion}")

    # 5) Write bracket log
    with open(BRACKET_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round_index", "round_name", "left", "right", "final_block_tz", "winner", "match_dir"])
        for row in log_rows:
            w.writerow(row)
        w.writerow(["", "Champion", champion, "", "", champion, ""])
    print(f"Wrote bracket results: {BRACKET_CSV}")

if __name__ == "__main__":
    run_tournament()
