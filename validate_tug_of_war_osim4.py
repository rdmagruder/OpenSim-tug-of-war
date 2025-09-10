# validate_tug_of_war_osim4.py
import math
import opensim as osim
import pathlib

# --- EDIT THIS ONLY ---
ENTRY_DIR = pathlib.Path(r"C:\Users\MoBL2\PycharmProjects\OpenSim-tug-of-war\entries\WyattYoung")
DEFAULT_XML = pathlib.Path(r"C:\Users\MoBL2\PycharmProjects\OpenSim-tug-of-war\default_control.xml")
# ----------------------

# auto-detect files
osim_files = list(ENTRY_DIR.glob("*.osim"))
xml_files  = list(ENTRY_DIR.glob("*.xml"))

if len(osim_files) != 1:
    raise RuntimeError(f"Expected exactly 1 .osim in {ENTRY_DIR}, found {len(osim_files)}.")

MODEL_PATH = osim_files[0]

if len(xml_files) == 1:
    CONTROLS_XML = xml_files[0]
    print(f"Found controls file in entry dir: {CONTROLS_XML}")
else:
    # fallback
    CONTROLS_XML = DEFAULT_XML
    print(f"No controls XML found in {ENTRY_DIR}, using default: {CONTROLS_XML}")

print(f"Using model:   {MODEL_PATH}")
print(f"Using control: {CONTROLS_XML}")

# Now continue with your validation as before
model = osim.Model(str(MODEL_PATH))
state = model.initSystem()
control_set = osim.ControlSet(str(CONTROLS_XML))

# Limits (same as original scripts)
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

def check(cond, msg):
    return "" if cond else ("    " + msg + "\n")

def check_ineq(lo, val, hi, name, units):
    return "" if (lo <= val <= hi) else \
        f"For {name}, the inequality {lo} {units} <= {val} {units} <= {hi} {units} does not hold.\n"

def validate_muscle(base):
    msg = ""
    name = base.getName()
    m = osim.Millard2012EquilibriumMuscle.safeDownCast(base)
    if m is None:
        raise Exception(f"Muscle '{name}' is not a Millard2012EquilibriumMuscle.")

    area_cm2 = m.get_max_isometric_force() / specificTension
    volume_cm3 = area_cm2 * (m.get_optimal_fiber_length() * 100.0)
    msg += check(volume_cm3 <= volumeMax, f"Volume of {volume_cm3:.4g} cm^3 exceeds max {volumeMax} cm^3.")

    power_W = m.get_max_isometric_force() * m.get_max_contraction_velocity() * m.get_optimal_fiber_length()
    msg += check(power_W <= powerMax, f"Power of {power_W:.4g} W exceeds max {powerMax} W.")

    msg += check(m.get_tendon_slack_length() >= tendonSlackLengthMin,
                 f"Tendon slack length {m.get_tendon_slack_length():.4g} m below min {tendonSlackLengthMin} m.")

    optL = m.get_optimal_fiber_length()
    msg += check_ineq(optimalFiberLengthMin, optL, optimalFiberLengthMax, "optimal fiber length", "m")

    penn_deg = m.get_pennation_angle_at_optimal() * 180.0 / math.pi
    msg += check_ineq(pennationAngleAtOptimalMin, penn_deg, pennationAngleAtOptimalMax,
                      "pennation angle at optimal fiber length", "deg")

    vmax = m.get_max_contraction_velocity()
    msg += check_ineq(maxContractionVelocityMin, vmax, maxContractionVelocityMax, "max contraction velocity", "lM0/s")

    activ = m.get_activation_time_constant()
    deact = m.get_deactivation_time_constant()
    msg += check_ineq(activationTimeConstantMin, activ, activationTimeConstantMax, "activation time constant", "s")
    msg += check_ineq(deactivationTimeConstantMin, deact, deactivationTimeConstantMax, "deactivation time constant", "s")
    msg += check_ineq(deactMinusActivTimeConstMin, deact - activ, deactMinusActivTimeConstMax,
                      "[deact. - activ.] time constant", "s")

    init_len = m.getGeometryPath().getLength(state)
    expected = optL + m.get_tendon_slack_length()
    msg += check(abs(init_len - expected) < 1e-10,
                 f"Initial MTU length is {init_len:.6g} m, but should be {optL:.6g} m + "
                 f"{m.get_tendon_slack_length():.6g} m = {expected:.6g} m.")
    msg += check_ineq(muscleTendonLengthMin, init_len, muscleTendonLengthMax, "initial MTU length", "m")

    if msg:
        print(f"Muscle '{name}' failed validation:\n{msg}The anti-doping agency has put you on their watch list.\n")
    else:
        print(f"Muscle '{name}' passed validation!\n")

# Validate all muscles
for i in range(model.getMuscles().getSize()):
    validate_muscle(model.getMuscles().get(i))

# Validate controls (piecewise-linear integral, bounds, etc.)
def integrate(x, y):
    total = 0.0
    for i in range(len(x)-1):
        total += (x[i+1]-x[i]) * 0.5*(y[i]+y[i+1])
    return total

def clamp(u): return 0.0 if u < 0 else (1.0 if u > 1.0 else u)

def validate_control(base):
    msg = ""
    name = base.getName()
    ctrl = osim.ControlLinear.safeDownCast(base)
    if ctrl is None:
        raise Exception(f"Control {name} must be a ControlLinear.")
    nodes = ctrl.getControlValues()
    t = [nodes.get(k).getTime()  for k in range(nodes.getSize())]
    u = [clamp(nodes.get(k).getValue()) for k in range(nodes.getSize())]
    msg += check(min(t) >= 0 and max(t) <= 1, f"Excitation times for {name} must be between 0 and 1.")
    msg += check(abs(t[0]) < 1e-10, "First time point must be at 0s")
    msg += check(abs(t[-1] - 1.0) < 1e-10, "Last time point must be at 1s")
    msg += check(t == sorted(t), f"Excitation times for {name} must be in ascending order.")
    integ = integrate(t, u)
    msg += check(integ <= excitationIntegralMax, f"Excitation integral {integ:.6g} exceeds max {excitationIntegralMax}.")
    if msg:
        print(f"Excitation '{name}' failed validation:\n{msg}")
    else:
        print(f"Excitation '{name}' passed validation!\n")

for i in range(control_set.getSize()):
    validate_control(control_set.get(i))
