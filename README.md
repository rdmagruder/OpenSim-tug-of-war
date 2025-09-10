# OpenSim Tug-of-War Competition Framework

A comprehensive framework for running OpenSim-based tug-of-war muscle design competitions in educational settings. This tool is designed for professors and teaching assistants to efficiently manage and run competitions among students who design and optimize muscle models.

## Overview

This framework automates the process of running biomechanics competitions where students:
1. Design and tune muscle models (.osim files) 
2. Create control strategies (.xml files) for muscle activation patterns
3. Compete in simulated tug-of-war matches

The system handles validation, group-stage tournaments, and single-elimination brackets automatically, making it easy to run competitions with many participants.

### What the Competition Simulates

Students compete by designing optimal muscle parameters for a tug-of-war simulation where:
- Two muscle models (left vs right) pull on opposite ends of a block
- Each muscle follows the student's designed activation pattern over 1 second
- The winner is determined by the final position of the block (positive = left wins, negative = right wins)
- Students must optimize within realistic physiological constraints (muscle volume, power, activation patterns, etc.)

## Prerequisites

- **OpenSim 4.x** (4.4 or higher recommended)
- **Python 3.7+** with OpenSim Python bindings
- **Required Python packages**: `opensim`, `numpy` (optional but recommended)

### Installing OpenSim Python Bindings

Follow the [OpenSim Python installation guide](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python) or install via conda:

```bash
conda install -c opensim-org opensim
```

## Setup

### 1. Directory Structure

Create the following directory structure in your project folder:

```
your-project/
├── entries/                    # Student submission folders
│   ├── student1/
│   │   ├── model.osim         # Student's muscle model
│   │   └── controls.xml       # Student's control strategy
│   ├── student2/
│   │   ├── model.osim
│   │   └── controls.xml
│   └── ...
├── group_stage_outputs/        # Generated automatically
├── default_control.xml         # Default control file (provided)
├── group_and_validate.py       # Main competition scripts
├── tournament.py
└── validate_tug_of_war_osim4.py
```

### 2. Configure Base Model Path

Edit the configuration section in each Python script to point to your OpenSim base model:

**In `group_and_validate.py` (line 55):**
```python
BASE_MODEL_PATH = Path(r"C:\path\to\your\OpenSim\Models\Tug_of_War\Tug_of_War_Millard.osim")
```

**In `tournament.py` (line 21):**
```python
BASE_MODEL_PATH = Path(r"C:\path\to\your\OpenSim\Models\Tug_of_War\Tug_of_War_Millard.osim")
```

The base model file is typically found in your OpenSim installation under:
- Windows: `C:\Users\[username]\Documents\OpenSim\4.x\Models\Tug_of_War\Tug_of_War_Millard.osim`
- Mac/Linux: `~/Documents/OpenSim/4.x/Models/Tug_of_War/Tug_of_War_Millard.osim`

**Note**: If you don't have this model, it may be available through the OpenSim GUI under Help → Download Models, or from the [OpenSim documentation](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Designing+a+Muscle+for+a+Tug-of-War+Competition).

### 3. Student Submission Requirements

Each student must submit **exactly one folder** in the `entries/` directory containing:

- **One .osim file**: Their optimized muscle model
- **One .xml file**: Their muscle control strategy (**required** for group competitions)

**Example student folder:**
```
entries/JohnDoe/
├── optimized_muscle.osim
└── my_control_strategy.xml
```

**Important notes:**
- **File naming**: The actual filenames don't matter, but there must be exactly one of each type per student
- **Control files**: The group competition script (`group_and_validate.py`) **requires** students to submit XML control files - students without XML files will be disqualified
- **Individual validation**: The `validate_tug_of_war_osim4.py` script will fall back to `default_control.xml` if no student XML is provided

### What Makes a Competitive Submission

Students should optimize their muscle models within the competition constraints:

**Muscle Model (.osim file)**:
- Must modify the `LeftMuscle` parameters in the base model
- Key parameters: max isometric force, optimal fiber length, tendon slack length, contraction velocity
- Must stay within physiological limits (volume ≤ 100 cm³, power ≤ 175 W, etc.)

**Control Strategy (.xml file)**:
- Defines muscle activation pattern over the 1-second simulation
- Must control `LeftMuscle.excitation` with values between 0 and 1
- Activation integral must be ≤ 0.5 (prevents constant maximum activation)
- Timing and intensity of activation greatly affects performance

## Running Competitions

### Step 1: Validate Individual Entries (Optional)

Before running the full competition, you can validate individual student submissions:

```bash
# Edit the ENTRY_DIR path in validate_tug_of_war_osim4.py first
python validate_tug_of_war_osim4.py
```

This checks if a student's muscle model and controls meet competition rules.

### Step 2: Run Group Stage

The group stage runs round-robin tournaments within groups and identifies winners:

```bash
# Standard competition with validation
python group_and_validate.py

# Open division (skip validation)
python group_and_validate.py --open

# Custom number of groups
python group_and_validate.py --groups 4
```

**Output files:**
- `group_stage_outputs/group_stage_results.csv` - All match results
- `group_stage_outputs/group_winners.csv` - Group winners for tournament
- `group_stage_outputs/invalid_entries.csv` - Validation failures (if any)

### Step 3: Run Tournament (Optional)

After the group stage, run a single-elimination tournament among group winners:

```bash
python tournament.py
```

**Output files:**
- `group_stage_outputs/tournament/tournament_results.csv` - Tournament bracket results

## Configuration Options

### Group Stage Settings

Edit `group_and_validate.py` to customize:

```python
# Number of groups (line 62)
NUM_GROUPS = 1  # Default is 1 group (round-robin with all students)

# Simulation duration (line 61)  
SIM_DURATION = 1.0

# Validation mode (line 67)
STRICT_VALIDATE = True  # False to include invalid entries with warnings
```

### Validation Rules

Competition rules are defined in `group_and_validate.py` (lines 40-52):

```python
specificTension = 35        # N/cm²
volumeMax = 100            # cm³  
powerMax = 175             # W
muscleTendonLengthMin, muscleTendonLengthMax = 0.15, 0.45
# ... other constraints
```

### Tournament Settings

In `tournament.py`, you can modify:

```python
SIM_DURATION = 1.0  # Simulation time per match
```

## Understanding Results

### Group Stage Results

**`group_stage_results.csv`** contains all match results:
- `group`: Which group the match was in
- `left`/`right`: Student names
- `final_block_tz`: Final position of the block (positive = left wins, negative = right wins)
- `winner`: Winner name or "TIE"
- `match_dir`: Folder with detailed simulation results

**`group_winners.csv`** shows the winner from each group:
- `winner`: Student name
- `wins`: Number of matches won
- `aggregate_margin`: Total margin across all matches (tiebreaker)

### Tournament Results

**`tournament_results.csv`** shows the bracket progression:
- `round_name`: Tournament round (Quarterfinals, Semifinals, etc.)
- `left`/`right`: Competitors
- `final_block_tz`: Block position
- `winner`: Match winner
- `match_dir`: Detailed results folder

## Troubleshooting

### Common Issues

**"Model file not found"**
- Check that `BASE_MODEL_PATH` points to the correct Tug_of_War_Millard.osim file
- Ensure OpenSim is properly installed

**"No valid entries found"**
- Check that `entries/` folder exists and contains student subfolders
- Each student folder must have exactly one .osim file
- Check file permissions

**"Import opensim failed"**
- Verify OpenSim Python bindings are installed: `python -c "import opensim"`
- Try reinstalling: `conda install -c opensim-org opensim`

**Student disqualified for missing XML**
- This is intentional in group competitions - students must provide control files
- Use the individual `validate_tug_of_war_osim4.py` script if you want to test with default controls
- To modify this behavior, edit the `find_entries()` function in `group_and_validate.py`

**"Could not locate 'block_tz' in states table"**
- The base model may be incorrect or corrupted
- Ensure you're using the official Tug_of_War_Millard.osim model

### Getting Detailed Error Information

For debugging specific simulation failures, check the individual match directories in:
- `group_stage_outputs/Group_[A-Z]/[student1]_vs_[student2]/`
- `group_stage_outputs/tournament/Round_[X]_[name]/[student1]_vs_[student2]/`

Each contains:
- `Combined_Tug_of_War.osim` - The combined model used
- `Combined_controls.xml` - The merged control file
- `setup_forward.xml` - OpenSim forward simulation setup
- `*_states.sto` - Simulation results

## Competition Workflow Summary

For a typical competition with 20+ students:

1. **Setup** (one time):
   - Install OpenSim and Python bindings
   - Configure paths in the Python scripts
   - Create `entries/` directory

2. **Collect submissions**:
   - Have students submit folders with their .osim and .xml files
   - Place each in `entries/student_name/`

3. **Run competition**:
   ```bash
   # Run group stage with multiple groups (e.g., 4 groups for 20 students)
   python group_and_validate.py --groups 4
   
   # Or single group for round-robin with all students
   python group_and_validate.py --groups 1
   
   # Optional: Run tournament among group winners (only if multiple groups)
   python tournament.py
   ```

4. **Review results**:
   - Check `group_winners.csv` for group champions
   - Check `tournament_results.csv` for overall champion (if tournament was run)
   - Review `invalid_entries.csv` for any disqualified students

This framework eliminates the need to run matches manually one-by-one, making large competitions feasible and efficient.

## Advanced Usage

### Custom Group Assignment

To manually assign students to specific groups, modify the `group_students()` function in `group_and_validate.py`.

### Custom Validation Rules

Modify the validation functions in `group_and_validate.py` to implement custom competition rules or constraints.

### Batch Processing Multiple Competitions

You can run multiple competitions by changing the `ENTRIES_DIR` and `WORK_DIR` paths between runs.

## Quick Reference

### Essential Commands
```bash
# Validate a single student submission
python validate_tug_of_war_osim4.py

# Run group stage with all students in one group (round-robin)
python group_and_validate.py --groups 1

# Run group stage with multiple groups 
python group_and_validate.py --groups 4

# Skip validation (open division)
python group_and_validate.py --open

# Run tournament among group winners
python tournament.py
```

### Key Files to Check After Running
- `group_stage_outputs/group_winners.csv` - Winners from each group
- `group_stage_outputs/invalid_entries.csv` - Students who were disqualified
- `group_stage_outputs/tournament/tournament_results.csv` - Tournament bracket

---

**Need help?** Check the OpenSim documentation at [simtk.org](https://simtk.org/home/opensim) or refer to the original competition documentation.