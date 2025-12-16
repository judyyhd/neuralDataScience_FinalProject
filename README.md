# Cross-Region Information Flow Analysis
## Allen Brain Observatory Neuropixels Data

This project analyzes how visual information flows through different brain regions during visual stimuli presentation, using simultaneous multi-region recordings from the Allen Brain Observatory.

## Scientific Motivation

### The Core Question
When a visual stimulus appears, how does neural activity cascade through the brain? Does V1 "talk to" V2, which then talks to higher areas? Or do multiple regions get activated in parallel?

### Conceptual Framework
Think of it like dropping a stone in a connected system of pools. The ripples don't just spread randomly - there are specific pathways. In the brain:
- V1 (primary visual cortex) receives direct input from thalamus
- Higher visual areas (V2, V4, etc.) receive input from V1 and each other
- Thalamus (LGN, LP) both sends and receives visual information
- Hippocampus receives highly processed information

### Analysis Phases

**Phase 1: Temporal Dynamics**
- For each brain region, compute average PSTH (peristimulus time histogram) after stimulus onset
- Measure response latency: when does each region first respond?
- Expected pattern: V1 → higher visual areas → hippocampus (with ~10-50ms delays)

**Phase 2: Pairwise Relationships**
- Cross-correlation between regions: does V1 activity at time t predict V2 activity at time t+delay?
- Noise correlation analysis: decompose correlations into stimulus-driven vs. connectivity-driven components
- Signal-noise decomposition reveals direct anatomical connections vs. indirect information flow

**Phase 3: Information Transfer** (Future Work)
- Mutual information: how much does knowing V1 activity reduce uncertainty about V2 activity?
- Transfer entropy: how much does V1's past help predict V2's future?
- Time-lagged decoding: train decoder on V1 activity to predict stimulus, then test if V2 activity at t+50ms contains the same information

**Phase 4: Stimulus-Dependent Flow** (Future Work)
- Does information flow differently for different stimuli (natural scenes vs. gratings)?
- Are some pathways stronger for certain visual features?

### What Makes This Project Strong
- **Uses unique dataset strength**: Simultaneous multi-region recording with Neuropixels probes
- **Clear narrative arc**: "We tracked how visual information flows through the brain"
- **Builds on core concepts**: Population coding, spike analysis, statistical inference
- **Multiple difficulty levels**: Start simple (latencies, correlations), add complexity as time permits

## Project Structure

```
neuralDataScience_FinalProject/
├── analysis_phase1_temporal_dynamics.py    # Response latencies, PSTHs
├── analysis_phase2_pairwise_relationships.py  # Cross-correlations, noise correlations
├── eda.py                                  # Data exploration script
├── finalReport.pdf                         # Final project report
├── results/                                # Output directory
│   ├── phase1_*.png                       # Phase 1 visualizations
│   ├── phase1_*.csv                       # Phase 1 results
│   ├── phase2_*.png                       # Phase 2 visualizations
│   └── phase2_*.csv                       # Phase 2 results
└── README.md                      # This file
```

## Installation & Setup

### 1. Create Conda Environment

The project includes an `environment.yml` file with all required dependencies:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate neuralDataScience

# Install additional packages for Phase 2 analysis
pip install seaborn statsmodels networkx
```

Alternatively, if you prefer to install manually:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels networkx allensdk
```

### 2. Data Directory Setup

The scripts will automatically create an `allen_data` directory in your home folder (`~/allen_data/`) for caching the Allen SDK data. First download will take some time (~several minutes to hours depending on session size).

## Running the Analyses

### Phase 1: Temporal Dynamics

This phase computes response latencies for different brain regions to identify the cascade of information flow.

```bash
cd /home/hy1331/NDS/neuralDataScience_FinalProject
python analysis_phase1_temporal_dynamics.py
```

**Session Selection:**
- Uses **brain_observatory** session type (not functional_connectivity)
- Prioritizes drifting gratings stimulus for strong, reliable visual responses
- Automatically selects session with <500 units for faster processing
- Requires minimum 10 units per region for statistical reliability

**What it does:**
- Loads Allen SDK Neuropixels data for visual regions
- Computes population-based PSTHs (peri-stimulus time histograms) for each region
  - Pools spikes from all units within a region for regional population response
  - Uses 1000 evenly-spaced trials (from total available) for performance
  - 10ms time bins, -200ms to +500ms window around stimulus onset
- Measures response latency using trial-based baseline variance
  - Baseline period: -200ms to 0ms before stimulus
  - Threshold: baseline_mean + 1.5 × trial-to-trial std (not temporal std)
  - This captures biological variability, not measurement noise
- Identifies the temporal cascade: V1 → V2 → higher areas → hippocampus
- Generates visualizations and saves results to CSV

**Outputs:**
- `results/phase1_regional_psths.png` - PSTH for each region with latency markers
- `results/phase1_latency_comparison.png` - Bar chart comparing latencies
- `results/phase1_cascade_diagram.png` - Visual cascade flow diagram
- `results/phase1_latencies.csv` - Numerical latency values

**Expected results:**
- V1 (VISp) should respond first (~40-60ms)
- Higher visual areas should respond ~10-30ms later
- Hippocampus should respond much later (~100-150ms)

**Technical Notes:**
- Population-based PSTH is much faster than per-neuron computation (O(n_trials) vs O(n_units × n_trials))
- Trial-based baseline variance properly captures response reliability
- 1.5 std threshold balances sensitivity vs false positives for population responses

### Phase 2: Pairwise Relationships

This phase analyzes functional connectivity between region pairs.

```bash
python analysis_phase2_pairwise_relationships.py
```

**What it does:**
- Computes cross-correlations between all region pairs
- Decomposes correlations into signal and noise components
  - **Signal correlation**: correlation of mean responses (stimulus-driven)
  - **Noise correlation**: correlation of trial-to-trial fluctuations (connectivity-driven)
  - Uses bootstrapping (1000 iterations) to compute 95% confidence intervals
- Tests functional connectivity patterns across visual and hippocampal regions

**Outputs:**
- `results/phase2_cross_correlations.png` - Correlograms for each pair
- `results/phase2_signal_vs_noise_comparison.png` - Signal vs. noise correlation comparison
- `results/phase2_cross_correlations.csv` - Peak lags and correlations
- `results/phase2_noise_correlations.csv` - Signal/noise correlations with confidence intervals

**Expected results:**
- Visual-visual pairs: Both signal and noise correlations positive (stimulus drive + connectivity)
- Visual-hippocampus pairs: Signal correlation only, minimal noise correlation (stimulus drive only)
- Cross-correlation peak lags reveal temporal relationships between regions

## Understanding the Outputs

### Phase 1: Latency Analysis

The **cascade diagram** shows the temporal ordering of region activation:

```
LGd (Thalamus) ─[Δ10ms]→ VISp (V1) ─[Δ20ms]→ VISl (V2) ─[Δ30ms]→ CA1 (Hippocampus)
```

This indicates information flows from thalamus to V1 to higher visual areas to hippocampus.

### Phase 2: Connectivity Analysis

**Cross-correlation peak lag interpretation:**
- Positive lag: First region leads second region
- Negative lag: Second region leads first region
- Zero lag: Simultaneous activity

**Noise correlation decomposition:**
- **Signal correlation**: How similar are the average responses? (stimulus-driven component)
- **Noise correlation**: How correlated are trial-to-trial fluctuations? (connectivity-driven component)
- Visual cortex pairs with high noise correlation suggest direct anatomical connections
- Visual-hippocampus pairs with low noise correlation suggest indirect information flow
- Bootstrap confidence intervals (95% CI) assess statistical significance

## Key Brain Regions

| Acronym | Full Name | Role |
|---------|-----------|------|
| LGd | Lateral Geniculate Nucleus | Thalamic relay of visual input |
| LP | Lateral Posterior Thalamus | Higher-order thalamus |
| VISp | Primary Visual Cortex | V1, first cortical processing |
| VISl | Lateral Visual Area | V2-like, higher visual |
| VISal | Anterolateral Visual | Higher visual processing |
| VISpm | Posteromedial Visual | Higher visual processing |
| VISam | Anteromedial Visual | Higher visual processing |
| CA1 | Hippocampus CA1 | Memory, spatial processing |
| CA3 | Hippocampus CA3 | Memory, pattern completion |

## Troubleshooting

### Issue: No units found in certain regions

**Solution:** The session might not have recordings from all regions. The script will work with whatever regions are available. You can modify `REGIONS_OF_INTEREST` in the scripts to focus on available regions.

### Issue: Download taking too long / Job hanging after download

**Solution:** The first run downloads ~50-500 MB per session. If the job hangs after download completes, the issue is slow NWB file access on network storage.

**For HPC clusters (recommended):** Copy data to local disk before processing:
```bash
# Add to your SLURM script before running Python:
if [ -d "$HOME/allen_data" ]; then
    echo "Copying Allen data to local /tmp for faster I/O..."
    mkdir -p /tmp/allen_data_$SLURM_JOB_ID
    cp -r $HOME/allen_data/* /tmp/allen_data_$SLURM_JOB_ID/
    export ALLEN_DATA_DIR=/tmp/allen_data_$SLURM_JOB_ID
    echo "✓ Data copied to /tmp (local disk)"
fi

python3 analysis_phase1_temporal_dynamics.py

# Cleanup after completion
rm -rf /tmp/allen_data_$SLURM_JOB_ID
```

This improves NWB file I/O by 100x (from network storage to local SSD).

**Other options:**
1. Let it finish - subsequent runs use cached data
2. Use a different session with fewer units
3. Work with a subset of regions

### Issue: Memory errors

**Solution:** The full session might be too large. Modify the code to:
```python
# Use shorter time window
end_time = start_time + 300  # Only first 5 minutes
```

## Next Steps (Phases 3 & 4)

### Phase 3: Information Transfer (To Be Implemented)
- Mutual information between regions
- Transfer entropy analysis
- Time-lagged decoding

### Phase 4: Stimulus-Dependent Flow (To Be Implemented)
- Compare different stimulus types
- Analyze feature-specific pathways
- Context-dependent connectivity

## Final Report

The complete analysis, results, and interpretations are documented in **`finalReport.pdf`** in this repository. The report includes:
- Detailed methodology for each analysis phase
- Results and visualizations
- Interpretation of cross-region information flow patterns
- Discussion of visual cortex vs. hippocampus connectivity differences
- Statistical analysis with confidence intervals

## Questions or Issues?

1. Check the console output for error messages
2. Verify all packages are installed correctly
3. Make sure you have sufficient disk space for Allen SDK cache
4. Review the project proposal and final report for conceptual understanding

## Data Source

**Allen Brain Observatory – Neuropixels Visual Coding (Brain Observatory 1.1)**

This project uses brain_observatory session type recordings with drifting gratings and other parametric visual stimuli. These sessions provide strong, reliable visual responses suitable for latency and connectivity analysis.

Session characteristics:
- Multi-region simultaneous recordings (Neuropixels probes)
- 7+ visual and hippocampal regions per session
- Multiple stimulus types: drifting gratings, gabors, flashes, natural movies
- 300+ units per session typical
- ~1-2 GB NWB file size per session

## Citation

Data from:
**Allen Institute for Brain Science (2019). Allen Brain Observatory – Neuropixels Visual Coding. Available from: https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels**

Analysis methods based on:
- Cross-correlation: Perkel et al. (1967)
- Noise correlation analysis: Cohen & Kohn (2011)
- Signal-noise decomposition: Averbeck et al. (2006)
- Population PSTH analysis: Churchland et al. (2012)
- Bootstrap confidence intervals: Efron & Tibshirani (1993)
