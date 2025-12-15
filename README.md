# Cross-Region Information Flow Analysis
## Allen Brain Observatory Neuropixels Data

This project analyzes how visual information flows through different brain regions during visual stimuli presentation, using simultaneous multi-region recordings from the Allen Brain Observatory.

## Project Structure

```
neuralDataScience_FinalProject/
├── analysis_phase1_temporal_dynamics.py    # Response latencies, PSTHs
├── analysis_phase2_pairwise_relationships.py  # Cross-correlations, Granger causality
├── eda.py                                  # Data exploration script
├── project_proposal.md                     # Project overview
├── results/                                # Output directory
│   ├── phase1_*.png                       # Phase 1 visualizations
│   ├── phase1_*.csv                       # Phase 1 results
│   ├── phase2_*.png                       # Phase 2 visualizations
│   └── phase2_*.csv                       # Phase 2 results
└── README_ANALYSIS.md                      # This file
```

## Installation & Setup

### 1. Install Required Packages

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

**What it does:**
- Loads Allen SDK Neuropixels data for visual regions
- Computes PSTHs (peri-stimulus time histograms) for each region
- Measures response latency (time to first significant response)
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

### Phase 2: Pairwise Relationships

This phase analyzes functional connectivity between region pairs.

```bash
python analysis_phase2_pairwise_relationships.py
```

**What it does:**
- Computes cross-correlations between all region pairs
- Tests Granger causality (does region A predict region B?)
- Computes spike-triggered averages
- Builds directed connectivity network

**Outputs:**
- `results/phase2_cross_correlations.png` - Correlograms for each pair
- `results/phase2_granger_network.png` - Directed network graph
- `results/phase2_spike_triggered_averages.png` - STAs for each pair
- `results/phase2_cross_correlations.csv` - Peak lags and correlations
- `results/phase2_granger_causality.csv` - Causality statistics

**Expected results:**
- Positive lags from V1 to higher areas (V1 leads)
- Significant Granger causality: V1 → V2 → higher areas
- Thalamus (LGd) should Granger-cause V1

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

**Granger causality:**
- Significant p-value (< 0.05): Region A helps predict region B
- Network graph shows directed edges A → B for significant relationships

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

### Issue: Granger causality errors

**Solution:** Granger causality requires sufficient data and stationarity. If you see errors:
- Increase `bin_size` parameter (e.g., from 0.05 to 0.1 seconds)
- Reduce `max_lag` parameter (e.g., from 10 to 5)
- This is expected for some region pairs

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

## Questions or Issues?

1. Check the console output for error messages
2. Verify all packages are installed correctly
3. Make sure you have sufficient disk space for Allen SDK cache
4. Review the project proposal for conceptual understanding

## Citation

Data from:
**Allen Institute for Brain Science (2019). Allen Brain Observatory – Neuropixels Visual Coding. Available from: https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels**

Analysis methods based on:
- Cross-correlation: Perkel et al. (1967)
- Granger causality: Granger (1969)
- Spike-triggered averaging: Dayan & Abbott (2001)
