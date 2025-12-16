"""
Phase 1: Temporal Dynamics Analysis
Cross-Region Information Flow During Visual Stimuli

This script implements:
- Computing PSTHs for each region
- Measuring response latencies
- Visualizing temporal dynamics across regions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Import data loading functions
from data_loader import (
    load_allen_cache,
    get_session_with_regions,
    load_session_data,
    filter_units_by_region,
    get_stimulus_events
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Setup paths
import os
DATA_DIR = Path(os.environ.get('ALLEN_DATA_DIR', str(Path.home() / 'allen_data')))
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path('/home/hy1331/NDS/neuralDataScience_FinalProject/results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Visual regions of interest (following thalamus -> V1 -> higher areas -> hippocampus)
REGIONS_OF_INTEREST = [
    'LGd',   # Lateral geniculate nucleus (thalamus)
    'LP',    # Lateral posterior thalamus
    'VISp',  # Primary visual cortex (V1)
    'VISl',  # Lateral visual area (V2-like)
    'VISal', # Anterolateral visual area
    'VISpm', # Posteromedial visual area
    'VISam', # Anteromedial visual area
    'CA1',   # Hippocampus CA1
    'CA3',   # Hippocampus CA3
]

# Analysis parameters
PSTH_BIN_SIZE = 0.005  # 10 ms bins
PSTH_WINDOW = (-0.2, 0.5)  # -200ms to +500ms around stimulus onset
BASELINE_WINDOW = (-0.2, 0)  # Baseline period before stimulus
SMOOTHING_SIGMA = 2  # Gaussian smoothing for PSTH (in bins)
MIN_UNITS_PER_REGION = 10  # Minimum units required for a region to be included
MAX_TRIALS = 1000  # Limit number of trials for faster computation (use subset)

# ============================================================================
# PSTH COMPUTATION
# ============================================================================

def compute_psth(spike_times_dict, unit_ids, event_times, window, bin_size, max_trials=None):
    """
    Compute peri-stimulus time histogram (PSTH) for multiple units
    
    Parameters:
    -----------
    spike_times_dict : dict
        Dictionary mapping unit_id to spike times
    unit_ids : array-like
        Units to include
    event_times : array-like
        Stimulus onset times
    window : tuple
        (start, end) time window relative to event
    bin_size : float
        Bin size in seconds
    max_trials : int, optional
        Maximum number of trials to use (for speed)
    
    Returns:
    --------
    psth : ndarray
        Average firing rate (Hz) over time
    time_bins : ndarray
        Time bin centers
    trial_matrix : ndarray
        (n_trials, n_bins) spike counts per trial
    """
    # Limit trials if requested
    if max_trials is not None and len(event_times) > max_trials:
        # Use evenly spaced subset
        indices = np.linspace(0, len(event_times)-1, max_trials, dtype=int)
        event_times = event_times[indices]
    
    # Create time bins
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    time_bins = bins[:-1] + bin_size / 2
    n_bins = len(time_bins)
    n_trials = len(event_times)
    
    # Pool all spikes from all units in the region
    all_spikes = []
    for unit_id in unit_ids:
        if unit_id in spike_times_dict:
            all_spikes.append(spike_times_dict[unit_id])
    
    if len(all_spikes) == 0:
        trial_matrix = np.zeros((n_trials, n_bins))
        psth = np.zeros(n_bins)
        return psth, time_bins, trial_matrix
    
    # Concatenate all spikes into one pool (population response)
    pooled_spikes = np.concatenate(all_spikes)
    
    # Compute population PSTH across trials
    trial_matrix = np.zeros((n_trials, n_bins))
    for trial_idx, event_time in enumerate(event_times):
        # Get all spikes relative to this event
        trial_spikes = pooled_spikes - event_time
        
        # Bin spikes
        counts, _ = np.histogram(trial_spikes, bins=bins)
        trial_matrix[trial_idx, :] = counts
    
    # Average across trials and convert to Hz (normalized by number of units)
    psth = trial_matrix.mean(axis=0) / (bin_size * len(unit_ids))
    
    return psth, time_bins, trial_matrix

def compute_region_psths(spike_times_dict, region_units, event_times, window, bin_size, max_trials=None):
    """
    Compute PSTH for each brain region
    
    Returns:
    --------
    region_psths : dict
        Dictionary mapping region to (psth, time_bins, trial_matrix)
    """
    region_psths = {}
    
    n_trials_used = min(len(event_times), max_trials) if max_trials else len(event_times)
    print(f"\nComputing PSTHs for each region (using {n_trials_used}/{len(event_times)} trials)...")
    import sys
    sys.stdout.flush()
    
    for idx, (region, unit_ids) in enumerate(region_units.items()):
        print(f"  [{idx+1}/{len(region_units)}] Processing {region} ({len(unit_ids)} units)...")
        sys.stdout.flush()
        
        psth, time_bins, trial_matrix = compute_psth(
            spike_times_dict, unit_ids, event_times, window, bin_size, max_trials
        )
        region_psths[region] = {
            'psth': psth,
            'time_bins': time_bins,
            'trial_matrix': trial_matrix,
            'n_units': len(unit_ids)
        }
        print(f"      ✓ {region}: mean rate = {psth.mean():.2f} Hz")
        sys.stdout.flush()
    
    return region_psths

# ============================================================================
# LATENCY ANALYSIS
# ============================================================================

def compute_response_latency(psth, time_bins, trial_matrix, baseline_window, bin_size, n_units, threshold_std=2.0):
    """
    Use trial-based variance for threshold, not temporal variance
    
    Parameters:
    -----------
    psth : ndarray
        Average firing rate (Hz) over time
    time_bins : ndarray
        Time bin centers
    trial_matrix : ndarray
        (n_trials, n_bins) spike counts per trial (raw counts, not Hz)
    baseline_window : tuple
        (start, end) time window for baseline
    bin_size : float
        Bin size in seconds
    n_units : int
        Number of units in the region (for converting counts to Hz)
    threshold_std : float
        Number of standard deviations above baseline for threshold
    """
    baseline_mask = (time_bins >= baseline_window[0]) & (time_bins < baseline_window[1])
    
    # Get baseline stats from AVERAGED psth
    baseline_mean = psth[baseline_mask].mean()
    
    # Get baseline variability from TRIALS (not time bins)
    # For each trial, sum all spikes during baseline period, then convert to rate
    baseline_trial_counts = trial_matrix[:, baseline_mask].sum(axis=1)  # Total spike count per trial during baseline
    baseline_duration = baseline_window[1] - baseline_window[0]  # Duration of baseline window
    baseline_trial_rates = baseline_trial_counts / (baseline_duration * n_units)  # Convert to Hz
    baseline_std = baseline_trial_rates.std()  # Variability across TRIALS in Hz
    
    # Now this is the real biological variability in proper units!
    threshold = baseline_mean + threshold_std * baseline_std
    print(f"    DEBUG: baseline_mean={baseline_mean:.2f} Hz")
    print(f"    DEBUG: baseline_std={baseline_std:.2f} Hz")
    print(f"    DEBUG: threshold={threshold:.2f} Hz")
    print(f"    DEBUG: max response rate={psth[time_bins >= 0].max():.2f} Hz")
    print(f"    DEBUG: n_trials={trial_matrix.shape[0]}, n_units={n_units}")
    
    # Rest is the same...
    post_stim_mask = time_bins >= 0
    post_stim_rates = psth[post_stim_mask]
    post_stim_times = time_bins[post_stim_mask]
    
    exceeds_threshold = post_stim_rates > threshold
    
    if exceeds_threshold.any():
        latency_idx = np.argmax(exceeds_threshold)
        latency = post_stim_times[latency_idx]
    else:
        latency = np.nan
    
    return latency, baseline_mean, baseline_std, threshold

def compute_region_latencies(region_psths, baseline_window, threshold_std=2.0):
    """
    Compute response latencies for all regions
    
    Returns:
    --------
    latencies : pd.DataFrame
        DataFrame with latency information per region
    """
    results = []
    
    print("\nComputing response latencies...")
    for region, data in region_psths.items():
        latency, baseline_mean, baseline_std, threshold = compute_response_latency(
            data['psth'], data['time_bins'], data['trial_matrix'], 
            baseline_window, PSTH_BIN_SIZE, data['n_units'], threshold_std
        )
        
        results.append({
            'region': region,
            'latency_ms': latency * 1000 if not np.isnan(latency) else np.nan,
            'baseline_rate': baseline_mean,
            'baseline_std': baseline_std,
            'threshold': threshold,
            'peak_rate': data['psth'].max(),
            'n_units': data['n_units']
        })
        
        if not np.isnan(latency):
            print(f"  {region}: {latency*1000:.1f} ms")
        else:
            print(f"  {region}: No clear response")
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_region_psths(region_psths, latencies_df, output_dir):
    """
    Plot PSTHs for all regions with latency markers
    """
    n_regions = len(region_psths)
    fig, axes = plt.subplots(n_regions, 1, figsize=(12, 2.5 * n_regions), sharex=True)
    
    if n_regions == 1:
        axes = [axes]
    
    # Sort regions by latency for hierarchical display
    latencies_df_sorted = latencies_df.sort_values('latency_ms')
    
    for idx, (_, row) in enumerate(latencies_df_sorted.iterrows()):
        region = row['region']
        if region not in region_psths:
            continue
            
        ax = axes[idx]
        data = region_psths[region]
        
        # Plot PSTH with smoothing
        psth_smooth = gaussian_filter1d(data['psth'], SMOOTHING_SIGMA)
        ax.plot(data['time_bins'] * 1000, psth_smooth, 'k-', linewidth=2, label='Smoothed')
        ax.plot(data['time_bins'] * 1000, data['psth'], 'gray', alpha=0.3, linewidth=1, label='Raw')
        
        # Mark baseline and threshold
        ax.axhline(row['baseline_rate'], color='blue', linestyle='--', alpha=0.5, label='Baseline')
        ax.axhline(row['threshold'], color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        # Mark stimulus onset
        ax.axvline(0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Stimulus')
        
        # Mark latency
        if not np.isnan(row['latency_ms']):
            ax.axvline(row['latency_ms'], color='orange', linestyle='--', linewidth=2, 
                      label=f"Latency: {row['latency_ms']:.1f}ms")
        
        # Formatting
        ax.set_ylabel('Rate (Hz)', fontsize=10)
        ax.set_title(f"{region} (n={row['n_units']} units)", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[-1].set_xlabel('Time from stimulus onset (ms)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_regional_psths.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved PSTH plot to {output_dir / 'phase1_regional_psths.png'}")
    plt.close()

def plot_latency_comparison(latencies_df, output_dir):
    """
    Plot latency comparison across regions
    """
    # Remove regions with no clear response
    latencies_df_valid = latencies_df.dropna(subset=['latency_ms']).sort_values('latency_ms')
    
    if len(latencies_df_valid) == 0:
        print("⚠ No valid latencies to plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Bar plot
    bars = ax.barh(latencies_df_valid['region'], latencies_df_valid['latency_ms'], 
                   color='steelblue', edgecolor='black')
    
    # Color code by hierarchy (if we have expected order)
    hierarchy_order = ['LGd', 'LP', 'VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'CA1', 'CA3']
    colors = plt.cm.viridis(np.linspace(0, 1, len(hierarchy_order)))
    color_map = dict(zip(hierarchy_order, colors))
    
    for bar, region in zip(bars, latencies_df_valid['region']):
        if region in color_map:
            bar.set_color(color_map[region])
    
    ax.set_xlabel('Response Latency (ms)', fontsize=12)
    ax.set_ylabel('Brain Region', fontsize=12)
    ax.set_title('Response Latencies Across Brain Regions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved latency comparison to {output_dir / 'phase1_latency_comparison.png'}")
    plt.close()

def plot_cascade_diagram(latencies_df, output_dir):
    """
    Plot information cascade diagram
    """
    latencies_df_valid = latencies_df.dropna(subset=['latency_ms']).sort_values('latency_ms')
    
    if len(latencies_df_valid) < 2:
        print("⚠ Need at least 2 regions for cascade diagram")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot as cascade
    for idx, (_, row) in enumerate(latencies_df_valid.iterrows()):
        y_pos = len(latencies_df_valid) - idx - 1
        x_pos = row['latency_ms']
        
        # Draw region box
        ax.scatter(x_pos, y_pos, s=500, c='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(x_pos, y_pos, row['region'], ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Draw arrow to next region
        if idx < len(latencies_df_valid) - 1:
            next_row = latencies_df_valid.iloc[idx + 1]
            ax.annotate('', xy=(next_row['latency_ms'], y_pos - 1), 
                       xytext=(x_pos, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
            
            # Add delay annotation
            delay = next_row['latency_ms'] - row['latency_ms']
            mid_x = (x_pos + next_row['latency_ms']) / 2
            mid_y = y_pos - 0.5
            ax.text(mid_x, mid_y, f"Δ{delay:.1f}ms", ha='center', va='center',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Time from Stimulus Onset (ms)', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Visual Information Cascade Through Brain Regions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_cascade_diagram.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved cascade diagram to {output_dir / 'phase1_cascade_diagram.png'}")
    plt.close()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """
    Main analysis pipeline for Phase 1: Temporal Dynamics
    """
    print("=" * 80)
    print("PHASE 1: TEMPORAL DYNAMICS ANALYSIS")
    print("Cross-Region Information Flow During Visual Stimuli")
    print("=" * 80)
    
    # Load data
    cache = load_allen_cache(DATA_DIR)
    session_id, session_info = get_session_with_regions(cache, required_regions=['VISp'])
    session, units, spike_times, stimulus_presentations = load_session_data(cache, session_id)
    
    # Filter units by region
    print("\nFiltering units by brain region...")
    import sys
    sys.stdout.flush()
    
    region_units = filter_units_by_region(units, REGIONS_OF_INTEREST, MIN_UNITS_PER_REGION)
    
    if len(region_units) == 0:
        print("✗ No regions meet the minimum unit threshold")
        print(f"  Try reducing MIN_UNITS_PER_REGION (currently {MIN_UNITS_PER_REGION})")
        return
    
    print(f"\n✓ Proceeding with {len(region_units)} region(s) that meet criteria")
    sys.stdout.flush()
    
    # Get stimulus onset times
    # Use drifting gratings for strong, reliable visual responses
    print("\nExtracting stimulus events...")
    sys.stdout.flush()
    event_times, selected_stimulus = get_stimulus_events(
        stimulus_presentations, 
        preferred_stimuli=['drifting_gratings', 'gabors', 'flashes']  # Prefer gratings
    )
    sys.stdout.flush()
    
    # Compute PSTHs
    region_psths = compute_region_psths(
        spike_times, region_units, event_times, 
        PSTH_WINDOW, PSTH_BIN_SIZE, MAX_TRIALS
    )
    
    # Compute latencies
    # Using 1.5 std instead of 2.0 because trial-to-trial variability is high
    latencies_df = compute_region_latencies(
        region_psths, BASELINE_WINDOW, threshold_std=1.5
    )
    
    # Save results
    latencies_df.to_csv(OUTPUT_DIR / 'phase1_latencies.csv', index=False)
    print(f"\n✓ Saved latencies to {OUTPUT_DIR / 'phase1_latencies.csv'}")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_region_psths(region_psths, latencies_df, OUTPUT_DIR)
    plot_latency_comparison(latencies_df, OUTPUT_DIR)
    plot_cascade_diagram(latencies_df, OUTPUT_DIR)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSession: {session_id}")
    print(f"Regions analyzed: {len(region_psths)}")
    print(f"Stimulus presentations: {len(event_times)}")
    print("\nLatencies (ms):")
    print(latencies_df[['region', 'latency_ms', 'n_units']].to_string(index=False))
    
    if len(latencies_df.dropna(subset=['latency_ms'])) >= 2:
        earliest = latencies_df.loc[latencies_df['latency_ms'].idxmin()]
        latest = latencies_df.loc[latencies_df['latency_ms'].idxmax()]
        print(f"\nEarliest response: {earliest['region']} at {earliest['latency_ms']:.1f} ms")
        print(f"Latest response: {latest['region']} at {latest['latency_ms']:.1f} ms")
        print(f"Total cascade time: {latest['latency_ms'] - earliest['latency_ms']:.1f} ms")
    
    print("\n✓ Phase 1 analysis complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
