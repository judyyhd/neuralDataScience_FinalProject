"""
Phase 1: Temporal Dynamics Analysis
Cross-Region Information Flow During Visual Stimuli

This script implements:
- Loading Allen SDK Neuropixels data
- Filtering units by brain region
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

# Allen SDK imports
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# ============================================================================
# CONFIGURATION
# ============================================================================

# Setup paths
DATA_DIR = Path.home() / 'allen_data'
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
PSTH_BIN_SIZE = 0.010  # 10 ms bins
PSTH_WINDOW = (-0.2, 0.5)  # -200ms to +500ms around stimulus onset
BASELINE_WINDOW = (-0.2, 0)  # Baseline period before stimulus
SMOOTHING_SIGMA = 2  # Gaussian smoothing for PSTH (in bins)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_allen_cache():
    """Initialize Allen SDK cache"""
    manifest_path = DATA_DIR / 'manifest.json'
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    return cache

def get_session_with_regions(cache, required_regions=None):
    """
    Find a session that has recordings from multiple visual regions
    
    Parameters:
    -----------
    cache : EcephysProjectCache
    required_regions : list, optional
        Minimum required regions
    
    Returns:
    --------
    session_id : int
        ID of suitable session
    session_info : pd.Series
        Metadata for the session
    """
    sessions = cache.get_session_table()
    
    if required_regions is None:
        required_regions = ['VISp', 'VISl']  # At minimum need V1 and one higher area
    
    # Filter sessions with visual stimulus presentation
    sessions = sessions[sessions['session_type'].str.contains('functional_connectivity', case=False, na=False)]
    
    # Find sessions with multiple regions
    for session_id in sessions.index:
        session_info = sessions.loc[session_id]
        recorded_regions = session_info['ecephys_structure_acronyms']
        
        # Check if required regions are present
        has_required = all(region in recorded_regions for region in required_regions)
        
        if has_required:
            print(f"\n✓ Found suitable session: {session_id}")
            print(f"  Available regions: {recorded_regions}")
            print(f"  Unit count: {session_info['unit_count']}")
            return session_id, session_info
    
    # If no perfect match, return first session with most regions
    session_id = sessions.index[0]
    session_info = sessions.loc[session_id]
    print(f"\n⚠ Using first available session: {session_id}")
    print(f"  Available regions: {session_info['ecephys_structure_acronyms']}")
    return session_id, session_info

def load_session_data(cache, session_id):
    """
    Load spike times, units, and stimulus data for a session
    
    Parameters:
    -----------
    cache : EcephysProjectCache
    session_id : int
    
    Returns:
    --------
    session : EcephysSession object
    units : pd.DataFrame
    spike_times : dict
    stimulus_presentations : pd.DataFrame
    """
    print(f"\nLoading session {session_id}...")
    session = cache.get_session_data(session_id)
    
    # Get units (neurons) with quality filtering
    units = session.units
    print(f"  Total units: {len(units)}")
    
    # Get spike times
    spike_times = session.spike_times
    
    # Get stimulus presentations
    stimulus_presentations = session.stimulus_presentations
    print(f"  Total stimulus presentations: {len(stimulus_presentations)}")
    
    return session, units, spike_times, stimulus_presentations

def filter_units_by_region(units, regions_of_interest):
    """
    Group units by brain region
    
    Parameters:
    -----------
    units : pd.DataFrame
    regions_of_interest : list
    
    Returns:
    --------
    region_units : dict
        Dictionary mapping region name to unit IDs
    """
    region_units = {}
    
    for region in regions_of_interest:
        region_mask = units['ecephys_structure_acronym'] == region
        unit_ids = units[region_mask].index.values
        
        if len(unit_ids) > 0:
            region_units[region] = unit_ids
            print(f"  {region}: {len(unit_ids)} units")
    
    return region_units

# ============================================================================
# PSTH COMPUTATION
# ============================================================================

def compute_psth(spike_times_dict, unit_ids, event_times, window, bin_size):
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
    
    Returns:
    --------
    psth : ndarray
        Average firing rate (Hz) over time
    time_bins : ndarray
        Time bin centers
    trial_matrix : ndarray
        (n_trials, n_bins) spike counts per trial
    """
    # Create time bins
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    time_bins = bins[:-1] + bin_size / 2
    n_bins = len(time_bins)
    n_trials = len(event_times)
    
    # Collect spike counts across all units and trials
    all_counts = []
    
    for unit_id in unit_ids:
        if unit_id not in spike_times_dict:
            continue
            
        spikes = spike_times_dict[unit_id]
        trial_counts = np.zeros((n_trials, n_bins))
        
        for trial_idx, event_time in enumerate(event_times):
            # Get spikes relative to event
            trial_spikes = spikes - event_time
            
            # Bin spikes
            counts, _ = np.histogram(trial_spikes, bins=bins)
            trial_counts[trial_idx, :] = counts
        
        all_counts.append(trial_counts)
    
    # Average across units and trials, convert to Hz
    if len(all_counts) > 0:
        all_counts = np.array(all_counts)  # (n_units, n_trials, n_bins)
        trial_matrix = all_counts.mean(axis=0)  # Average across units: (n_trials, n_bins)
        psth = trial_matrix.mean(axis=0) / bin_size  # Average across trials and convert to Hz
    else:
        trial_matrix = np.zeros((n_trials, n_bins))
        psth = np.zeros(n_bins)
    
    return psth, time_bins, trial_matrix

def compute_region_psths(spike_times_dict, region_units, event_times, window, bin_size):
    """
    Compute PSTH for each brain region
    
    Returns:
    --------
    region_psths : dict
        Dictionary mapping region to (psth, time_bins, trial_matrix)
    """
    region_psths = {}
    
    print("\nComputing PSTHs for each region...")
    for region, unit_ids in region_units.items():
        psth, time_bins, trial_matrix = compute_psth(
            spike_times_dict, unit_ids, event_times, window, bin_size
        )
        region_psths[region] = {
            'psth': psth,
            'time_bins': time_bins,
            'trial_matrix': trial_matrix,
            'n_units': len(unit_ids)
        }
        print(f"  {region}: mean rate = {psth.mean():.2f} Hz")
    
    return region_psths

# ============================================================================
# LATENCY ANALYSIS
# ============================================================================

def compute_response_latency(psth, time_bins, baseline_window, threshold_std=2.0):
    """
    Compute response latency as first time point exceeding baseline + threshold
    
    Parameters:
    -----------
    psth : ndarray
        Firing rate over time
    time_bins : ndarray
        Time bin centers
    baseline_window : tuple
        (start, end) for baseline period
    threshold_std : float
        Number of standard deviations above baseline mean
    
    Returns:
    --------
    latency : float
        Response latency in seconds (or np.nan if no response)
    baseline_mean : float
    baseline_std : float
    threshold : float
    """
    # Get baseline statistics
    baseline_mask = (time_bins >= baseline_window[0]) & (time_bins < baseline_window[1])
    baseline_rates = psth[baseline_mask]
    baseline_mean = baseline_rates.mean()
    baseline_std = baseline_rates.std()
    
    # Define threshold
    threshold = baseline_mean + threshold_std * baseline_std
    
    # Find first time point after stimulus onset exceeding threshold
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
            data['psth'], data['time_bins'], baseline_window, threshold_std
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
    cache = load_allen_cache()
    session_id, session_info = get_session_with_regions(cache, required_regions=['VISp'])
    session, units, spike_times, stimulus_presentations = load_session_data(cache, session_id)
    
    # Filter units by region
    print("\nFiltering units by brain region...")
    region_units = filter_units_by_region(units, REGIONS_OF_INTEREST)
    
    if len(region_units) == 0:
        print("✗ No units found in regions of interest")
        return
    
    # Get stimulus onset times (use first stimulus type available)
    stimulus_types = stimulus_presentations['stimulus_name'].unique()
    print(f"\nAvailable stimulus types: {stimulus_types}")
    
    # Prefer natural scenes or gabors
    preferred_stimuli = ['natural_scenes', 'natural_movie', 'gabors', 'flashes']
    selected_stimulus = None
    for stim in preferred_stimuli:
        matching = [s for s in stimulus_types if stim in s.lower()]
        if matching:
            selected_stimulus = matching[0]
            break
    
    if selected_stimulus is None:
        selected_stimulus = stimulus_types[0]
    
    print(f"Using stimulus type: {selected_stimulus}")
    
    stim_subset = stimulus_presentations[
        stimulus_presentations['stimulus_name'] == selected_stimulus
    ]
    event_times = stim_subset['start_time'].values
    print(f"Number of stimulus presentations: {len(event_times)}")
    
    # Compute PSTHs
    region_psths = compute_region_psths(
        spike_times, region_units, event_times, 
        PSTH_WINDOW, PSTH_BIN_SIZE
    )
    
    # Compute latencies
    latencies_df = compute_region_latencies(
        region_psths, BASELINE_WINDOW, threshold_std=2.0
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
