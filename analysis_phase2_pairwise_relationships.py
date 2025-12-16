"""
Phase 2: Pairwise Relationships Analysis
Cross-Region Information Flow During Visual Stimuli

This script implements:
- Cross-correlation between brain regions
- Noise correlation analysis (signal vs. noise decomposition)
- Functional connectivity inference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial

# Import Phase 1 functions for data loading
from analysis_phase1_temporal_dynamics import (
    load_allen_cache, get_session_with_regions, load_session_data,
    filter_units_by_region, compute_psth
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Setup paths - support /tmp data location for faster I/O on HPC
import os
DATA_DIR = Path(os.environ.get('ALLEN_DATA_DIR', str(Path.home() / 'allen_data')))
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path('/home/hy1331/NDS/neuralDataScience_FinalProject/results')
OUTPUT_DIR.mkdir(exist_ok=True)

REGIONS_OF_INTEREST = ['LGd', 'LP', 'VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'CA1', 'CA3']

# Analysis parameters
CC_BIN_SIZE = 0.010  # 10 ms bins for cross-correlation
CC_MAX_LAG = 0.200   # ±200 ms max lag

# ============================================================================
# CROSS-CORRELATION ANALYSIS
# ============================================================================

def bin_spike_trains(spike_times_dict, unit_ids, start_time, end_time, bin_size):
    """
    Bin spike trains for multiple units into firing rate vectors
    
    Parameters:
    -----------
    spike_times_dict : dict
    unit_ids : array-like
    start_time : float
    end_time : float
    bin_size : float
    
    Returns:
    --------
    binned_rates : ndarray
        (n_bins,) average firing rate across units
    time_bins : ndarray
    """
    bins = np.arange(start_time, end_time + bin_size, bin_size)
    n_bins = len(bins) - 1
    
    all_counts = []
    for unit_id in unit_ids:
        if unit_id not in spike_times_dict:
            continue
        spikes = spike_times_dict[unit_id]
        counts, _ = np.histogram(spikes, bins=bins)
        all_counts.append(counts)
    
    if len(all_counts) > 0:
        # Average across units and convert to Hz
        avg_counts = np.mean(all_counts, axis=0)
        binned_rates = avg_counts / bin_size
    else:
        binned_rates = np.zeros(n_bins)
    
    time_bins = bins[:-1] + bin_size / 2
    return binned_rates, time_bins

def compute_cross_correlation(rate1, rate2, max_lag_bins):
    """
    Compute normalized cross-correlation between two rate vectors
    
    Parameters:
    -----------
    rate1, rate2 : ndarray
        Firing rate vectors
    max_lag_bins : int
        Maximum lag in bins
    
    Returns:
    --------
    cc : ndarray
        Cross-correlation values
    lags : ndarray
        Lag values (in bins)
    """
    # Normalize (z-score)
    rate1_z = (rate1 - rate1.mean()) / (rate1.std() + 1e-9)
    rate2_z = (rate2 - rate2.mean()) / (rate2.std() + 1e-9)
    
    # Compute cross-correlation
    cc = signal.correlate(rate1_z, rate2_z, mode='full') / len(rate1_z)
    
    # Extract lags around zero
    center = len(cc) // 2
    cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    
    return cc, lags

def compute_pairwise_cross_correlations(spike_times_dict, region_units, 
                                       start_time, end_time, bin_size, max_lag):
    """
    Compute cross-correlations for all region pairs
    
    Returns:
    --------
    cc_results : dict
        Dictionary with keys (region1, region2) and values (cc, lags, peak_lag, peak_cc)
    """
    max_lag_bins = int(max_lag / bin_size)
    cc_results = {}
    
    print("\nComputing cross-correlations...")
    
    # Get binned rates for all regions
    region_rates = {}
    for region, unit_ids in region_units.items():
        rates, time_bins = bin_spike_trains(
            spike_times_dict, unit_ids, start_time, end_time, bin_size
        )
        region_rates[region] = rates
    
    # Compute pairwise cross-correlations
    regions = list(region_units.keys())
    for i, region1 in enumerate(regions):
        for region2 in regions[i+1:]:
            rate1 = region_rates[region1]
            rate2 = region_rates[region2]
            
            cc, lags = compute_cross_correlation(rate1, rate2, max_lag_bins)
            
            # Find peak (both positive and negative)
            pos_peak_idx = np.argmax(cc)
            neg_peak_idx = np.argmin(cc)
            
            # Use whichever has larger absolute value
            if abs(cc[pos_peak_idx]) > abs(cc[neg_peak_idx]):
                peak_idx = pos_peak_idx
            else:
                peak_idx = neg_peak_idx
                
            peak_lag = lags[peak_idx] * bin_size * 1000  # Convert to ms
            peak_cc = cc[peak_idx]
            
            cc_results[(region1, region2)] = {
                'cc': cc,
                'lags': lags * bin_size * 1000,  # Convert to ms
                'peak_lag': peak_lag,
                'peak_cc': peak_cc
            }
            
            print(f"  {region1} <-> {region2}: peak={peak_cc:.3f} at {peak_lag:.1f}ms")
    
    return cc_results

# ============================================================================
# NOISE CORRELATION ANALYSIS
# ============================================================================

def compute_trial_psth(spike_times_dict, unit_ids, event_times, window, bin_size):
    """
    Compute trial-by-trial PSTH matrix for a region (vectorized)
    
    Parameters:
    -----------
    spike_times_dict : dict
    unit_ids : array-like
    event_times : ndarray
        Stimulus onset times
    window : tuple
        (start, end) time window relative to event
    bin_size : float
    
    Returns:
    --------
    trial_matrix : ndarray, shape (n_trials, n_bins)
        Firing rates (Hz) per trial
    time_bins : ndarray
        Time relative to event
    """
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    n_bins = len(bins) - 1
    n_trials = len(event_times)
    n_units = len(unit_ids)
    
    # Accumulate counts across all units
    trial_matrix = np.zeros((n_trials, n_bins))
    
    # Pool all spikes from all units in region
    all_spikes = []
    for unit_id in unit_ids:
        if unit_id in spike_times_dict:
            all_spikes.extend(spike_times_dict[unit_id])
    
    if len(all_spikes) > 0:
        all_spikes = np.sort(all_spikes)
        
        # Vectorized: for each trial, use searchsorted to find spikes in window
        for trial_idx, event_time in enumerate(event_times):
            # Find spikes in window using binary search
            start_idx = np.searchsorted(all_spikes, event_time + window[0], side='left')
            end_idx = np.searchsorted(all_spikes, event_time + window[1], side='right')
            
            if start_idx < end_idx:
                # Get spikes relative to event
                trial_spikes = all_spikes[start_idx:end_idx] - event_time
                # Bin them
                counts, _ = np.histogram(trial_spikes, bins=bins)
                trial_matrix[trial_idx] = counts
    
    # Convert to firing rate (Hz)
    if n_units > 0:
        trial_matrix = trial_matrix / (bin_size * n_units)
    
    time_bins = bins[:-1] + bin_size / 2
    return trial_matrix, time_bins

def compute_noise_correlations(region1_trials, region2_trials, n_bootstrap=1000, random_seed=11089167):
    """
    Decompose correlation into signal and noise components with bootstrapping
    
    Parameters:
    -----------
    region1_trials, region2_trials : ndarray
        Shape (n_trials, n_time_bins) - firing rates per trial
    n_bootstrap : int
        Number of bootstrap iterations for confidence intervals
    random_seed : int
        Random seed for reproducible bootstrapping
    
    Returns:
    --------
    results : dict
        - signal_corr: correlation of mean responses
        - noise_corr_overall: correlation of trial fluctuations (pooled)
        - noise_corr_ci_lower: 2.5th percentile from bootstrap
        - noise_corr_ci_upper: 97.5th percentile from bootstrap
    """
    n_trials = region1_trials.shape[0]
    
    # Signal = mean response across trials
    signal1 = region1_trials.mean(axis=0)  # (n_time_bins,)
    signal2 = region2_trials.mean(axis=0)
    
    # Noise = trial-by-trial deviations from mean
    noise1 = region1_trials - signal1  # broadcasting
    noise2 = region2_trials - signal2
    
    # Signal correlation (should match cross-correlation at lag=0)
    if len(signal1) > 1:
        signal_corr = np.corrcoef(signal1, signal2)[0, 1]
    else:
        signal_corr = np.nan
    
    # Noise correlation - overall (flatten all time bins and trials)
    noise1_flat = noise1.flatten()
    noise2_flat = noise2.flatten()
    if len(noise1_flat) > 1:
        noise_corr_overall = np.corrcoef(noise1_flat, noise2_flat)[0, 1]
    else:
        noise_corr_overall = np.nan
    
    # Bootstrap confidence intervals for noise correlation
    if n_trials >= 3 and n_bootstrap > 0:
        rng = np.random.RandomState(random_seed)
        bootstrap_corrs = []
        for _ in range(n_bootstrap):
            # Resample trials with replacement
            boot_idx = rng.choice(n_trials, size=n_trials, replace=True)
            boot_noise1 = noise1[boot_idx].flatten()
            boot_noise2 = noise2[boot_idx].flatten()
            
            if len(boot_noise1) > 1:
                boot_corr = np.corrcoef(boot_noise1, boot_noise2)[0, 1]
                if not np.isnan(boot_corr):
                    bootstrap_corrs.append(boot_corr)
        
        if len(bootstrap_corrs) > 0:
            noise_corr_ci_lower = np.percentile(bootstrap_corrs, 2.5)
            noise_corr_ci_upper = np.percentile(bootstrap_corrs, 97.5)
        else:
            noise_corr_ci_lower = np.nan
            noise_corr_ci_upper = np.nan
    else:
        noise_corr_ci_lower = np.nan
        noise_corr_ci_upper = np.nan
    
    return {
        'signal_corr': signal_corr,
        'noise_corr_overall': noise_corr_overall,
        'noise_corr_ci_lower': noise_corr_ci_lower,
        'noise_corr_ci_upper': noise_corr_ci_upper
    }

def compute_pairwise_noise_correlations(spike_times_dict, region_units, 
                                       event_times, window, bin_size):
    """
    Compute noise correlations for all region pairs
    
    Returns:
    --------
    nc_results : dict
        Dictionary with keys (region1, region2) and noise correlation results
    """
    print("\nComputing noise correlations...")
    
    # Get trial matrices for all regions
    region_trials = {}
    for region, unit_ids in region_units.items():
        trial_matrix, time_bins = compute_trial_psth(
            spike_times_dict, unit_ids, event_times, window, bin_size
        )
        region_trials[region] = trial_matrix
        print(f"  {region}: {trial_matrix.shape[0]} trials × {trial_matrix.shape[1]} bins")
    
    # Compute pairwise noise correlations
    nc_results = {}
    regions = list(region_units.keys())
    
    for i, region1 in enumerate(regions):
        for region2 in regions[i+1:]:
            result = compute_noise_correlations(
                region_trials[region1],
                region_trials[region2]
            )
            
            nc_results[(region1, region2)] = result
            
            print(f"  {region1} <-> {region2}: signal_corr={result['signal_corr']:.3f}, "
                  f"noise_corr={result['noise_corr_overall']:.3f} "
                  f"[{result['noise_corr_ci_lower']:.3f}, {result['noise_corr_ci_upper']:.3f}]")
    
    return nc_results



# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cross_correlations(cc_results, output_dir):
    """
    Plot cross-correlation matrices and individual correlograms
    """
    if len(cc_results) == 0:
        return
    
    # Individual correlograms
    n_pairs = len(cc_results)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, ((r1, r2), data) in enumerate(cc_results.items()):
        ax = axes[idx]
        
        ax.plot(data['lags'], data['cc'], 'k-', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(data['peak_lag'], color='red', linestyle='--', alpha=0.7,
                  label=f"Peak: {data['peak_lag']:.1f}ms")
        
        ax.set_xlabel('Lag (ms)', fontsize=10)
        ax.set_ylabel('Cross-correlation', fontsize=10)
        ax.set_title(f"{r1} ← → {r2}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_cross_correlations.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved cross-correlations to {output_dir / 'phase2_cross_correlations.png'}")
    plt.close()

def plot_signal_vs_noise_comparison(nc_results, output_dir):
    """
    Plot comparison of signal vs. noise correlations
    """
    if len(nc_results) == 0:
        return
    
    # Extract data for plotting
    pairs = []
    signal_corrs = []
    noise_corrs = []
    
    for (r1, r2), data in nc_results.items():
        pairs.append(f"{r1}-{r2}")
        signal_corrs.append(data['signal_corr'])
        noise_corrs.append(data['noise_corr_overall'])
    
    # Create comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: Bar plot comparison
    x = np.arange(len(pairs))
    width = 0.35
    
    # Extract confidence intervals
    noise_ci_lower = [nc_results[(r1, r2)]['noise_corr_ci_lower'] for r1, r2 in nc_results.keys()]
    noise_ci_upper = [nc_results[(r1, r2)]['noise_corr_ci_upper'] for r1, r2 in nc_results.keys()]
    noise_errors = [np.array(noise_corrs) - np.array(noise_ci_lower),
                    np.array(noise_ci_upper) - np.array(noise_corrs)]
    
    ax1.bar(x - width/2, signal_corrs, width, label='Signal correlation',
            color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, noise_corrs, width, label='Noise correlation (95% CI)',
            color='coral', alpha=0.8, yerr=noise_errors, capsize=3, error_kw={'linewidth': 1.5})
    
    ax1.set_xlabel('Region Pair', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax1.set_title('Signal vs. Noise Correlation by Region Pair',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pairs, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Signal vs Noise scatter
    ax2.scatter(signal_corrs, noise_corrs, s=100, alpha=0.6, c='purple')
    
    # Add region pair labels
    for i, pair in enumerate(pairs):
        ax2.annotate(pair, (signal_corrs[i], noise_corrs[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Add diagonal reference line
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Signal Correlation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Noise Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('Signal vs. Noise Correlation\n(diagonal = equal contribution)',
                  fontsize=13, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_signal_vs_noise_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved signal vs noise comparison to {output_dir / 'phase2_signal_vs_noise_comparison.png'}")
    plt.close()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """
    Main analysis pipeline for Phase 2: Pairwise Relationships
    """
    print("=" * 80)
    print("PHASE 2: PAIRWISE RELATIONSHIPS ANALYSIS")
    print("Cross-Region Information Flow During Visual Stimuli")
    print("=" * 80)
    
    # Load data (reuse Phase 1 functions)
    from analysis_phase1_temporal_dynamics import (
        PSTH_WINDOW
    )
    
    cache = load_allen_cache(DATA_DIR)
    session_id, session_info = get_session_with_regions(cache, required_regions=['VISp'])
    session, units, spike_times, stimulus_presentations = load_session_data(cache, session_id)
    
    # Filter units by region
    print("\nFiltering units by brain region...")
    import sys
    sys.stdout.flush()
    region_units = filter_units_by_region(units, REGIONS_OF_INTEREST)
    
    if len(region_units) < 2:
        print("✗ Need at least 2 regions for pairwise analysis")
        return
    
    # Get stimulus events (using Phase 1 function)
    print("\nExtracting stimulus events...")
    sys.stdout.flush()
    from data_loader import get_stimulus_events
    event_times, selected_stimulus = get_stimulus_events(
        stimulus_presentations, 
        preferred_stimuli=['drifting_gratings', 'gabors', 'flashes']
    )
    sys.stdout.flush()
    
    # Define analysis window (restrict to peri-stimulus periods)
    # Use stimulus-aligned windows to capture information flow during active processing
    peri_window = (-0.2, 0.5)  # 200ms before to 500ms after stimulus
    
    # Use a subset of stimulus presentations to keep it manageable
    max_stims = 1000
    if len(event_times) > max_stims:
        indices = np.linspace(0, len(event_times)-1, max_stims, dtype=int)
        event_times = event_times[indices]
    
    print(f"Using {len(event_times)} stimulus presentations")
    sys.stdout.flush()
    
    # Define time segments around stimuli (vectorized)
    time_segments = [(t + peri_window[0], t + peri_window[1]) for t in event_times]
    
    start_time = time_segments[0][0]
    end_time = time_segments[-1][1]
    
    print(f"\nAnalysis window: {start_time:.1f} to {end_time:.1f} seconds")
    print(f"Duration: {end_time - start_time:.1f} seconds")
    print(f"Analyzing {len(region_units)} regions = {len(region_units)*(len(region_units)-1)} pairwise relationships")
    sys.stdout.flush()
    
    # Cross-correlation analysis
    print("\n1. Computing cross-correlations...")
    sys.stdout.flush()
    cc_results = compute_pairwise_cross_correlations(
        spike_times, region_units, start_time, end_time,
        CC_BIN_SIZE, CC_MAX_LAG
    )
    print(f"   ✓ Completed {len(cc_results)} pairs")
    sys.stdout.flush()
    
    # Noise correlation analysis (decompose signal vs. noise)
    print("\n2. Computing noise correlations...")
    sys.stdout.flush()
    nc_results = compute_pairwise_noise_correlations(
        spike_times, region_units, event_times,
        window=peri_window, bin_size=CC_BIN_SIZE
    )
    print(f"   ✓ Completed {len(nc_results)} pairs")
    sys.stdout.flush()
    
    # Save results
    cc_df = pd.DataFrame([
        {'region1': r1, 'region2': r2, 
         'peak_lag_ms': data['peak_lag'], 'peak_cc': data['peak_cc']}
        for (r1, r2), data in cc_results.items()
    ])
    cc_df.to_csv(OUTPUT_DIR / 'phase2_cross_correlations.csv', index=False)
    
    if len(nc_results) > 0:
        nc_df = pd.DataFrame([
            {'region1': r1, 'region2': r2,
             'signal_corr': data['signal_corr'],
             'noise_corr': data['noise_corr_overall'],
             'noise_corr_ci_lower': data['noise_corr_ci_lower'],
             'noise_corr_ci_upper': data['noise_corr_ci_upper']}
            for (r1, r2), data in nc_results.items()
        ])
        nc_df.to_csv(OUTPUT_DIR / 'phase2_noise_correlations.csv', index=False)
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_cross_correlations(cc_results, OUTPUT_DIR)
    plot_signal_vs_noise_comparison(nc_results, OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nRegion pairs analyzed: {len(cc_results)}")
    
    if len(nc_results) > 0:
        print(f"\nNoise correlation decomposition (with 95% CI):")
        for (r1, r2), data in nc_results.items():
            print(f"  {r1}-{r2}: signal={data['signal_corr']:.3f}, "
                  f"noise={data['noise_corr_overall']:.3f} [{data['noise_corr_ci_lower']:.3f}, {data['noise_corr_ci_upper']:.3f}]")
    
    print("\n✓ Phase 2 analysis complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
