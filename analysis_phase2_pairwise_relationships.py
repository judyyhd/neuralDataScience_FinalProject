"""
Phase 2: Pairwise Relationships Analysis
Cross-Region Information Flow During Visual Stimuli

This script implements:
- Cross-correlation between brain regions
- Granger causality analysis
- Spike-triggered averages
- Directional connectivity inference
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
CC_BIN_SIZE = 0.020  # 20 ms bins for cross-correlation
CC_MAX_LAG = 0.200   # ±200 ms max lag
STA_WINDOW = (-0.100, 0.020)  # 100ms before to 20ms after spike
STA_BIN_SIZE = 0.002  # 2 ms resolution for STA

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
# GRANGER CAUSALITY ANALYSIS
# ============================================================================

def compute_granger_causality(rate1, rate2, max_lag=10):
    """
    Test if rate1 Granger-causes rate2
    
    Parameters:
    -----------
    rate1, rate2 : ndarray
        Time series
    max_lag : int
        Maximum lag to test
    
    Returns:
    --------
    results : dict
        F-statistics and p-values for each lag
    """
    # Prepare data (need to be same length, no NaNs)
    data = np.column_stack([rate2, rate1])  # [dependent, independent]
    
    # Remove any NaNs or infs
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]
    
    if len(data) < max_lag * 2:
        return None
    
    try:
        # Run Granger causality test
        gc_result = grangercausalitytests(data, max_lag, verbose=False)
        
        # Extract results
        results = {}
        for lag in range(1, max_lag + 1):
            f_stat = gc_result[lag][0]['ssr_ftest'][0]
            p_value = gc_result[lag][0]['ssr_ftest'][1]
            results[lag] = {'f_stat': f_stat, 'p_value': p_value}
        
        return results
    except Exception as e:
        print(f"    Granger causality failed: {e}")
        return None

def compute_pairwise_granger(spike_times_dict, region_units, 
                            start_time, end_time, bin_size=0.050, max_lag=10):
    """
    Compute Granger causality for all directed region pairs
    
    Returns:
    --------
    gc_results : dict
        Dictionary with keys (region1 -> region2) and causality results
    """
    print("\nComputing Granger causality...")
    
    # Get binned rates
    region_rates = {}
    for region, unit_ids in region_units.items():
        rates, _ = bin_spike_trains(
            spike_times_dict, unit_ids, start_time, end_time, bin_size
        )
        region_rates[region] = rates
    
    gc_results = {}
    regions = list(region_units.keys())
    
    for region1 in regions:
        for region2 in regions:
            if region1 == region2:
                continue
            
            print(f"  Testing {region1} -> {region2}...")
            
            rate1 = region_rates[region1]
            rate2 = region_rates[region2]
            
            results = compute_granger_causality(rate1, rate2, max_lag)
            
            if results is not None:
                # Find lag with minimum p-value
                best_lag = min(results.keys(), key=lambda k: results[k]['p_value'])
                best_p = results[best_lag]['p_value']
                best_f = results[best_lag]['f_stat']
                
                gc_results[(region1, region2)] = {
                    'best_lag': best_lag,
                    'best_p_value': best_p,
                    'best_f_stat': best_f,
                    'significant': best_p < 0.05,
                    'all_results': results
                }
                
                sig_str = "✓" if best_p < 0.05 else "✗"
                print(f"    {sig_str} Best lag={best_lag}, p={best_p:.4f}, F={best_f:.2f}")
    
    return gc_results

# ============================================================================
# SPIKE-TRIGGERED AVERAGE
# ============================================================================

def compute_spike_triggered_average(trigger_spikes, target_spikes, window, bin_size):
    """
    Compute spike-triggered average of target region activity
    
    Parameters:
    -----------
    trigger_spikes : ndarray
        Spike times from trigger region
    target_spikes : ndarray
        Spike times from target region
    window : tuple
        (start, end) time window relative to trigger spike
    bin_size : float
    
    Returns:
    --------
    sta : ndarray
        Spike-triggered average (Hz)
    sta_times : ndarray
        Time relative to trigger spike
    """
    # Create bins for STA
    sta_bins = np.arange(window[0], window[1] + bin_size, bin_size)
    sta_times = sta_bins[:-1] + bin_size / 2
    n_sta_bins = len(sta_times)
    
    # Accumulate target spikes around each trigger spike
    all_relative_spikes = []
    
    for trigger_time in trigger_spikes:
        # Get target spikes relative to this trigger
        relative_spikes = target_spikes - trigger_time
        
        # Keep only those in window
        in_window = (relative_spikes >= window[0]) & (relative_spikes < window[1])
        all_relative_spikes.extend(relative_spikes[in_window])
    
    # Histogram all relative spikes
    if len(all_relative_spikes) > 0:
        counts, _ = np.histogram(all_relative_spikes, bins=sta_bins)
        # Convert to Hz: counts / (n_triggers * bin_size)
        sta = counts / (len(trigger_spikes) * bin_size)
    else:
        sta = np.zeros(n_sta_bins)
    
    return sta, sta_times

def compute_pairwise_stas(spike_times_dict, region_units, 
                         start_time, end_time, window, bin_size):
    """
    Compute spike-triggered averages for all region pairs
    Uses spike times directly without pre-binning for better temporal precision
    """
    print("\nComputing spike-triggered averages...")
    
    sta_results = {}
    regions = list(region_units.keys())
    
    for region1 in regions:
        # Get all spikes from trigger region
        trigger_spikes = []
        for unit_id in region_units[region1]:
            if unit_id in spike_times_dict:
                spikes = spike_times_dict[unit_id]
                mask = (spikes >= start_time) & (spikes <= end_time)
                trigger_spikes.extend(spikes[mask])
        
        trigger_spikes = np.array(trigger_spikes)
        
        if len(trigger_spikes) == 0:
            continue
        
        for region2 in regions:
            if region1 == region2:
                continue
            
            # Get all spikes from target region
            target_spikes = []
            for unit_id in region_units[region2]:
                if unit_id in spike_times_dict:
                    spikes = spike_times_dict[unit_id]
                    mask = (spikes >= start_time) & (spikes <= end_time)
                    target_spikes.extend(spikes[mask])
            
            target_spikes = np.array(target_spikes)
            
            # Compute STA directly from spike times
            sta, sta_times = compute_spike_triggered_average(
                trigger_spikes, target_spikes, window, bin_size
            )
            
            sta_results[(region1, region2)] = {
                'sta': sta,
                'times': sta_times * 1000,  # Convert to ms
                'n_trigger_spikes': len(trigger_spikes),
                'n_target_spikes': len(target_spikes)
            }
            
            print(f"  {region1} -> {region2}: {len(trigger_spikes)} trigger spikes, {len(target_spikes)} target spikes")
    
    return sta_results

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

def plot_granger_network(gc_results, output_dir):
    """
    Plot directed network graph based on Granger causality
    """
    import networkx as nx
    
    # Build directed graph
    G = nx.DiGraph()
    
    for (r1, r2), data in gc_results.items():
        if data['significant']:
            G.add_edge(r1, r2, weight=-np.log10(data['best_p_value'] + 1e-10))
    
    if len(G.edges()) == 0:
        print("⚠ No significant Granger causality relationships found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue',
                          edgecolors='black', linewidths=2, ax=ax)
    
    # Draw edges with varying width based on significance
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w/2 for w in weights], 
                          edge_color='gray', arrows=True,
                          arrowsize=20, arrowstyle='->', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    ax.set_title('Granger Causality Network\n(Directed edges indicate significant causal influence)',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_granger_network.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved Granger network to {output_dir / 'phase2_granger_network.png'}")
    plt.close()

def plot_spike_triggered_averages(sta_results, output_dir):
    """
    Plot spike-triggered averages
    """
    if len(sta_results) == 0:
        return
    
    n_pairs = len(sta_results)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, ((r1, r2), data) in enumerate(sta_results.items()):
        ax = axes[idx]
        
        # Smooth STA
        sta_smooth = gaussian_filter1d(data['sta'], sigma=2)
        
        ax.plot(data['times'], sta_smooth, 'k-', linewidth=2)
        ax.axhline(data['sta'].mean(), color='blue', linestyle='--', alpha=0.5, label='Baseline')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Trigger spike')
        
        ax.set_xlabel('Time from trigger (ms)', fontsize=10)
        ax.set_ylabel(f'{r2} rate (Hz)', fontsize=10)
        ax.set_title(f"Trigger: {r1} spike → Response: {r2}\n(n={data['n_trigger_spikes']} spikes)",
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_spike_triggered_averages.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved STAs to {output_dir / 'phase2_spike_triggered_averages.png'}")
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
    region_units = filter_units_by_region(units, REGIONS_OF_INTEREST)
    
    if len(region_units) < 2:
        print("✗ Need at least 2 regions for pairwise analysis")
        return
    
    # Define analysis window (restrict to peri-stimulus periods)
    # Use stimulus-aligned windows to capture information flow during active processing
    peri_window = (-0.2, 0.5)  # 200ms before to 500ms after stimulus
    
    # Collect time segments around each stimulus
    time_segments = []
    for _, stim in stimulus_presentations.iterrows():
        seg_start = stim['start_time'] + peri_window[0]
        seg_end = stim['start_time'] + peri_window[1]
        time_segments.append((seg_start, seg_end))
    
    # For simplicity, use first continuous block (or concatenate multiple)
    # Here we'll use a subset of stimulus presentations to keep it manageable
    max_stims = 1000
    if len(time_segments) > max_stims:
        indices = np.linspace(0, len(time_segments)-1, max_stims, dtype=int)
        time_segments = [time_segments[i] for i in indices]
    
    start_time = time_segments[0][0]
    end_time = time_segments[-1][1]
    
    print(f"\nAnalysis window: {start_time:.1f} to {end_time:.1f} seconds")
    print(f"Duration: {end_time - start_time:.1f} seconds")
    
    # Cross-correlation analysis
    cc_results = compute_pairwise_cross_correlations(
        spike_times, region_units, start_time, end_time,
        CC_BIN_SIZE, CC_MAX_LAG
    )
    
    # Granger causality analysis (use smaller bins for faster interactions)
    gc_results = compute_pairwise_granger(
        spike_times, region_units, start_time, end_time,
        bin_size=0.025, max_lag=10  # 25ms bins to capture 10-50ms interactions
    )
    
    # Spike-triggered averages
    sta_results = compute_pairwise_stas(
        spike_times, region_units, start_time, end_time,
        STA_WINDOW, STA_BIN_SIZE
    )
    
    # Save results
    cc_df = pd.DataFrame([
        {'region1': r1, 'region2': r2, 
         'peak_lag_ms': data['peak_lag'], 'peak_cc': data['peak_cc']}
        for (r1, r2), data in cc_results.items()
    ])
    cc_df.to_csv(OUTPUT_DIR / 'phase2_cross_correlations.csv', index=False)
    
    if len(gc_results) > 0:
        gc_df = pd.DataFrame([
            {'source': r1, 'target': r2,
             'best_lag': data['best_lag'], 'p_value': data['best_p_value'],
             'f_stat': data['best_f_stat'], 'significant': data['significant']}
            for (r1, r2), data in gc_results.items()
        ])
        gc_df.to_csv(OUTPUT_DIR / 'phase2_granger_causality.csv', index=False)
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_cross_correlations(cc_results, OUTPUT_DIR)
    plot_granger_network(gc_results, OUTPUT_DIR)
    plot_spike_triggered_averages(sta_results, OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nRegion pairs analyzed: {len(cc_results)}")
    
    if len(gc_results) > 0:
        sig_gc = sum(1 for data in gc_results.values() if data['significant'])
        print(f"Significant Granger causality relationships: {sig_gc}/{len(gc_results)}")
    
    print("\n✓ Phase 2 analysis complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
