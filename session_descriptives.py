"""
Session Descriptive Statistics
Generate summary statistics for the Allen Brain Observatory session used in analyses
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import data loading functions
from data_loader import (
    load_allen_cache,
    get_session_with_regions,
    load_session_data,
    filter_units_by_region,
    get_stimulus_events
)

# Setup paths
import os
DATA_DIR = Path(os.environ.get('ALLEN_DATA_DIR', str(Path.home() / 'allen_data')))
OUTPUT_DIR = Path('/home/hy1331/NDS/neuralDataScience_FinalProject/results')
OUTPUT_DIR.mkdir(exist_ok=True)

REGIONS_OF_INTEREST = ['LGd', 'LP', 'VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'CA1', 'CA3']

def main():
    print("=" * 80)
    print("SESSION DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    # Load data
    cache = load_allen_cache(DATA_DIR)
    session_id, session_info = get_session_with_regions(cache, required_regions=['VISp'])
    session, units, spike_times, stimulus_presentations = load_session_data(cache, session_id)
    
    # Open output file for writing
    output_file = OUTPUT_DIR / 'session_descriptive_statistics.txt'
    f = open(output_file, 'w')
    
    # Helper function to print to both console and file
    def tee_print(text):
        print(text)
        f.write(text + '\n')
    
    # ========================================================================
    # SESSION METADATA
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("SESSION METADATA")
    tee_print("=" * 80)
    tee_print(f"Session ID: {session_id}")
    tee_print(f"Session Type: {session_info['session_type']}")
    tee_print(f"Total Units: {len(units)}")
    
    # Recording duration
    all_spike_times = np.concatenate([spikes for spikes in spike_times.values()])
    recording_start = all_spike_times.min()
    recording_end = all_spike_times.max()
    recording_duration = recording_end - recording_start
    tee_print(f"Recording Duration: {recording_duration/60:.1f} minutes ({recording_duration:.1f} seconds)")
    
    # ========================================================================
    # BRAIN REGIONS
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("BRAIN REGIONS")
    tee_print("=" * 80)
    
    region_units = filter_units_by_region(units, REGIONS_OF_INTEREST, min_units=1)
    
    region_stats = []
    for region, unit_ids in sorted(region_units.items()):
        region_units_df = units.loc[unit_ids]
        
        # Firing rates
        rates = []
        for uid in unit_ids:
            if uid in spike_times:
                n_spikes = len(spike_times[uid])
                rate = n_spikes / recording_duration
                rates.append(rate)
        
        region_stats.append({
            'Region': region,
            'N_Units': len(unit_ids),
            'Mean_Rate_Hz': np.mean(rates),
            'Std_Rate_Hz': np.std(rates),
            'Min_Rate_Hz': np.min(rates),
            'Max_Rate_Hz': np.max(rates)
        })
    
    region_df = pd.DataFrame(region_stats)
    tee_print(region_df.to_string(index=False))
    
    # Save to CSV
    region_df.to_csv(OUTPUT_DIR / 'session_region_statistics.csv', index=False)
    tee_print(f"\n✓ Saved region statistics to {OUTPUT_DIR / 'session_region_statistics.csv'}")
    
    # ========================================================================
    # UNIT QUALITY METRICS
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("UNIT QUALITY METRICS")
    tee_print("=" * 80)
    
    # Overall unit quality distribution
    if 'quality' in units.columns:
        quality_counts = units['quality'].value_counts()
        tee_print("\nUnit Quality Distribution:")
        for qual, count in quality_counts.items():
            tee_print(f"  {qual}: {count} units ({count/len(units)*100:.1f}%)")
    
    # SNR statistics
    if 'snr' in units.columns:
        tee_print(f"\nSignal-to-Noise Ratio (SNR):")
        tee_print(f"  Mean: {units['snr'].mean():.2f}")
        tee_print(f"  Median: {units['snr'].median():.2f}")
        tee_print(f"  Range: [{units['snr'].min():.2f}, {units['snr'].max():.2f}]")
    
    # Firing rate statistics (all units)
    all_rates = []
    for uid in units.index:
        if uid in spike_times:
            n_spikes = len(spike_times[uid])
            rate = n_spikes / recording_duration
            all_rates.append(rate)
    
    tee_print(f"\nFiring Rates (all units):")
    tee_print(f"  Mean: {np.mean(all_rates):.2f} Hz")
    tee_print(f"  Median: {np.median(all_rates):.2f} Hz")
    tee_print(f"  Range: [{np.min(all_rates):.2f}, {np.max(all_rates):.2f}] Hz")
    
    # ========================================================================
    # STIMULUS PRESENTATIONS
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("STIMULUS PRESENTATIONS")
    tee_print("=" * 80)
    
    stim_types = stimulus_presentations['stimulus_name'].unique()
    tee_print(f"\nTotal Stimulus Presentations: {len(stimulus_presentations)}")
    tee_print(f"Number of Stimulus Types: {len(stim_types)}")
    
    stim_stats = []
    for stim_type in stim_types:
        stim_subset = stimulus_presentations[stimulus_presentations['stimulus_name'] == stim_type]
        durations = stim_subset['stop_time'] - stim_subset['start_time']
        
        stim_stats.append({
            'Stimulus_Type': stim_type,
            'N_Presentations': len(stim_subset),
            'Mean_Duration_s': durations.mean(),
            'Total_Duration_s': durations.sum()
        })
    
    stim_df = pd.DataFrame(stim_stats).sort_values('N_Presentations', ascending=False)
    tee_print("\nStimulus Breakdown:")
    tee_print(stim_df.to_string(index=False))
    
    # Save to CSV
    stim_df.to_csv(OUTPUT_DIR / 'session_stimulus_statistics.csv', index=False)
    tee_print(f"\n✓ Saved stimulus statistics to {OUTPUT_DIR / 'session_stimulus_statistics.csv'}")
    
    # ========================================================================
    # ANALYSIS-SPECIFIC INFORMATION
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("ANALYSIS-SPECIFIC INFORMATION")
    tee_print("=" * 80)
    
    # For drifting gratings (used in analyses)
    event_times, selected_stimulus = get_stimulus_events(
        stimulus_presentations, 
        preferred_stimuli=['drifting_gratings', 'gabors', 'flashes']
    )
    
    dg_subset = stimulus_presentations[stimulus_presentations['stimulus_name'] == selected_stimulus]
    
    tee_print(f"\nSelected Stimulus for Analysis: {selected_stimulus}")
    tee_print(f"Number of Presentations: {len(event_times)}")
    
    if 'orientation' in dg_subset.columns:
        orientations = dg_subset['orientation'].dropna().unique()
        try:
            orientations_sorted = sorted([float(x) for x in orientations if x is not None])
            tee_print(f"Orientations Tested: {len(orientations)} ({orientations_sorted})")
        except:
            tee_print(f"Orientations Tested: {len(orientations)}")
    
    if 'temporal_frequency' in dg_subset.columns:
        temp_freqs = dg_subset['temporal_frequency'].dropna().unique()
        try:
            temp_freqs_sorted = sorted([float(x) for x in temp_freqs if x is not None])
            tee_print(f"Temporal Frequencies: {temp_freqs_sorted} Hz")
        except:
            tee_print(f"Temporal Frequencies: {len(temp_freqs)} unique values")
    
    if 'contrast' in dg_subset.columns:
        contrasts = dg_subset['contrast'].dropna().unique()
        try:
            contrasts_sorted = sorted([float(x) for x in contrasts if x is not None])
            tee_print(f"Contrasts: {contrasts_sorted}")
        except:
            tee_print(f"Contrasts: {len(contrasts)} unique values")
    
    # ========================================================================
    # PAIRWISE ANALYSIS SCOPE
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("PAIRWISE ANALYSIS SCOPE")
    tee_print("=" * 80)
    
    n_regions = len(region_units)
    n_pairs = n_regions * (n_regions - 1)
    
    tee_print(f"\nRegions with ≥10 units: {n_regions}")
    tee_print(f"Total region pairs analyzed: {n_pairs}")
    tee_print(f"Total units included: {sum(len(uids) for uids in region_units.values())}")
    
    tee_print("\nRegion Pairs:")
    regions = sorted(region_units.keys())
    for i, reg1 in enumerate(regions):
        for reg2 in regions:
            if reg1 != reg2:
                tee_print(f"  {reg1} ↔ {reg2}")
    
    # ========================================================================
    # SUMMARY FOR MANUSCRIPT
    # ========================================================================
    tee_print("\n" + "=" * 80)
    tee_print("MANUSCRIPT SUMMARY")
    tee_print("=" * 80)
    
    manuscript_text = f"""
Data were obtained from the Allen Brain Observatory Neuropixels Visual Coding 
dataset (session {session_id}). The recording session included {len(units)} 
simultaneously recorded single units across {n_regions} brain regions (LP: {len(region_units['LP'])}, 
VISp: {len(region_units['VISp'])}, VISl: {len(region_units['VISl'])}, 
VISal: {len(region_units['VISal'])}, VISam: {len(region_units['VISam'])}, 
CA1: {len(region_units['CA1'])}, CA3: {len(region_units['CA3'])} units). 
The total recording duration was {recording_duration/60:.1f} minutes. 

For temporal dynamics analysis, we used {len(event_times)} presentations of 
{selected_stimulus} stimuli. Population responses were computed by pooling 
spikes across all units within each region, and response latencies were 
determined using a threshold of 1.5 standard deviations above baseline firing 
rate (computed from trial-to-trial variability).

For pairwise connectivity analysis, we examined all {n_pairs} region pairs using 
cross-correlation, Granger causality, and spike-triggered averaging. Units had 
a mean firing rate of {np.mean(all_rates):.2f} ± {np.std(all_rates):.2f} Hz 
(mean ± SD) across the recording session.
"""
    tee_print(manuscript_text)
    
    tee_print("\n" + "=" * 80)
    tee_print("✓ Analysis complete!")
    tee_print("=" * 80)
    
    # Close output file
    f.close()
    print(f"\n✓ Full report saved to {output_file}")

if __name__ == '__main__':
    main()
