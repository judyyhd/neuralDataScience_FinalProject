"""
Data Loading and Preprocessing Module
Allen SDK Neuropixels Data Access

This module provides reusable functions for:
- Initializing Allen SDK cache
- Finding sessions with specific brain regions
- Loading session data (spike times, units, stimuli)
- Filtering units by brain region with quality checks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default data directory
DEFAULT_DATA_DIR = Path.home() / 'allen_data'
DEFAULT_DATA_DIR.mkdir(exist_ok=True)

# Default minimum units per region
DEFAULT_MIN_UNITS = 10

# ============================================================================
# DATA LOADING
# ============================================================================

def load_allen_cache(data_dir=None):
    """
    Initialize Allen SDK cache
    
    Parameters:
    -----------
    data_dir : Path or str, optional
        Directory for cache storage. Defaults to ~/allen_data
    
    Returns:
    --------
    cache : EcephysProjectCache
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
    
    manifest_path = data_dir / 'manifest.json'
    print(f"\nInitializing Allen SDK cache at: {data_dir}")
    print("This will download metadata files on first run (~1-2 minutes)...")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    print("✓ Cache initialized")
    return cache


def get_session_with_regions(cache, required_regions=None, max_units=500):
    """
    Find a session that has recordings from multiple visual regions
    
    Parameters:
    -----------
    cache : EcephysProjectCache
    required_regions : list, optional
        Minimum required regions. Defaults to ['VISp', 'VISl']
    max_units : int, optional
        Prefer sessions with fewer than this many units for faster loading
    
    Returns:
    --------
    session_id : int
        ID of suitable session
    session_info : pd.Series
        Metadata for the session
    """
    print("\nFetching session table...")
    sessions = cache.get_session_table()
    print(f"✓ Found {len(sessions)} total sessions")
    
    if required_regions is None:
        required_regions = ['VISp', 'VISl']  # At minimum need V1 and one higher area
    
    # Filter sessions with visual stimulus presentation
    # brain_observatory sessions have drifting gratings and other classic visual stimuli
    sessions = sessions[sessions['session_type'].str.contains('brain_observatory', case=False, na=False)]
    print(f"  {len(sessions)} sessions with brain_observatory type")
    
    # Find sessions with multiple regions (prefer smaller sessions for faster loading)
    print(f"\nSearching for session with regions: {required_regions}")
    candidate_sessions = []
    
    for session_id in sessions.index:
        session_info = sessions.loc[session_id]
        recorded_regions = session_info['ecephys_structure_acronyms']
        
        # Check if required regions are present
        has_required = all(region in recorded_regions for region in required_regions)
        
        if has_required:
            candidate_sessions.append((session_id, session_info['unit_count'], session_info))
    
    if candidate_sessions:
        # Sort by unit count (prefer smaller sessions for faster loading)
        candidate_sessions.sort(key=lambda x: x[1])
        
        # Show all candidates
        print(f"\n  Found {len(candidate_sessions)} candidate sessions:")
        for sess_id, n_units, _ in candidate_sessions[:5]:  # Show first 5
            print(f"    Session {sess_id}: {n_units} units")
        
        # Select smallest session with <max_units if possible, otherwise take smallest
        small_sessions = [s for s in candidate_sessions if s[1] < max_units]
        if small_sessions:
            session_id, unit_count, session_info = small_sessions[0]
            print(f"  → Selecting session with <{max_units} units for faster download")
        else:
            session_id, unit_count, session_info = candidate_sessions[0]
            print(f"  → Selecting smallest available session")
        
        print(f"\n✓ Using session: {session_id}")
        print(f"  Available regions: {session_info['ecephys_structure_acronyms']}")
        print(f"  Unit count: {unit_count}")
        print(f"  Estimated download: {unit_count * 0.5:.0f}-{unit_count * 2:.0f} MB")
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
    print("⏳ Downloading NWB file if not cached (this may take 5-30 minutes for first download)...")
    print("   Step 1/4: Fetching session object from Allen SDK...")
    
    import sys
    import time
    sys.stdout.flush()  # Force output to display
    
    start_time = time.time()
    session = cache.get_session_data(session_id)
    elapsed = time.time() - start_time
    print(f"✓ Session object loaded ({elapsed:.1f}s)")
    sys.stdout.flush()
    
    # Get units (neurons) - this triggers actual data loading
    print("   Step 2/4: Loading units table (this may take 5-15 min on first load)...")
    sys.stdout.flush()
    start_time = time.time()
    units = session.units
    elapsed = time.time() - start_time
    print(f"  ✓ Total units: {len(units)} ({elapsed:.1f}s)")
    sys.stdout.flush()
    
    # Get spike times - can be slow for large sessions
    print("   Step 3/4: Loading spike times (this may take 5-10 min)...")
    sys.stdout.flush()
    start_time = time.time()
    spike_times = session.spike_times
    elapsed = time.time() - start_time
    print(f"  ✓ Spike times loaded for {len(spike_times)} units ({elapsed:.1f}s)")
    sys.stdout.flush()
    
    # Get stimulus presentations
    print("   Step 4/4: Loading stimulus presentations...")
    sys.stdout.flush()
    start_time = time.time()
    stimulus_presentations = session.stimulus_presentations
    elapsed = time.time() - start_time
    print(f"  ✓ Total stimulus presentations: {len(stimulus_presentations)} ({elapsed:.1f}s)")
    sys.stdout.flush()
    
    return session, units, spike_times, stimulus_presentations


def filter_units_by_region(units, regions_of_interest, min_units=None):
    """
    Group units by brain region and filter out sparse regions
    
    Parameters:
    -----------
    units : pd.DataFrame
        Units table from Allen SDK session
    regions_of_interest : list
        List of brain region acronyms to extract
    min_units : int, optional
        Minimum number of units required to include a region.
        Defaults to DEFAULT_MIN_UNITS (10)
    
    Returns:
    --------
    region_units : dict
        Dictionary mapping region name to unit IDs (only regions with >= min_units)
    """
    if min_units is None:
        min_units = DEFAULT_MIN_UNITS
    
    region_units = {}
    excluded_regions = []
    
    print(f"\nUnit counts per region (minimum {min_units} required):")
    
    for region in regions_of_interest:
        region_mask = units['ecephys_structure_acronym'] == region
        unit_ids = units[region_mask].index.values
        
        if len(unit_ids) >= min_units:
            region_units[region] = unit_ids
            print(f"  ✓ {region}: {len(unit_ids)} units")
        elif len(unit_ids) > 0:
            excluded_regions.append((region, len(unit_ids)))
            print(f"  ✗ {region}: {len(unit_ids)} units (too sparse, excluded)")
        else:
            print(f"  - {region}: 0 units (not recorded)")
    
    if excluded_regions:
        print(f"\n⚠ Excluded {len(excluded_regions)} sparse region(s):")
        for region, count in excluded_regions:
            print(f"    {region}: only {count} unit(s)")
    
    return region_units


def get_stimulus_events(stimulus_presentations, stimulus_name=None, preferred_stimuli=None):
    """
    Extract stimulus onset times for analysis
    
    Parameters:
    -----------
    stimulus_presentations : pd.DataFrame
        Stimulus presentations table from Allen SDK
    stimulus_name : str, optional
        Specific stimulus type to use. If None, will auto-select.
    preferred_stimuli : list, optional
        List of preferred stimulus types in priority order.
        Defaults to ['natural_scenes', 'natural_movie', 'gabors', 'flashes']
    
    Returns:
    --------
    event_times : ndarray
        Array of stimulus onset times
    selected_stimulus : str
        Name of the selected stimulus type
    """
    if preferred_stimuli is None:
        preferred_stimuli = ['natural_scenes', 'natural_movie', 'gabors', 'flashes']
    
    stimulus_types = stimulus_presentations['stimulus_name'].unique()
    print(f"\nAvailable stimulus types: {stimulus_types}")
    
    # Use specified stimulus or auto-select from preferred list
    if stimulus_name is not None:
        selected_stimulus = stimulus_name
    else:
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
    
    return event_times, selected_stimulus


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_complete_dataset(regions_of_interest, data_dir=None, min_units=None, 
                          required_regions=None, max_units=500):
    """
    Convenience function to load complete dataset in one call
    
    Parameters:
    -----------
    regions_of_interest : list
        Brain regions to analyze
    data_dir : Path or str, optional
        Cache directory
    min_units : int, optional
        Minimum units per region
    required_regions : list, optional
        Regions that must be present in session
    max_units : int, optional
        Prefer sessions with fewer units
    
    Returns:
    --------
    cache : EcephysProjectCache
    session_id : int
    session : EcephysSession
    units : pd.DataFrame
    spike_times : dict
    stimulus_presentations : pd.DataFrame
    region_units : dict
    """
    print("=" * 80)
    print("LOADING ALLEN SDK NEUROPIXELS DATA")
    print("=" * 80)
    
    # Load cache and find session
    cache = load_allen_cache(data_dir)
    session_id, session_info = get_session_with_regions(cache, required_regions, max_units)
    
    # Load session data
    session, units, spike_times, stimulus_presentations = load_session_data(cache, session_id)
    
    # Filter units by region
    print("\nFiltering units by brain region...")
    region_units = filter_units_by_region(units, regions_of_interest, min_units)
    
    if len(region_units) == 0:
        print("✗ No regions meet the minimum unit threshold")
        if min_units is None:
            min_units = DEFAULT_MIN_UNITS
        print(f"  Try reducing min_units (currently {min_units})")
        return None
    
    print(f"\n✓ Successfully loaded data with {len(region_units)} region(s)")
    
    return cache, session_id, session, units, spike_times, stimulus_presentations, region_units
