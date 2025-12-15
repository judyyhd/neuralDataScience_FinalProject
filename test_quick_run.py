"""
Quick test script to verify Allen SDK setup without long downloads
This uses only metadata and doesn't download full NWB files
"""

import numpy as np
import pandas as pd
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Setup paths
DATA_DIR = Path.home() / 'allen_data'
DATA_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("QUICK TEST: Allen SDK Setup Verification")
print("=" * 80)

# Test 1: Initialize cache (downloads only small metadata files)
print("\n[Test 1] Initializing cache...")
manifest_path = DATA_DIR / 'manifest.json'
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
print("✓ Cache initialized successfully")

# Test 2: Load sessions table
print("\n[Test 2] Loading sessions table...")
sessions = cache.get_session_table()
print(f"✓ Found {len(sessions)} sessions")
print(f"  Session types: {sessions['session_type'].unique()}")

# Test 3: Find visual sessions
print("\n[Test 3] Finding visual sessions...")
visual_sessions = sessions[sessions['session_type'].str.contains('functional_connectivity', case=False, na=False)]
print(f"✓ Found {len(visual_sessions)} functional connectivity sessions")

if len(visual_sessions) > 0:
    # Show info about smallest session
    session_info = visual_sessions.iloc[0]
    session_id = visual_sessions.index[0]
    
    print(f"\n[Test 4] Example session info:")
    print(f"  Session ID: {session_id}")
    print(f"  Unit count: {session_info['unit_count']}")
    print(f"  Regions: {session_info['ecephys_structure_acronyms']}")
    print(f"  Date: {session_info['published_at']}")
    
    # Show which regions are available
    all_regions = set()
    for regions in visual_sessions['ecephys_structure_acronyms']:
        # Filter out NaN values
        valid_regions = [r for r in regions if pd.notna(r)]
        all_regions.update(valid_regions)
    
    print(f"\n[Test 5] Available brain regions across all sessions:")
    for region in sorted(all_regions):
        count = sum(region in [r for r in regions if pd.notna(r)] for regions in visual_sessions['ecephys_structure_acronyms'])
        print(f"  {region}: appears in {count}/{len(visual_sessions)} sessions")
    
    # Check for our target regions
    target_regions = ['VISp', 'VISl', 'VISal', 'LGd', 'LP', 'CA1']
    print(f"\n[Test 6] Checking for target regions:")
    for region in target_regions:
        if region in all_regions:
            print(f"  ✓ {region} available")
        else:
            print(f"  ✗ {region} NOT available")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nYour Allen SDK setup is working correctly.")
print(f"Cache directory: {DATA_DIR}")
print("\nTo run full analysis (will download session data):")
print("  python analysis_phase1_temporal_dynamics.py")
print("\n⚠ First run will download 100MB-2GB and take 5-30 minutes")
print("  Subsequent runs use cached data and are much faster")
