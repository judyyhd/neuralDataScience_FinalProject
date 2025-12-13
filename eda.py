"""
EDA Script for neuralDataScience Labs Data
Extracts and analyzes data from all labs to inform final project focus
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_PATH = Path("/Users/judyyhd/Desktop/Fall 25/NDS/neuralDataScience")
OUTPUT_DIR = Path("/Users/judyyhd/Desktop/Fall 25/NDS/final project")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize data collection
data_summary = {}

print("=" * 80)
print("NEURAL DATA SCIENCE - DATA EXPLORATION")
print("=" * 80)

# ============================================================================
# LAB 1: Electrophysiology - Monkey MT (V5) Neuron Response
# ============================================================================
print("\n[LAB 1] Electrophysiology - MT Neuron Data")
print("-" * 80)

try:
    mt_data = pd.read_csv(BASE_PATH / "lab1_ephys_mt" / "mt_neuron.csv")
    print(f"✓ Loaded MT neuron data: {mt_data.shape}")
    print(f"  Columns: {list(mt_data.columns)}")
    print(f"  Data types:\n{mt_data.dtypes}")
    print(f"\n  Summary statistics:")
    print(mt_data.describe())
    data_summary['Lab 1: MT Neuron'] = {
        'source': 'CSV file',
        'shape': mt_data.shape,
        'columns': list(mt_data.columns),
        'data': mt_data
    }
except Exception as e:
    print(f"✗ Error loading MT data: {e}")

# ============================================================================
# LAB 2-10: BigQuery and Allen SDK Data
# ============================================================================
print("\n" + "=" * 80)
print("BIGQUERY & ALLEN SDK DATA SOURCES")
print("=" * 80)

data_sources_info = {
    'Lab 2': {
        'dataset': 'neural-ds-fe73.lab1_ephys.mt',
        'description': 'MT neuron data via BigQuery',
        'fields': ['trial_id', 'spike_times', 'condition_id', 'condition_angle', 'unit_label'],
        'focus': 'Single unit analysis via cloud'
    },
    'Lab 3': {
        'dataset': 'neural-ds-fe73.lab1_ephys.mt',
        'description': 'Curve fitting on MT neuron tuning curves',
        'focus': 'Direction tuning'
    },
    'Lab 4': {
        'dataset': 'Neural population coding data',
        'description': 'Population coding from mouse V1',
        'focus': 'Information theory analysis'
    },
    'Lab 5': {
        'dataset': 'Population coding continuation',
        'description': 'Algorithm implementation for decoding',
        'focus': 'Decoding algorithms'
    },
    'Lab 6': {
        'dataset': 'neural-ds-fe73.lab6_mouse_lfp',
        'description': 'Mouse electrophysiology recordings',
        'focus': 'LFP signal analysis'
    },
    'Lab 7': {
        'dataset': 'neural-ds-fe73.lab6_mouse_lfp',
        'description': 'Mouse ephys signal analysis',
        'focus': 'Temporal dynamics'
    },
    'Lab 8': {
        'dataset': 'neural-ds-fe73.lab6_mouse_lfp.auditory_cortex',
        'description': 'Mouse auditory cortex LFP frequency analysis',
        'fields': ['session', 'condition', 'frequency', 'amplitude', 'trial_num', 'trace'],
        'focus': 'Frequency domain analysis, FFT'
    },
    'Lab 9': {
        'dataset': 'Allen Brain Observatory Neuropixels Data (via EcephysProjectCache)',
        'description': 'Large-scale multi-area neural recordings',
        'focus': 'PCA dimensionality reduction'
    },
    'Lab 10': {
        'dataset': 'Allen Brain Observatory Neuropixels Data',
        'description': 'Neuropixels recordings from multiple brain areas',
        'focus': 'NMF decomposition, factor analysis'
    }
}

print("\nData Sources Overview:")
for lab, info in data_sources_info.items():
    print(f"\n{lab}: {info['description']}")
    print(f"  Dataset: {info['dataset']}")
    if 'fields' in info:
        print(f"  Key fields: {', '.join(info['fields'])}")
    print(f"  Focus: {info['focus']}")

# ============================================================================
# ANALYZE AVAILABLE DATA TYPES AND SIZES
# ============================================================================
print("\n" + "=" * 80)
print("DATA CHARACTERISTICS SUMMARY")
print("=" * 80)

data_types_summary = {
    'Single Unit Electrophysiology': {
        'labs': ['Lab 1', 'Lab 2', 'Lab 3'],
        'data_type': 'Spike times, timestamps',
        'neurons': 'MT neurons (monkey visual cortex)',
        'variables': 'Direction angle, condition ID',
        'size_estimate': 'Medium (~1000s of trials)'
    },
    'Population Coding': {
        'labs': ['Lab 4', 'Lab 5'],
        'data_type': 'Neural population responses',
        'neurons': 'Multiple neurons, decoded stimulus',
        'variables': 'Stimulus information, decoding accuracy',
        'size_estimate': 'Medium-Large'
    },
    'Mouse Electrophysiology (LFP)': {
        'labs': ['Lab 6', 'Lab 7', 'Lab 8'],
        'data_type': 'Local Field Potential time series',
        'neurons': 'Mouse cortex (auditory & other areas)',
        'variables': 'Frequency, amplitude, time series',
        'size_estimate': 'Large (continuous recordings)'
    },
    'Large-Scale Neuropixels': {
        'labs': ['Lab 9', 'Lab 10'],
        'data_type': 'Multi-electrode array recordings',
        'neurons': 'Hundreds of neurons across areas',
        'variables': 'Spike times, session info, brain regions',
        'size_estimate': 'Very Large (100+ neurons/session)'
    }
}

print("\nData Types by Research Topic:\n")
for topic, details in data_types_summary.items():
    print(f"📊 {topic}")
    print(f"   Labs: {', '.join(details['labs'])}")
    print(f"   Type: {details['data_type']}")
    print(f"   Neurons: {details['neurons']}")
    print(f"   Variables: {details['variables']}")
    print(f"   Size: {details['size_estimate']}")
    print()

# ============================================================================
# RECOMMENDATIONS FOR PROJECT FOCUS
# ============================================================================
print("=" * 80)
print("RECOMMENDATIONS FOR PROJECT FOCUS")
print("=" * 80)

recommendations = """
Based on the available data, here are recommended project directions:

1. 🎯 SPIKE TIMING ANALYSIS (Labs 1-3)
   - Most accessible locally (mt_neuron.csv available)
   - Classic neuroscience topic: direction selectivity in MT
   - Could implement: tuning curve fitting, information theory metrics
   - Complexity: Low-Medium | Data size: Small

2. 📈 POPULATION DECODING (Labs 4-5)
   - Intermediate complexity
   - Learn about information theory and decoding algorithms
   - Build classifier to decode stimulus from population response
   - Complexity: Medium | Data size: Medium

3. 🔊 FREQUENCY DOMAIN ANALYSIS (Labs 8)
   - Analyze LFP signals in frequency domain (FFT, spectrograms)
   - Understand oscillations in auditory cortex
   - Could compare frequency responses across conditions
   - Complexity: Medium | Data size: Large

4. 🧠 HIGH-DIMENSIONAL NEURAL DATA (Labs 9-10)
   - Allen Brain Observatory data (very comprehensive)
   - Learn dimensionality reduction (PCA, NMF)
   - Compare neural representations across brain areas
   - Complexity: High | Data size: Very Large

5. 🔄 COMPARATIVE ANALYSIS
   - Compare single-unit (monkey MT) vs population (mouse)
   - Compare time-domain (spikes) vs frequency-domain (LFP)
   - Cross-species, cross-modality comparison
   - Complexity: Medium-High | Data size: Depends on selection

EASIEST START:
→ Start with Lab 1 (mt_neuron.csv locally available)
→ Add Lab 6-8 for larger dataset exploration if desired
→ Use Labs 9-10 if interested in big data techniques

RECOMMENDATION:
For a balanced project, combine Labs 1-3 (accessible) with insights from
Labs 6-8 (larger scale), or focus entirely on Labs 9-10 for a data-heavy project.
"""

print(recommendations)

# ============================================================================
# SAVE SUMMARY REPORT
# ============================================================================
summary_file = OUTPUT_DIR / "data_exploration_summary.txt"
with open(summary_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NEURAL DATA SCIENCE - DATA EXPLORATION SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("AVAILABLE DATA SOURCES:\n")
    f.write("-" * 80 + "\n")
    for lab, info in data_sources_info.items():
        f.write(f"\n{lab}: {info['description']}\n")
        f.write(f"  Dataset: {info['dataset']}\n")
        if 'fields' in info:
            f.write(f"  Key fields: {', '.join(info['fields'])}\n")
        f.write(f"  Focus: {info['focus']}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("DATA TYPES SUMMARY:\n")
    f.write("-" * 80 + "\n")
    for topic, details in data_types_summary.items():
        f.write(f"\n{topic}\n")
        f.write(f"  Labs: {', '.join(details['labs'])}\n")
        f.write(f"  Type: {details['data_type']}\n")
        f.write(f"  Size: {details['size_estimate']}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write(recommendations)

print(f"\n✓ Summary saved to: {summary_file}")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")

try:
    # Visualization 1: Lab 1 MT data overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Lab 1: MT Neuron Data Overview', fontsize=14, fontweight='bold')
    
    # Spike count distribution
    if 'spike_count' in mt_data.columns:
        axes[0, 0].hist(mt_data['spike_count'], bins=20, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Spike Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Spike Count Distribution')
    
    # Direction angle distribution
    if 'direction' in mt_data.columns:
        axes[0, 1].hist(mt_data['direction'], bins=20, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('Direction (degrees)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Direction Angle Distribution')
    
    # Condition distribution
    if 'condition_id' in mt_data.columns:
        condition_counts = mt_data['condition_id'].value_counts()
        axes[1, 0].bar(range(len(condition_counts)), condition_counts.values, color='mediumseagreen', edgecolor='black')
        axes[1, 0].set_xlabel('Condition ID')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Trials per Condition')
    
    # Data shape and columns
    axes[1, 1].axis('off')
    info_text = f"Dataset Shape: {mt_data.shape}\n\nColumns:\n"
    for i, col in enumerate(mt_data.columns[:8]):  # Show first 8 columns
        info_text += f"  • {col}\n"
    if len(mt_data.columns) > 8:
        info_text += f"  ... and {len(mt_data.columns) - 8} more"
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lab1_mt_overview.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: lab1_mt_overview.png")
    plt.close()
    
except Exception as e:
    print(f"✗ Error creating visualizations: {e}")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print(f"\nNext steps:")
print("1. Review the data exploration summary")
print("2. Choose a research focus area")
print("3. Load specific lab data as needed for your project")
