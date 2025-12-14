"""
EDA Script for neuralDataScience Labs Data
Extracts and analyzes data from all labs to inform final project focus
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import nbformat
import re

# Set up paths
BASE_PATH = Path("/home/hy1331/NDS/neuralDataScience")
OUTPUT_DIR = Path("/home/hy1331/NDS/neuralDataScience_FinalProject")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize data collection
data_summary = {}

print("=" * 80)
print("NEURAL DATA SCIENCE - DATA EXPLORATION")
print("=" * 80)

def extract_dataframe_info_from_notebook(notebook_path):
    """Extract information about dataframes created in a notebook"""
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        info = {
            'sql_queries': [],
            'table_references': [],
            'dataframe_columns': [],
            'dataframe_shapes': [],
            'dataframe_row_counts': []
        }
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                
                # Extract SQL queries - including %%bigquery magic
                # Check for %%bigquery magic command first
                if '%%bigquery' in source:
                    # Extract the SQL query after %%bigquery
                    sql_match = re.search(r'%%bigquery.*?\n(.*)', source, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        info['sql_queries'].append(sql_match.group(1).strip())
                
                # Also check for regular triple-quote SQL
                sql_matches = re.findall(r'"""(SELECT.*?)"""', source, re.DOTALL | re.IGNORECASE)
                info['sql_queries'].extend(sql_matches)
                
                # Extract BigQuery table references (proper format: project.dataset.table)
                table_matches = re.findall(r'FROM\s+`?([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)`?', source, re.IGNORECASE)
                info['table_references'].extend(table_matches)
                
                # Extract dataframe column names from df.head() outputs
                if hasattr(cell, 'outputs'):
                    for output in cell.outputs:
                        if output.output_type == 'execute_result':
                            # Check if output has data
                            if hasattr(output, 'data'):
                                if 'text/html' in output.data:
                                    # Parse HTML table headers
                                    html_text = output.data['text/html']
                                    header_matches = re.findall(r'<th[^>]*>(.*?)</th>', html_text)
                                    if header_matches and len(header_matches) > 0:
                                        # Filter out empty or index columns
                                        columns = [h for h in header_matches if h and not h.isspace()]
                                        if columns:
                                            info['dataframe_columns'].append(columns)
                                            # Try to get row count from the table
                                            row_matches = re.findall(r'<tr[^>]*>', html_text)
                                            if row_matches:
                                                # Subtract 1 for header row
                                                info['dataframe_row_counts'].append(len(row_matches) - 1)
                                elif 'text/plain' in output.data:
                                    text = output.data['text/plain']
                                    # Look for shape information
                                    shape_match = re.search(r'\((\d+),\s*(\d+)\)', text)
                                    if shape_match:
                                        info['dataframe_shapes'].append((int(shape_match.group(1)), int(shape_match.group(2))))
        
        return info
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# LAB 1: Electrophysiology - Monkey MT (V5) Neuron Response
# ============================================================================
print("\n[LAB 1] Electrophysiology - MT Neuron Data")
print("-" * 80)

try:
    mt_data = pd.read_csv(BASE_PATH / "lab1_ephys_mt" / "mt_neuron.csv")
    print(f"✓ Loaded MT neuron data")
    print(f"  Data shape: {mt_data.shape[0]} rows × {mt_data.shape[1]} columns")
    print(f"  Number of features: {mt_data.shape[1]}")
    print(f"  Column names: {list(mt_data.columns)}")
    print(f"  Data types: {dict(mt_data.dtypes)}")
    print(f"  Memory usage: {mt_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    data_summary['Lab 1: MT Neuron'] = {
        'source': 'CSV file',
        'rows': mt_data.shape[0],
        'columns': mt_data.shape[1],
        'column_names': list(mt_data.columns),
        'dtypes': dict(mt_data.dtypes),
        'memory_mb': mt_data.memory_usage(deep=True).sum() / 1024**2
    }
except Exception as e:
    print(f"✗ Error loading MT data: {e}")

# ============================================================================
# LAB 2: BigQuery Integration (Same MT data as Lab 1)
# ============================================================================
print("\n[LAB 2] BigQuery Integration - MT Neuron Data")
print("-" * 80)

lab2_notebook = BASE_PATH / "lab2_bigquery_integration" / "lab2_ephys_mt_with_bigquery.ipynb"
if lab2_notebook.exists():
    lab2_info = extract_dataframe_info_from_notebook(lab2_notebook)
    print(f"  Data source: BigQuery")
    if lab2_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab2_info['table_references']))}")
    if lab2_info['sql_queries']:
        # Parse first query to extract column names
        first_query = lab2_info['sql_queries'][0]
        # Extract column names from SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', first_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_str = select_match.group(1)
            columns = [c.strip().split()[-1] for c in columns_str.split(',') if c.strip() and c.strip() != '*']
            if columns:
                print(f"  Number of features: {len(columns)}")
                print(f"  Column names: {columns}")
    if lab2_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab2_info['dataframe_columns'][0]}")
    if lab2_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab2_info['dataframe_shapes']}")
    data_summary['Lab 2: MT BigQuery'] = {
        'source': 'BigQuery',
        'tables': list(set(lab2_info['table_references'])),
        'sql_queries_found': len(lab2_info['sql_queries']),
        'dataframe_columns': lab2_info['dataframe_columns'],
        'dataframe_shapes': lab2_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")
    data_summary['Lab 2: MT BigQuery'] = {'error': 'Notebook not found'}

# ============================================================================
# LAB 3: Curve Fitting
# ============================================================================
print("\n[LAB 3] Curve Fitting")
print("-" * 80)

lab3_notebook = BASE_PATH / "lab3_curve_fitting" / "lab3_curve_fitting.ipynb"
if lab3_notebook.exists():
    lab3_info = extract_dataframe_info_from_notebook(lab3_notebook)
    print(f"  Data source: BigQuery")
    if lab3_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab3_info['table_references']))}")
    if lab3_info['sql_queries']:
        for i, query in enumerate(lab3_info['sql_queries'][:2]):  # First 2 queries
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                columns_str = select_match.group(1)
                columns = [c.strip().split()[-1].replace(',', '') for c in columns_str.split(',') if c.strip() and c.strip() != '*']
                if columns:
                    print(f"  Query {i+1} - Number of features: {len(columns)}")
                    print(f"  Query {i+1} - Column names: {columns}")
    if lab3_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab3_info['dataframe_columns'][0]}")
    if lab3_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab3_info['dataframe_shapes']}")
    data_summary['Lab 3: Curve Fitting'] = {
        'source': 'BigQuery',
        'tables': list(set(lab3_info['table_references'])),
        'sql_queries_found': len(lab3_info['sql_queries']),
        'dataframe_columns': lab3_info['dataframe_columns'],
        'dataframe_shapes': lab3_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 4: Population Coding Data
# ============================================================================
print("\n[LAB 4] Population Coding Data")
print("-" * 80)

lab4_notebook = BASE_PATH / "lab4_population_coding_data" / "lab4_population_coding_data.ipynb"
if lab4_notebook.exists():
    lab4_info = extract_dataframe_info_from_notebook(lab4_notebook)
    print(f"  Data source: BigQuery")
    if lab4_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab4_info['table_references']))}")
    if lab4_info['sql_queries']:
        for i, query in enumerate(lab4_info['sql_queries'][:2]):
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                columns_str = select_match.group(1)
                columns = [c.strip().split()[-1].replace(',', '') for c in columns_str.split(',') if c.strip() and c.strip() != '*']
                if columns:
                    print(f"  Query {i+1} - Number of features: {len(columns)}")
                    print(f"  Query {i+1} - Column names: {columns}")
    if lab4_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab4_info['dataframe_columns'][0]}")
    if lab4_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab4_info['dataframe_shapes']}")
    data_summary['Lab 4: Population Coding'] = {
        'source': 'BigQuery',
        'tables': list(set(lab4_info['table_references'])),
        'sql_queries_found': len(lab4_info['sql_queries']),
        'dataframe_columns': lab4_info['dataframe_columns'],
        'dataframe_shapes': lab4_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 5: Population Coding Algorithm
# ============================================================================
print("\n[LAB 5] Population Coding Algorithm")
print("-" * 80)

lab5_notebook = BASE_PATH / "lab5_population_coding_algorithm" / "lab5_population_coding_algorithm.ipynb"
if lab5_notebook.exists():
    lab5_info = extract_dataframe_info_from_notebook(lab5_notebook)
    print(f"  Data source: BigQuery")
    if lab5_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab5_info['table_references']))}")
    if lab5_info['sql_queries']:
        for i, query in enumerate(lab5_info['sql_queries'][:2]):
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                columns_str = select_match.group(1)
                columns = [c.strip().split()[-1].replace(',', '') for c in columns_str.split(',') if c.strip() and c.strip() != '*']
                if columns:
                    print(f"  Query {i+1} - Number of features: {len(columns)}")
                    print(f"  Query {i+1} - Column names: {columns}")
    if lab5_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab5_info['dataframe_columns'][0]}")
    if lab5_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab5_info['dataframe_shapes']}")
    data_summary['Lab 5: Population Coding Algorithm'] = {
        'source': 'BigQuery',
        'tables': list(set(lab5_info['table_references'])),
        'sql_queries_found': len(lab5_info['sql_queries']),
        'dataframe_columns': lab5_info['dataframe_columns'],
        'dataframe_shapes': lab5_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 6: Mouse Ephys Data
# ============================================================================
print("\n[LAB 6] Mouse Ephys Data")
print("-" * 80)

lab6_notebook = BASE_PATH / "lab6_mouse_ephys_data" / "lab6_mouse_ephys_data.ipynb"
if lab6_notebook.exists():
    lab6_info = extract_dataframe_info_from_notebook(lab6_notebook)
    print(f"  Data source: BigQuery")
    if lab6_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab6_info['table_references']))}")
    if lab6_info['sql_queries']:
        for i, query in enumerate(lab6_info['sql_queries'][:2]):
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                columns_str = select_match.group(1)
                columns = [c.strip().split()[-1].replace(',', '') for c in columns_str.split(',') if c.strip() and c.strip() != '*']
                if columns:
                    print(f"  Query {i+1} - Number of features: {len(columns)}")
                    print(f"  Query {i+1} - Column names: {columns}")
    if lab6_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab6_info['dataframe_columns'][0]}")
    if lab6_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab6_info['dataframe_shapes']}")
    data_summary['Lab 6: Mouse Ephys'] = {
        'source': 'BigQuery',
        'tables': list(set(lab6_info['table_references'])),
        'sql_queries_found': len(lab6_info['sql_queries']),
        'dataframe_columns': lab6_info['dataframe_columns'],
        'dataframe_shapes': lab6_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 7: Mouse Ephys Analysis
# ============================================================================
print("\n[LAB 7] Mouse Ephys Analysis")
print("-" * 80)

lab7_notebook = BASE_PATH / "lab7_mouse_ephys_analysis" / "lab7_mouse_ephys_analysis.ipynb"
if lab7_notebook.exists():
    lab7_info = extract_dataframe_info_from_notebook(lab7_notebook)
    print(f"  Data source: BigQuery")
    if lab7_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab7_info['table_references']))}")
    if lab7_info['sql_queries']:
        for i, query in enumerate(lab7_info['sql_queries'][:2]):
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                columns_str = select_match.group(1)
                columns = [c.strip().split()[-1].replace(',', '') for c in columns_str.split(',') if c.strip() and c.strip() != '*']
                if columns:
                    print(f"  Query {i+1} - Number of features: {len(columns)}")
                    print(f"  Query {i+1} - Column names: {columns}")
    if lab7_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab7_info['dataframe_columns'][0]}")
    if lab7_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab7_info['dataframe_shapes']}")
    data_summary['Lab 7: Mouse Ephys Analysis'] = {
        'source': 'BigQuery',
        'tables': list(set(lab7_info['table_references'])),
        'sql_queries_found': len(lab7_info['sql_queries']),
        'dataframe_columns': lab7_info['dataframe_columns'],
        'dataframe_shapes': lab7_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 8: Mouse LFP Frequency Analysis
# ============================================================================
print("\n[LAB 8] Mouse LFP Frequency Analysis")
print("-" * 80)

lab8_notebook = BASE_PATH / "lab8_mouse_ephys_frequency_analysis" / "lab8_mouse_lfp_frequency_analysis.ipynb"
if lab8_notebook.exists():
    lab8_info = extract_dataframe_info_from_notebook(lab8_notebook)
    print(f"  Data source: BigQuery")
    if lab8_info['table_references']:
        print(f"  Tables used: {', '.join(set(lab8_info['table_references']))}")
    if lab8_info['sql_queries']:
        for i, query in enumerate(lab8_info['sql_queries'][:2]):
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                columns_str = select_match.group(1)
                columns = [c.strip().split()[-1].replace(',', '') for c in columns_str.split(',') if c.strip() and c.strip() != '*']
                if columns:
                    print(f"  Query {i+1} - Number of features: {len(columns)}")
                    print(f"  Query {i+1} - Column names: {columns}")
    if lab8_info['dataframe_columns']:
        print(f"  Actual DataFrame columns from notebook output: {lab8_info['dataframe_columns'][0]}")
    if lab8_info['dataframe_shapes']:
        print(f"  DataFrame shapes observed: {lab8_info['dataframe_shapes']}")
    data_summary['Lab 8: Mouse LFP'] = {
        'source': 'BigQuery',
        'tables': list(set(lab8_info['table_references'])),
        'sql_queries_found': len(lab8_info['sql_queries']),
        'dataframe_columns': lab8_info['dataframe_columns'],
        'dataframe_shapes': lab8_info['dataframe_shapes']
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 9: Neuropixels PCA
# ============================================================================
print("\n[LAB 9] Neuropixels PCA")
print("-" * 80)

lab9_notebook = BASE_PATH / "lab9_neuropixels_pca" / "lab9_neuropixels_data.ipynb"
if lab9_notebook.exists():
    lab9_info = extract_dataframe_info_from_notebook(lab9_notebook)
    print(f"  Data source: Allen SDK - EcephysProjectCache")
    print(f"  Note: This lab uses the Allen Brain Observatory Neuropixels dataset")
    
    # Extract information about Allen SDK data tables
    if lab9_info['dataframe_columns']:
        print(f"  Number of DataFrames found: {len(lab9_info['dataframe_columns'])}")
        for i, cols in enumerate(lab9_info['dataframe_columns'][:3]):  # Show first 3
            # Clean up the column list (remove row numbers)
            actual_cols = [c for c in cols if not c.isdigit() and c not in ['...', '']]
            if actual_cols:
                print(f"  DataFrame {i+1} - Number of features: {len(actual_cols)}")
                print(f"  DataFrame {i+1} - Column names: {actual_cols}")
    
    # Look for X.shape outputs in the notebook
    with open(lab9_notebook, 'r') as f:
        content = f.read()
        # Find X.shape outputs
        shape_matches = re.findall(r'"text/plain":\s*\[\s*"\((\d+),\s*(\d+)\)"', content)
        if shape_matches:
            print(f"  Spike matrix shapes found: {shape_matches}")
    
    data_summary['Lab 9: Neuropixels PCA'] = {
        'source': 'Allen SDK',
        'access_method': 'EcephysProjectCache.from_warehouse()',
        'dataframe_columns': lab9_info['dataframe_columns'],
        'note': 'Large-scale multi-area recordings with sessions, units, probes, channels tables'
    }
else:
    print("  Notebook not found")

# ============================================================================
# LAB 10: Neuropixels NMF
# ============================================================================
print("\n[LAB 10] Neuropixels NMF")
print("-" * 80)

lab10_notebook = BASE_PATH / "lab10_neuropixels_nmf" / "lab10_neuropixels_nmf.ipynb"
if lab10_notebook.exists():
    lab10_info = extract_dataframe_info_from_notebook(lab10_notebook)
    print(f"  Data source: Allen SDK - EcephysProjectCache")
    print(f"  Note: This lab uses the Allen Brain Observatory Neuropixels dataset")
    
    # Extract information about Allen SDK data tables
    if lab10_info['dataframe_columns']:
        print(f"  Number of DataFrames found: {len(lab10_info['dataframe_columns'])}")
        for i, cols in enumerate(lab10_info['dataframe_columns'][:3]):  # Show first 3
            # Clean up the column list (remove row numbers)
            actual_cols = [c for c in cols if not c.isdigit() and c not in ['...', '']]
            if actual_cols:
                print(f"  DataFrame {i+1} - Number of features: {len(actual_cols)}")
                print(f"  DataFrame {i+1} - Column names: {actual_cols}")
    
    # Look for X.shape outputs in the notebook
    with open(lab10_notebook, 'r') as f:
        content = f.read()
        # Find X.shape outputs
        shape_matches = re.findall(r'"text/plain":\s*\[\s*"\((\d+),\s*(\d+)\)"', content)
        if shape_matches:
            print(f"  Spike matrix shapes found: {shape_matches}")
    
    data_summary['Lab 10: Neuropixels NMF'] = {
        'source': 'Allen SDK',
        'access_method': 'EcephysProjectCache.from_warehouse()',
        'dataframe_columns': lab10_info['dataframe_columns'],
        'note': 'Large-scale multi-area recordings with sessions, units tables'
    }
else:
    print("  Notebook not found")

# ============================================================================
# SAVE SUMMARY REPORT
# ============================================================================
summary_file = OUTPUT_DIR / "data_exploration_summary.txt"
with open(summary_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NEURAL DATA SCIENCE - DATA EXPLORATION SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DETAILED DATA INFORMATION BY LAB:\n")
    f.write("=" * 80 + "\n\n")
    
    for lab_name, details in data_summary.items():
        f.write(f"{lab_name}\n")
        f.write("-" * 80 + "\n")
        for key, value in details.items():
            if key != 'data':
                f.write(f"  {key}: {value}\n")
        f.write("\n")

print(f"\n✓ Summary saved to: {summary_file}")

# ============================================================================
# SAVE DATA SUMMARY AS JSON
# ============================================================================
json_file = OUTPUT_DIR / "data_summary.json"
json_summary = {}
for key, value in data_summary.items():
    json_summary[key] = {k: str(v) for k, v in value.items() if k != 'data'}

with open(json_file, 'w') as f:
    json.dump(json_summary, f, indent=2)

print(f"✓ Data summary saved to: {json_file}")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
