#!/bin/bash
#SBATCH --job-name=phase2_pairwise
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hy1331@nyu.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Copy Allen data to local /tmp for faster access (if it exists)
if [ -d "$HOME/allen_data" ]; then
    echo "Copying Allen data to local /tmp for faster I/O..."
    mkdir -p /tmp/allen_data_$SLURM_JOB_ID
    cp -r $HOME/allen_data/* /tmp/allen_data_$SLURM_JOB_ID/
    export ALLEN_DATA_DIR=/tmp/allen_data_$SLURM_JOB_ID
    echo "✓ Data copied to /tmp (local disk)"
    echo ""
fi

# Activate conda environment (adjust environment name if needed)
# If you have a specific environment, uncomment and modify:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Run Phase 2 analysis
echo "Starting Phase 2: Pairwise Relationships Analysis"
echo ""
python3 analysis_phase2_pairwise_relationships.py

# Cleanup local data
if [ -d "/tmp/allen_data_$SLURM_JOB_ID" ]; then
    echo ""
    echo "Cleaning up local /tmp data..."
    rm -rf /tmp/allen_data_$SLURM_JOB_ID
    echo "✓ Cleanup complete"
fi

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
