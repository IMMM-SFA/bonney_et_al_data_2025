#!/bin/bash
#SBATCH --job-name=hmm_workflow_array
#SBATCH --array=1-3  # Adjust based on number of combinations in filter_basin_combinations.txt
#SBATCH --output=hmm_workflow_array_%A_%a.out
#SBATCH --error=hmm_workflow_array_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition=normal

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate virtual environment if needed
# source .venv/bin/activate

# Set up environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Read parameters for this job from the combinations file
PARAMS_FILE="filter_basin_combinations.txt"
if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file $PARAMS_FILE not found!"
    exit 1
fi

# Read the line corresponding to this array task ID
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")

# Skip comment lines
if [[ $PARAMS == \#* ]]; then
    echo "Skipping comment line: $PARAMS"
    exit 0
fi

# Parse filter and basin names
FILTER_NAME=$(echo $PARAMS | cut -d',' -f1)
BASIN_NAME=$(echo $PARAMS | cut -d',' -f2)

echo "=========================================="
echo "Processing Filter: $FILTER_NAME"
echo "Processing Basin: $BASIN_NAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

# Change to the workflow directory
cd workflow

# Function to check if a script succeeded
check_success() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1 failed for Filter: $FILTER_NAME, Basin: $BASIN_NAME"
        exit 1
    fi
    echo "$1 completed successfully at: $(date)"
}

# Step 1: Train HMM models
echo "Step 1/4: Training HMM models..."
echo "Start time: $(date)"
python II_HMM_Streamflow_Generation/i_train_hmm_models.py --filter "$FILTER_NAME" --basin "$BASIN_NAME"
check_success "HMM training"

# Step 2: Generate synthetic streamflow
echo "Step 2/4: Generating synthetic streamflow..."
echo "Start time: $(date)"
python II_HMM_Streamflow_Generation/ii_generate_synthetic_streamflow.py --filter "$FILTER_NAME" --basin "$BASIN_NAME"
check_success "Synthetic streamflow generation"

# Step 3: Execute WRAP simulations
echo "Step 3/4: Executing WRAP simulations..."
echo "Start time: $(date)"
python III_WRAP_Execution/i_execute_wrap.py --filter "$FILTER_NAME" --basin "$BASIN_NAME"
check_success "WRAP execution"

# Step 4: Process diversions and reservoirs
echo "Step 4/4: Processing diversions and reservoirs..."
echo "Start time: $(date)"
python III_WRAP_Execution/ii_process_diversions_reservoirs.py --filter "$FILTER_NAME" --basin "$BASIN_NAME"
check_success "Diversions and reservoirs processing"

echo "=========================================="
echo "HMM Workflow Completed Successfully!"
echo "Filter: $FILTER_NAME"
echo "Basin: $BASIN_NAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "End time: $(date)"
echo "=========================================="

# Print resource usage for this job
echo "Job completed. Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,ExitCode
