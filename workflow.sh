#!/bin/bash
# hmm_workflow_local.sh
# Run HMM workflow combinations without SLURM

# ==== USER CONFIGURATION ====
PARAMS_FILE="filter_basin_combinations.txt"
MAX_PARALLEL_JOBS=8   # adjust based on your CPU policy
CPUS_PER_TASK=5       # used by Python (e.g., NumPy/OpenMP)
LOG_DIR="logs"
WORKFLOW_DIR="workflow"

# Wine configuration (from your README)
export INSTALL_DIR="$HOME/.local/wine"
export WINEPREFIX="$HOME/.wine-hpc"
export WINEARCH="win64"
export PATH="$INSTALL_DIR/bin:$PATH"

# Limit wine to use a fixed range of cores (adjust for your node)
TASKSET_CORES="0-63"

mkdir -p "$LOG_DIR"

echo "Starting HMM workflow on $(hostname)"
echo "Time: $(date)"
echo "Available cores: $(nproc)"
echo "Using up to $MAX_PARALLEL_JOBS parallel jobs"

export OMP_NUM_THREADS=$CPUS_PER_TASK
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run one combination
run_job() {
    local line_num=$1
    local params=$2
    local logfile="${LOG_DIR}/hmm_workflow_${line_num}.log"

    # Skip comments
    if [[ $params == \#* || -z $params ]]; then
        echo "Skipping line $line_num: $params" | tee -a "$logfile"
        return
    fi

    FILTER_NAME=$(echo $params | cut -d',' -f1)
    BASIN_NAME=$(echo $params | cut -d',' -f2)

    echo "----------------------------------------" | tee -a "$logfile"
    echo "Running Filter: $FILTER_NAME | Basin: $BASIN_NAME" | tee -a "$logfile"
    echo "Start Time: $(date)" | tee -a "$logfile"
    echo "----------------------------------------" | tee -a "$logfile"

    cd "$WORKFLOW_DIR" || exit 1

    # Step 1
    echo "[1/4] Training HMM..." | tee -a "../$logfile"
    python II_HMM_Streamflow_Generation/i_train_hmm_models.py --filter "$FILTER_NAME" --basin "$BASIN_NAME" >> "../$logfile" 2>&1 || { echo "Failed: HMM training" | tee -a "../$logfile"; exit 1; }

    # Step 2
    echo "[2/4] Generating synthetic streamflow..." | tee -a "../$logfile"
    python II_HMM_Streamflow_Generation/ii_generate_synthetic_streamflow.py --filter "$FILTER_NAME" --basin "$BASIN_NAME" >> "../$logfile" 2>&1 || { echo "Failed: Synthetic generation" | tee -a "../$logfile"; exit 1; }

    # Step 3
    echo "[3/4] Executing WRAP simulations..." | tee -a "../$logfile"

    if [ ! -d "$WINEPREFIX" ]; then
        echo "Initializing Wine prefix..." | tee -a "../$logfile"
        taskset -c $TASKSET_CORES wineboot >> "../$logfile" 2>&1
    fi

    python III_WRAP_Execution/i_execute_wrap.py --filter "$FILTER_NAME" --basin "$BASIN_NAME" >> "../$logfile" 2>&1 || { echo "Failed: WRAP execution" | tee -a "../$logfile"; exit 1; }

    # Step 4
    echo "[4/4] Processing diversions/reservoirs..." | tee -a "../$logfile"
    python III_WRAP_Execution/ii_process_diversions_reservoirs.py --filter "$FILTER_NAME" --basin "$BASIN_NAME" >> "../$logfile" 2>&1 || { echo "Failed: Diversions/reservoirs" | tee -a "../$logfile"; exit 1; }

    cd - >/dev/null

    echo "✅ Completed $FILTER_NAME / $BASIN_NAME at $(date)" | tee -a "$logfile"
}

# ==== MAIN LOOP ====
if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file $PARAMS_FILE not found!"
    exit 1
fi

line_num=0
while IFS= read -r line; do
    ((line_num++))
    # Each job runs in its own subshell (&)
    (
        # Activate your Python virtual environment
        if [ -d ".venv" ]; then
            echo "Activating virtual environment for job $line_num"
            source .venv/bin/activate
        else
            echo "WARNING: .venv directory not found, skipping activation."
        fi

        run_job "$line_num" "$line"
    ) &
    
    # Limit parallel jobs
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
        sleep 5
    done
done < "$PARAMS_FILE"

wait

echo "=========================================="
echo "All workflow combinations completed."
echo "End Time: $(date)"
echo "=========================================="
