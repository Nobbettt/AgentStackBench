#!/bin/bash
# Script to run SWE-bench Pro instances in batches with concurrency support
# Usage: ./run_probench_batch.sh [OPTIONS] [batch_size] [start_batch] [end_batch] [max_concurrent_batches]
#   OPTIONS:
#     -i, --instance-file FILE  Path to instance IDs file (default: ./pro_instance_ids.txt)
#     -o, --output-dir DIR      Output directory for results (default: ../../../result/pro)
#     --config-dir DIR          Directory for batch config files (default: script dir)
#     --no-cleanup              Keep batch config files after run (for debugging)
#   POSITIONAL:
#     batch_size: Number of instances per batch (default: 10)
#     start_batch: Starting batch number (default: 1)
#     end_batch: Ending batch number (default: 0 means all batches)
#     max_concurrent_batches: Maximum concurrent batches (default: 3)
# See BATCH_SCRIPTS_USAGE.md for full documentation.

# Don't use set -e in concurrent execution as we want to continue even if one batch fails

# Parse optional arguments
CLEANUP_CONFIGS=true
INSTANCE_FILE=""
OUTPUT_DIR_OVERRIDE=""
CONFIG_DIR_OVERRIDE=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--instance-file)
            INSTANCE_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP_CONFIGS=false
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Configuration
BATCH_SIZE=${POSITIONAL[0]:-10}
START_BATCH=${POSITIONAL[1]:-1}
END_BATCH=${POSITIONAL[2]:-0}
MAX_CONCURRENT_BATCHES=${POSITIONAL[3]:-3}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTANCE_FILE="${INSTANCE_FILE:-$SCRIPT_DIR/pro_instance_ids.txt}"
CONFIG_DIR="${CONFIG_DIR_OVERRIDE:-$SCRIPT_DIR}"

# Model configuration
MODEL_CONFIG="gpt-5"
AGENT="CodeActAgent"
MAX_ITER=200
NUM_WORKERS=1
SPLIT="test"
DATASET="ScaleAI/SWE-bench_Pro"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$SCRIPT_DIR/../../../result/pro}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if instance file exists
if [ ! -f "$INSTANCE_FILE" ]; then
    print_error "Instance file not found: $INSTANCE_FILE"
    exit 1
fi

# Read all instance IDs
INSTANCE_IDS=()
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        INSTANCE_IDS+=("$line")
    fi
done < "$INSTANCE_FILE"

TOTAL_INSTANCES=${#INSTANCE_IDS[@]}
TOTAL_BATCHES=$(( (TOTAL_INSTANCES + BATCH_SIZE - 1) / BATCH_SIZE ))

if [ "$END_BATCH" -eq 0 ]; then
    END_BATCH=$TOTAL_BATCHES
fi

print_info "========================================="
print_info "SWE-bench Pro Batch Evaluation"
print_info "========================================="
print_info "Instance file: $INSTANCE_FILE"
print_info "Total instances: $TOTAL_INSTANCES"
print_info "Batch size: $BATCH_SIZE"
print_info "Total batches: $TOTAL_BATCHES"
print_info "Running batches: $START_BATCH to $END_BATCH"
print_info "Dataset: $DATASET"
print_info "Model: $MODEL_CONFIG"
print_info "Agent: $AGENT"
print_info "Max iterations: $MAX_ITER"
print_info "Max concurrent batches: $MAX_CONCURRENT_BATCHES"
print_info "========================================="

# Change to OpenHands root directory
cd "$(dirname "$0")/../../.."

# Function to run a batch
run_batch() {
    local batch_num=$1
    local start_idx=$(( (batch_num - 1) * BATCH_SIZE ))
    local end_idx=$(( start_idx + BATCH_SIZE - 1 ))
    
    # Ensure end_idx doesn't exceed array bounds
    if [ $end_idx -ge $TOTAL_INSTANCES ]; then
        end_idx=$(( TOTAL_INSTANCES - 1 ))
    fi
    
    local batch_instances=()
    for (( i=$start_idx; i<=$end_idx; i++ )); do
        batch_instances+=("${INSTANCE_IDS[$i]}")
    done
    
    local batch_size_actual=${#batch_instances[@]}
    
    print_info ""
    print_info "========================================="
    print_info "Batch $batch_num/$TOTAL_BATCHES"
    print_info "========================================="
    print_info "Instances in this batch: $batch_size_actual"
    print_info "Instance IDs:"
    for instance in "${batch_instances[@]}"; do
        echo "  - $instance"
    done
    print_info "========================================="
    
    # Create batch-specific config file to avoid conflicts in concurrent execution
    local batch_config_file="${CONFIG_DIR}/config_pro_batch_${batch_num}.toml"
    print_info "Creating config file for batch $batch_num..."
    echo "# SWE-bench Pro Batch $batch_num configuration" > "$batch_config_file"
    echo "selected_ids = [" >> "$batch_config_file"
    for (( i=0; i<${#batch_instances[@]}; i++ )); do
        if [ $i -eq $((${#batch_instances[@]} - 1)) ]; then
            echo "    \"${batch_instances[$i]}\"" >> "$batch_config_file"
        else
            echo "    \"${batch_instances[$i]}\"," >> "$batch_config_file"
        fi
    done
    echo "]" >> "$batch_config_file"
    
    print_info "Batch-specific config file created: $batch_config_file"
    print_info "Instance IDs in config: ${batch_instances[*]}"
    
    # Run the evaluation with environment variable to specify config file
    # This avoids conflicts when multiple batches run concurrently
    print_info "Starting evaluation for batch $batch_num..."
    local batch_note="pro-batch-${batch_num}-of-${TOTAL_BATCHES}"
    
    # Export the config file path as environment variable
    # The run_infer.py script will use SWE_BENCH_CONFIG_FILE if set
    export SWE_BENCH_CONFIG_FILE="$batch_config_file"
    
    # Note: We don't use --eval-n-limit here because selected_ids already filters the dataset
    # The filter_dataset function will handle the filtering based on selected_ids in the config file.
    if poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
        --agent-cls $AGENT \
        --llm-config $MODEL_CONFIG \
        --max-iterations $MAX_ITER \
        --eval-num-workers $NUM_WORKERS \
        --eval-output-dir "$OUTPUT_DIR" \
        --eval-note "$batch_note" \
        --dataset $DATASET \
        --split $SPLIT \
        --filter-config "$batch_config_file"; then
        print_success "Batch $batch_num completed successfully!"
    else
        print_error "Batch $batch_num failed!"
        print_warning "Continuing with next batch..."
    fi
    
    # Unset the environment variable
    unset SWE_BENCH_CONFIG_FILE
    
    if [ "$CLEANUP_CONFIGS" = true ]; then
        rm -f "$batch_config_file"
    else
        print_info "Batch config file kept at: $batch_config_file (for debugging)"
    fi
}

# Function to run batch in background and track results
run_batch_async() {
    local batch_num=$1
    local batch_log="${OUTPUT_DIR}/batch_${batch_num}.log"
    
    # Create output dir if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Run batch and redirect output to log file
    if run_batch $batch_num >> "$batch_log" 2>&1; then
        echo "SUCCESS:batch_${batch_num}" > "${OUTPUT_DIR}/batch_${batch_num}.status"
        return 0
    else
        echo "FAILED:batch_${batch_num}" > "${OUTPUT_DIR}/batch_${batch_num}.status"
        return 1
    fi
}

# Run batches with concurrency control
SUCCESSFUL_BATCHES=0
FAILED_BATCHES=0
declare -A PIDS_BATCH_MAP  # Map PID to batch number

# Clean up any existing status files
rm -f "${OUTPUT_DIR}"/batch_*.status "${OUTPUT_DIR}"/batch_*.log

print_info "Starting concurrent batch execution (max $MAX_CONCURRENT_BATCHES concurrent batches)..."

for (( batch=$START_BATCH; batch<=$END_BATCH; batch++ )); do
    # Wait if we've reached max concurrent batches
    while [ ${#PIDS_BATCH_MAP[@]} -ge $MAX_CONCURRENT_BATCHES ]; do
        # Check for finished processes
        for pid in "${!PIDS_BATCH_MAP[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished, wait for it and check result
                wait "$pid"
                exit_code=$?
                
                # Get batch number from map
                finished_batch=${PIDS_BATCH_MAP[$pid]}
                if [ $exit_code -eq 0 ]; then
                    ((SUCCESSFUL_BATCHES++))
                    print_success "Batch $finished_batch completed (concurrent)"
                else
                    ((FAILED_BATCHES++))
                    print_error "Batch $finished_batch failed (concurrent)"
                fi
                
                # Remove finished PID from map
                unset PIDS_BATCH_MAP[$pid]
                break
            fi
        done
        
        # If still at max capacity, wait a bit before checking again
        if [ ${#PIDS_BATCH_MAP[@]} -ge $MAX_CONCURRENT_BATCHES ]; then
            sleep 2
        fi
    done
    
    # Start new batch in background
    print_info "Starting batch $batch (concurrent execution)..."
    run_batch_async $batch &
    bg_pid=$!
    PIDS_BATCH_MAP[$bg_pid]=$batch
    print_info "Batch $batch started (PID: $bg_pid)"
    
    # Small delay to avoid overwhelming the system
    sleep 1
done

# Wait for all remaining batches to complete
print_info "Waiting for all remaining batches to complete..."
for pid in "${!PIDS_BATCH_MAP[@]}"; do
    finished_batch=${PIDS_BATCH_MAP[$pid]}
    
    if wait $pid; then
        ((SUCCESSFUL_BATCHES++))
        print_success "Batch $finished_batch completed (final wait)"
    else
        ((FAILED_BATCHES++))
        print_error "Batch $finished_batch failed (final wait)"
    fi
done

# Summary
print_info ""
print_info "========================================="
print_info "SWE-bench Pro Batch Evaluation Summary"
print_info "========================================="
print_info "Total batches run: $((END_BATCH - START_BATCH + 1))"
print_success "Successful batches: $SUCCESSFUL_BATCHES"
if [ $FAILED_BATCHES -gt 0 ]; then
    print_error "Failed batches: $FAILED_BATCHES"
fi
print_info "========================================="
print_info "Results are saved in:"
print_info "  $OUTPUT_DIR/princeton-nlp__SWE-bench_Pro-test/"
print_info ""
print_info "Batch execution logs:"
print_info "  $OUTPUT_DIR/batch_*.log"
print_info "========================================="

# Clean up batch config files if not already done per-batch
if [ "$CLEANUP_CONFIGS" = true ]; then
    rm -f "${CONFIG_DIR}"/config_pro_batch_*.toml
    print_info "Cleaned up batch config files"
fi

# Exit with error if any batch failed
if [ $FAILED_BATCHES -gt 0 ]; then
    exit 1
fi

exit 0
