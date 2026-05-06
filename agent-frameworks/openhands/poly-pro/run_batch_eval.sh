#!/bin/bash
# Wrapper script to run PolyBench and/or SWE-bench Pro batch evaluation
# Usage: ./run_batch_eval.sh [OPTIONS] [BENCH] [batch_size] [start_batch] [end_batch] [max_concurrent_batches]
# See RUN_BATCH_EVAL_USAGE.md for full documentation.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWE_BENCH_DIR="${SCRIPT_DIR}/evaluation/benchmarks/swe_bench"

# Defaults
BENCH_MODE="both"
POLY_INSTANCE_FILE=""
PRO_INSTANCE_FILE=""
OUTPUT_BASE_DIR=""
EXTRA_ARGS=()
CLEANUP_FLAG=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${CYAN}========================================${NC}\n${CYAN}$1${NC}\n${CYAN}========================================${NC}"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS] [BENCH] [batch_size] [start_batch] [end_batch] [max_concurrent_batches]

OPTIONS:
  -i, --poly-instance-file FILE   Instance file for PolyBench (default: swe_bench/poly_instance_ids.txt)
  -I, --pro-instance-file FILE    Instance file for SWE-bench Pro (default: swe_bench/pro_instance_ids.txt)
  -o, --output-dir DIR            Base output directory (default: result/; creates result/poly and result/pro)
  --no-cleanup                    Keep batch config files after run
  -h, --help                      Show this help

BENCH: poly | pro | both  (default: both)
  - poly: Run only SWE-PolyBench
  - pro:  Run only SWE-bench Pro
  - both: Run both benchmarks sequentially

POSITIONAL (passed to each batch script):
  batch_size:            Instances per batch (default: 10)
  start_batch:           Start batch number (default: 1)
  end_batch:             End batch number, 0=all (default: 0)
  max_concurrent_batches: Max concurrent batches (default: 3)

Examples:
  $0                                    # Run both with defaults
  $0 poly                               # Run PolyBench only
  $0 -i my_poly.txt -I my_pro.txt both  # Custom instance files
  $0 -o ./outputs poly 15 1 5 4         # PolyBench, batch 1-5, 15/batch, 4 concurrent
  $0 --no-cleanup pro                   # Pro only, keep configs for debugging

See RUN_BATCH_EVAL_USAGE.md for full documentation.
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--poly-instance-file)
            POLY_INSTANCE_FILE="$2"
            shift 2
            ;;
        -I|--pro-instance-file)
            PRO_INSTANCE_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP_FLAG="--no-cleanup"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        poly|pro|both)
            BENCH_MODE="$1"
            shift
            break
            ;;
        -*)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Remaining args are positional for batch scripts
BATCH_SIZE=${1:-10}
START_BATCH=${2:-1}
END_BATCH=${3:-0}
MAX_CONCURRENT=${4:-3}

# Build common args for batch scripts
build_batch_args() {
    local args=()
    [[ -n "$OUTPUT_BASE_DIR" ]] && args+=(-o "${OUTPUT_BASE_DIR}")
    [[ -n "$CLEANUP_FLAG" ]] && args+=($CLEANUP_FLAG)
    args+=("$BATCH_SIZE" "$START_BATCH" "$END_BATCH" "$MAX_CONCURRENT")
    echo "${args[@]}"
}

BATCH_ARGS=$(build_batch_args)

# Verify swe_bench scripts exist
if [[ ! -f "${SWE_BENCH_DIR}/run_polybench_batch.sh" ]] || [[ ! -f "${SWE_BENCH_DIR}/run_probench_batch.sh" ]]; then
    print_error "Batch scripts not found in ${SWE_BENCH_DIR}"
    exit 1
fi

print_header "Poly-Pro Batch Evaluation Wrapper"
print_info "Bench mode: $BENCH_MODE"
print_info "Batch params: size=$BATCH_SIZE, start=$START_BATCH, end=$END_BATCH, concurrent=$MAX_CONCURRENT"
[[ -n "$OUTPUT_BASE_DIR" ]] && print_info "Output base: $OUTPUT_BASE_DIR"
echo ""

cd "$SCRIPT_DIR"

run_polybench() {
    print_header "Running SWE-PolyBench"
    local poly_args=()
    [[ -n "$POLY_INSTANCE_FILE" ]] && poly_args+=(-i "$POLY_INSTANCE_FILE")
    [[ -n "$OUTPUT_BASE_DIR" ]] && poly_args+=(-o "${OUTPUT_BASE_DIR}/poly")
    [[ -n "$CLEANUP_FLAG" ]] && poly_args+=($CLEANUP_FLAG)
    if bash "${SWE_BENCH_DIR}/run_polybench_batch.sh" "${poly_args[@]}" $BATCH_SIZE $START_BATCH $END_BATCH $MAX_CONCURRENT; then
        print_success "PolyBench evaluation completed"
        return 0
    else
        print_error "PolyBench evaluation failed"
        return 1
    fi
}

run_probench() {
    print_header "Running SWE-bench Pro"
    local pro_args=()
    [[ -n "$PRO_INSTANCE_FILE" ]] && pro_args+=(-i "$PRO_INSTANCE_FILE")
    [[ -n "$OUTPUT_BASE_DIR" ]] && pro_args+=(-o "${OUTPUT_BASE_DIR}/pro")
    [[ -n "$CLEANUP_FLAG" ]] && pro_args+=($CLEANUP_FLAG)
    if bash "${SWE_BENCH_DIR}/run_probench_batch.sh" "${pro_args[@]}" $BATCH_SIZE $START_BATCH $END_BATCH $MAX_CONCURRENT; then
        print_success "SWE-bench Pro evaluation completed"
        return 0
    else
        print_error "SWE-bench Pro evaluation failed"
        return 1
    fi
}

EXIT_CODE=0

case "$BENCH_MODE" in
    poly)
        run_polybench || EXIT_CODE=1
        ;;
    pro)
        run_probench || EXIT_CODE=1
        ;;
    both)
        run_polybench || EXIT_CODE=1
        echo ""
        run_probench || EXIT_CODE=1
        ;;
    *)
        print_error "Invalid bench mode: $BENCH_MODE (use poly, pro, or both)"
        exit 1
        ;;
esac

echo ""
print_header "Batch Evaluation Complete"
if [[ $EXIT_CODE -eq 0 ]]; then
    print_success "All evaluations completed successfully"
else
    print_error "Some evaluations failed"
fi
exit $EXIT_CODE
