#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${OPENHANDS_RUN_DIR:-$SCRIPT_DIR}"
CONFIG_FILE="$RUN_DIR/config.toml"
RUN_INFER_SCRIPT="$RUN_DIR/evaluation/benchmarks/swe_bench/scripts/run_infer.sh"

PRESET=""                  # quick / medium / verified / full
LIMIT=""                   # If set explicitly, overrides PRESET (unless instance file is used)
CONCURRENCY=4
MAX_ITERS=100
AGENT="CodeActAgent"
MODEL="llm.gpt5"
DATASET="princeton-nlp/SWE-bench_Verified"
INSTANCE_FILE=""           # Optional: file with one instance_id per line
NO_CONFIRM=false

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Examples:
  # Default: 174 verified instances, 4 workers
  $(basename "$0")

  # Quick check (10 instances)
  $(basename "$0") --preset quick

  # Medium scale (50 instances), no confirmation
  $(basename "$0") --preset medium --no-confirm

  # Full 500 instances (expensive)
  $(basename "$0") --preset full

  # Custom: 30 instances, 8 workers, 200 max iterations
  $(basename "$0") --limit 30 --concurrency 8 --max-iters 200

Options:
  --preset {quick|medium|verified|full}
      - quick   : 10  instances (quick sanity check)
      - medium  : 50  instances (medium scale)
      - verified: 174 verified instances (default / recommended)
      - full    : 500 instances (full SWE-bench Verified, expensive)

  --limit N
      Override instance count, takes precedence over --preset.

  --concurrency N
      Number of parallel workers (default 4).

  --max-iters N
      Max iterations per instance (default 100).

  --agent NAME
      OpenHands agent name (default CodeActAgent).

  --model NAME
      Model identifier (default llm.eval_gpt5).

  --dataset NAME
      Hugging Face dataset name (default princeton-nlp/SWE-bench_Verified).

  --instance-file PATH
      File containing instance_ids (one per line). For SWE-bench Verified, this
      will be written into evaluation/benchmarks/swe_bench/config.toml as
      selected_ids, and the run will use exactly these instances (LIMIT is
      derived from the file and overrides --preset/--limit).

  --no-confirm
      Do not ask for confirmation (useful for CI / automation).

  -h, --help
      Show this help message.

Notes:
  - If selected_500_instances.csv exists but verified_instances.txt does not,
    the script will auto-generate verified_instances.txt for later reuse.
EOF
}

##############################################################################
# Argument parsing
##############################################################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="${2:-}"; shift 2 ;;
    --preset=*)
      PRESET="${1#*=}"; shift 1 ;;
    --limit)
      LIMIT="${2:-}"; shift 2 ;;
    --limit=*)
      LIMIT="${1#*=}"; shift 1 ;;
    --concurrency)
      CONCURRENCY="${2:-}"; shift 2 ;;
    --concurrency=*)
      CONCURRENCY="${1#*=}"; shift 1 ;;
    --max-iters)
      MAX_ITERS="${2:-}"; shift 2 ;;
    --max-iters=*)
      MAX_ITERS="${1#*=}"; shift 1 ;;
    --agent)
      AGENT="${2:-}"; shift 2 ;;
    --agent=*)
      AGENT="${1#*=}"; shift 1 ;;
    --model)
      MODEL="${2:-}"; shift 2 ;;
    --model=*)
      MODEL="${1#*=}"; shift 1 ;;
    --dataset)
      DATASET="${2:-}"; shift 2 ;;
    --dataset=*)
      DATASET="${1#*=}"; shift 1 ;;
    --instance-file)
      INSTANCE_FILE="${2:-}"; shift 2 ;;
    --instance-file=*)
      INSTANCE_FILE="${1#*=}"; shift 1 ;;
    --no-confirm)
      NO_CONFIRM=true; shift 1 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 1 ;;
  esac
done

##############################################################################
# Pre-flight checks
##############################################################################

cd "$RUN_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config file not found: $CONFIG_FILE"
  echo "Please create and configure config.toml under the verified directory."
  exit 1
fi

if [ ! -x "$RUN_INFER_SCRIPT" ]; then
  if [ -f "$RUN_INFER_SCRIPT" ]; then
    chmod +x "$RUN_INFER_SCRIPT"
  else
    echo "ERROR: run_infer script not found: $RUN_INFER_SCRIPT"
    exit 1
  fi
fi

# Check that API key has been configured (handle two placeholder styles)
if grep -q 'api_key = ""' "$CONFIG_FILE" || grep -q "sk-your-api-key-here" "$CONFIG_FILE"; then
  echo "ERROR: config.toml still contains a placeholder API key."
  echo "Please set a real API key in $CONFIG_FILE."
  exit 1
fi

# If running SWE-bench Verified, auto-generate verified_instances.txt if missing
if [[ "$DATASET" == "princeton-nlp/SWE-bench_Verified" ]]; then
  if [ -f "selected_500_instances.csv" ] && [ ! -f "verified_instances.txt" ]; then
    echo "Generating verified_instances.txt from selected_500_instances.csv..."
    grep "^Verified" selected_500_instances.csv | cut -d',' -f2 > verified_instances.txt
    echo "Generated verified_instances.txt with $(wc -l < verified_instances.txt) lines."
    echo ""
  fi
fi

# If an explicit instance file is provided for SWE-bench Verified, write selected_ids
if [ -n "$INSTANCE_FILE" ]; then
  if [[ "$DATASET" != "princeton-nlp/SWE-bench_Verified" ]]; then
    echo "WARNING: --instance-file is currently only supported for SWE-bench Verified."
    echo "         The option will be ignored for dataset: $DATASET"
  else
    # Resolve to absolute path if needed
    if [[ "$INSTANCE_FILE" != /* ]]; then
      INSTANCE_FILE="$RUN_DIR/$INSTANCE_FILE"
    fi

    if [ ! -f "$INSTANCE_FILE" ]; then
      echo "ERROR: instance file not found: $INSTANCE_FILE"
      exit 1
    fi

    echo "Using instance file: $INSTANCE_FILE"

    python3 - "$INSTANCE_FILE" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

instance_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
cfg_file = run_dir / "evaluation/benchmarks/swe_bench/config.toml"

if not instance_path.is_file():
    raise SystemExit(f"Instance file not found: {instance_path}")

lines = [ln.strip() for ln in instance_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
ids_unique = list(dict.fromkeys(lines))  # preserve order, de-duplicate

if not ids_unique:
    raise SystemExit(f"No non-empty instance_ids found in {instance_path}")

out_lines = []
out_lines.append("# Auto-generated by run_verified.sh from instance file.")
out_lines.append("selected_ids = [")
for x in ids_unique:
    out_lines.append(f"    '{x}',")
out_lines.append("]")

cfg_file.parent.mkdir(parents=True, exist_ok=True)
cfg_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print(f"Wrote {len(ids_unique)} selected_ids to {cfg_file}")
PY

    # When instance file is used, LIMIT is derived from it and overrides presets
    LIMIT=$(wc -l < "$INSTANCE_FILE" | tr -d ' ')
    echo "Instance count derived from file: $LIMIT"
  fi
fi

##############################################################################
# Determine instance count from PRESET / LIMIT
##############################################################################

if [ -n "$PRESET" ]; then
  case "$PRESET" in
    quick)
      LIMIT=${LIMIT:-10} ;;
    medium)
      LIMIT=${LIMIT:-50} ;;
    verified)
      LIMIT=${LIMIT:-174} ;;
    full)
      LIMIT=${LIMIT:-500} ;;
    *)
      echo "ERROR: unknown preset: $PRESET (expected one of quick|medium|verified|full)"
      exit 1 ;;
  esac
fi

if [ -z "${LIMIT:-}" ]; then
  # Default behavior: 174 verified instances
  LIMIT=174
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [ "$LIMIT" -le 0 ]; then
  echo "ERROR: invalid instance count: $LIMIT"
  exit 1
fi

##############################################################################
# Show configuration and confirm
##############################################################################

echo "====================================="
echo "SWE-bench Verified run configuration (OpenHands)"
echo "====================================="
echo "Run directory : $RUN_DIR"
echo "Dataset       : $DATASET"
echo "Model         : $MODEL"
echo "Agent         : $AGENT"
echo "Instances     : $LIMIT"
echo "Max iterations: $MAX_ITERS"
echo "Concurrency   : $CONCURRENCY"
echo "Config file   : $CONFIG_FILE"
echo "run_infer     : $RUN_INFER_SCRIPT"
echo "====================================="
echo ""

if [ "$NO_CONFIRM" = false ]; then
  read -p "Start run? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Run cancelled."
    exit 0
  fi
fi

##############################################################################
# Environment variables and command construction
##############################################################################

export ITERATIVE_EVAL_MODE=true
export EVAL_CONDENSER="${EVAL_CONDENSER:-summarizer_for_eval}"

CMD=(
  "$RUN_INFER_SCRIPT"
  "$MODEL"
  "HEAD"
  "$AGENT"
  "$LIMIT"
  "$MAX_ITERS"
  "$CONCURRENCY"
  "$DATASET"
  "test"
)

##############################################################################
# Run (foreground only)
##############################################################################

echo "Starting run..."
echo ""
"${CMD[@]}"
echo ""
echo "====================================="
echo "Run finished."
echo "====================================="
echo ""
echo "You can evaluate results with: ./evaluate_results.sh"

