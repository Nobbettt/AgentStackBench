set -euo pipefail

# api_key.sh
# export OPENAI_API_KEY=
# export OPENAI_BASE_URL=
# export OPENAI_MODEL=
# export OPENAI_EMBED_URL=
source script/api_key.sh

export PYTHONPATH="$(pwd)"

: "${TARGET_ID:?TARGET_ID is required}"
: "${SWEBENCH_LANG:?SWEBENCH_LANG is required}"
: "${FOLDER_NAME:?FOLDER_NAME is required}"
: "${BENCH_NAME:?BENCH_NAME is required}"

export DATASET="${DATASET:-local_json}"
export SPLIT="${SPLIT:-test}"
export NJ="${NJ:-1}"
export NUM_SETS="${NUM_SETS:-1}"
export NUM_SAMPLES_PER_SET="${NUM_SAMPLES_PER_SET:-1}"

if [ -z "${DATA_ROOT:-}" ]; then
  export DATA_ROOT="$(pwd)/../data/${BENCH_NAME}"
fi

# Ensure a clean output directory for this instance.
OUTPUT_DIR="results/${FOLDER_NAME}"
OUTPUT_ABS="$(pwd)/${OUTPUT_DIR}"
INPUT_ABS="${OUTPUT_ABS}/input_data"
if [ -n "${DATA_ROOT:-}" ] && [ "${DATA_ROOT#${INPUT_ABS}}" != "${DATA_ROOT}" ]; then
  mkdir -p "${OUTPUT_DIR}"
  find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 ! -name "input_data" -exec rm -rf {} +
else
  rm -rf "${OUTPUT_DIR}"
fi

./script/localization1.1.sh
./script/localization1.2.sh
./script/localization1.3.sh
./script/localization1.4.sh
./script/localization2.1.sh
./script/localization3.1.sh
./script/localization3.2.sh

./script/repair.sh
./script/selection3.1.sh
