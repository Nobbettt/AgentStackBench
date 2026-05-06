#!/bin/bash
set -e

echo "========================================="
echo "SWE-bench Verified result evaluation"
echo "========================================="
echo ""

OUTPUT_DIR="evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: output directory not found."
    echo "Please run an evaluation job first."
    exit 1
fi

echo "Searching for output.jsonl files..."
echo ""

FOUND_FILES=$(find "$OUTPUT_DIR" -name "output.jsonl" 2>/dev/null || true)

if [ -z "$FOUND_FILES" ]; then
    echo "ERROR: no output.jsonl files found."
    echo "Please run an evaluation job and wait for it to finish."
    exit 1
fi

echo "Select an output file to evaluate:"
echo ""
select OUTPUT_FILE in $FOUND_FILES "Exit"; do
    if [ "$OUTPUT_FILE" = "Exit" ]; then
        echo "Evaluation cancelled."
        exit 0
    fi

    if [ -n "$OUTPUT_FILE" ]; then
        echo ""
        echo "Selected file: $OUTPUT_FILE"
        echo ""

        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")

        echo "File info:"
        echo "  Size     : $FILE_SIZE"
        echo "  Instances: $LINE_COUNT"
        echo ""

        echo "Starting evaluation..."
        echo ""

        ./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh "$OUTPUT_FILE"

        echo ""
        echo "========================================="
        echo "Evaluation finished."
        echo "========================================="
        echo ""

        REPORT_DIR=$(dirname "$OUTPUT_FILE")

        if [ -f "$REPORT_DIR/report.json" ]; then
            echo "Evaluation report (report.json):"
            echo ""
            cat "$REPORT_DIR/report.json" | python3 -m json.tool
            echo ""
        fi

        if [ -f "$REPORT_DIR/README.md" ]; then
            echo "Detailed report path:"
            echo "  $REPORT_DIR/README.md"
            echo ""
            echo "You can inspect it with:"
            echo "  cat $REPORT_DIR/README.md"
            echo "  or"
            echo "  less $REPORT_DIR/README.md"
        fi

        break
    else
        echo "Invalid choice, please try again."
    fi
done

