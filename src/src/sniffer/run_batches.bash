#!/bin/bash

if [ -z "$1" ]; then
	echo "Usage: ./run_batches.bash <substance_name> [target_rows]"
	echo "Example: ./run_batches.bash tea 60"
	exit 1
fi

SUBSTANCE=$1
TARGET_ROWS=${2:-60}
DATA_DIR="/opt/projects/bme688/DATA"
CSV_FILE="${DATA_DIR}/dataset_${SUBSTANCE}_master.csv"

echo "=================================================="
echo " Self-Healing Collector for: $SUBSTANCE"
echo " Target Data Rows: $TARGET_ROWS"
echo "=================================================="

while true; do
	if [ -f "$CSV_FILE" ]; then
		TOTAL_LINES=$(wc -l <"$CSV_FILE")
		CURRENT_ROWS=$((TOTAL_LINES > 0 ? TOTAL_LINES - 1 : 0))
	else
		CURRENT_ROWS=0
	fi

	if [ "$CURRENT_ROWS" -ge "$TARGET_ROWS" ]; then
		echo -e "\nDone! $CSV_FILE now has $CURRENT_ROWS data rows."
		rm -f core
		break
	fi

	echo -e "\n[$SUBSTANCE] Progress: $CURRENT_ROWS / $TARGET_ROWS rows. Starting batch..."
	python -m sniffer collect "$SUBSTANCE"
	sleep 2
done
