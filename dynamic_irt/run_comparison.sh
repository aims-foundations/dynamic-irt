#!/bin/bash
# Script to train CIRT and GPIRT models and compare their performance

set -e  # Exit on error

# Configuration
COURSE_NAME="dsa_hk231"
SEED=42
CIRT_CONCENTRATION=10.0
CIRT_EPOCHS=10000
GPIRT_KERNEL="RBF"
GPIRT_LENGTH_SCALE=1.0
GPIRT_MAX_EPOCH=1000
GPIRT_WARMUP=200

echo "=========================================="
echo "CIRT vs GPIRT Model Comparison Pipeline"
echo "=========================================="
echo "Course: $COURSE_NAME"
echo "Seed: $SEED"
echo ""

# Change to the dynamic_irt directory
cd "$(dirname "$0")"

# Step 1: Train CIRT model
echo "[1/3] Training CIRT model..."
echo "   Concentration: $CIRT_CONCENTRATION"
echo "   Epochs: $CIRT_EPOCHS"
echo ""

if [ -f "results/${COURSE_NAME}_${CIRT_CONCENTRATION}/model.pkl" ]; then
    echo "   ✓ CIRT model already exists. Skipping training."
    echo "   (Delete results/${COURSE_NAME}_${CIRT_CONCENTRATION}/ to retrain)"
else
    python cirt.py \
        --course_name "$COURSE_NAME" \
        --seed "$SEED" \
        --concentration "$CIRT_CONCENTRATION" \
        --epochs "$CIRT_EPOCHS"
    echo "   ✓ CIRT training complete"
fi
echo ""

# Step 2: Train GPIRT model
echo "[2/3] Training GPIRT model..."
echo "   Kernel: $GPIRT_KERNEL"
echo "   Length scale: $GPIRT_LENGTH_SCALE"
echo "   Max epochs: $GPIRT_MAX_EPOCH"
echo "   Warmup: $GPIRT_WARMUP"
echo ""

GPIRT_FOLDER="results/${COURSE_NAME}_s${SEED}_D1_PL1_hmc_kernel${GPIRT_KERNEL}_ls${GPIRT_LENGTH_SCALE}"

if [ -f "${GPIRT_FOLDER}/ability.pt" ] && [ -f "${GPIRT_FOLDER}/difficulty.pt" ]; then
    echo "   ✓ GPIRT model already exists. Skipping training."
    echo "   (Delete ${GPIRT_FOLDER}/ to retrain)"
else
    python gpirt/inference.py \
        --course_name "$COURSE_NAME" \
        --seed "$SEED" \
        --kernel "$GPIRT_KERNEL" \
        --length_scale "$GPIRT_LENGTH_SCALE" \
        --fitting_method "hmc" \
        --max_epoch "$GPIRT_MAX_EPOCH" \
        --warmup_steps "$GPIRT_WARMUP"
    echo "   ✓ GPIRT training complete"
fi
echo ""

# Step 3: Run comparison
echo "[3/3] Running comparison analysis..."
echo ""

python compare_cirt_gpirt.py \
    --course_name "$COURSE_NAME" \
    --seed "$SEED" \
    --cirt_concentration "$CIRT_CONCENTRATION" \
    --gpirt_kernel "$GPIRT_KERNEL" \
    --gpirt_length_scale "$GPIRT_LENGTH_SCALE" \
    --output_folder "comparison_results"

echo ""
echo "=========================================="
echo "✓ Comparison complete!"
echo "=========================================="
echo "Results saved in comparison_results/${COURSE_NAME}_*/"
echo ""
