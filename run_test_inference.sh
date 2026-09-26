#!/bin/bash
set -e

echo "=========================================================="
echo "Amazon ML Challenge 2026 - Production Test Inference"
echo "=========================================================="

# Activate the virtual environment
source .venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=.

# Run the inference script
python3 src/run_inference.py \
    --s1 dataset/test/test_source1.tsv \
    --s2 dataset/test/test_source2.tsv \
    --s3 dataset/test/test_source3.tsv \
    --index_dir experiments/final_hardening/test_index \
    --out submission.csv \
    --model experiments/final_hardening/artifacts/xgb_15000.ubj \
    --threshold 0.98

echo "Inference pipeline finished successfully."
echo "Your submission is ready at: submission.csv"
