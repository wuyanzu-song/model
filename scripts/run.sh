#!/bin/bash

# Transformer from Scratch - Training Script
# Usage: bash scripts/run.sh

echo "Setting up environment..."

# Create conda environment
conda create -n transformer python=3.10 -y
conda activate transformer

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p checkpoints results

echo "Starting training..."
python src/train.py \
    --config configs/base.yaml \
    --seed 42

echo "Training completed!"
echo "Checkpoints saved in: checkpoints/"
echo "Training plot saved as: training_loss.png"