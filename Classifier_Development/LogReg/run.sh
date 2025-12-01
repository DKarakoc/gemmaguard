#!/bin/bash
#
# Execution script for Logistic Regression Hyperparameter Optimization
#
# Uses Optuna for Bayesian optimization of TF-IDF + LogReg hyperparameters.
# Optimizes F1 score (refusal class) using 3-fold stratified CV.
#
# Usage: ./run.sh
#

set -e  # Exit on error

# Check if running from correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "train_logreg_hpopt.py" ]; then
    echo "Error: train_logreg_hpopt.py not found"
    exit 1
fi

# Create results directory
mkdir -p results

# Check if data exists
DATA_DIR="data"
if [ ! -f "$DATA_DIR/full_training_data.jsonl" ]; then
    echo "Error: Training data not found at $DATA_DIR/full_training_data.jsonl"
    exit 1
fi

# Run optimization
echo "Starting hyperparameter optimization"
echo ""
uv run train_logreg_hpopt.py

echo "Results saved to results/:"
echo "  - best_params.json          Best hyperparameters"
echo "  - best_model_metrics.json   Full evaluation metrics"
echo "  - best_model.pkl            Trained model"
echo "  - trials_summary.csv        All trial results"
echo "  - optimization_history.png  Convergence plot"
echo "  - param_importances.png     Parameter importance"
echo "  - parallel_coordinate.png   Parameter relationships"
echo "  - confusion_matrix.png      Classification errors"
echo "  - feature_importances.txt   Top features per class"
echo "  - study.db                  Optuna database"
