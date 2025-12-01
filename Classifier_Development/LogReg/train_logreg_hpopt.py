#!/usr/bin/env python3
"""
Logistic Regression with Hyperparameter Optimization for Refusal Classification

Uses Optuna for Bayesian hyperparameter optimization of TF-IDF + Logistic Regression.
Optimizes F1 score (refusal class) using 5-fold stratified cross-validation.
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    make_scorer,
)

# Optimization settings
N_TRIALS = 50
N_CV_FOLDS = 3
RANDOM_STATE = 42
OPTIMIZATION_METRIC = 'f1_refusal'  # F1 score for refusal class

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed_data"
TRAIN_FILE = DATA_DIR / "full_training_data.jsonl"
TEST_FILE = DATA_DIR / "full_test_data.jsonl"

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_jsonl(filepath: Path) -> pd.DataFrame:
    """Load JSONL file into pandas DataFrame"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def prepare_data(df: pd.DataFrame, text_field: str = 'response') -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from dataframe

    Args:
        df: DataFrame with 'response' and 'label' columns
        text_field: Which field to use for classification

    Returns:
        X: Array of text samples
        y: Array of labels
    """
    X = df[text_field].values
    y = df['label'].values
    return X, y


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    """
    Sample hyperparameters from the search space using Optuna trial.

    Returns:
        dict: Sampled hyperparameters for TF-IDF and Logistic Regression
    """
    # TF-IDF hyperparameters
    max_features = trial.suggest_categorical('tfidf_max_features', [5000, 10000])
    ngram_range_str = trial.suggest_categorical('tfidf_ngram_range', ['1,1', '1,2', '1,3', '2,2'])
    min_df = trial.suggest_int('tfidf_min_df', 1, 5)
    max_df = trial.suggest_float('tfidf_max_df', 0.85, 1.0)
    sublinear_tf = trial.suggest_categorical('tfidf_sublinear_tf', [True, False])
    stop_words = trial.suggest_categorical('tfidf_stop_words', ['english', 'none'])

    # Logistic Regression hyperparameters
    C = trial.suggest_float('logreg_C', 1e-4, 100.0, log=True)
    penalty = trial.suggest_categorical('logreg_penalty', ['l1', 'l2'])
    class_weight = trial.suggest_categorical('logreg_class_weight', ['balanced', 'none'])

    # Solver selection based on penalty
    if penalty == 'l1':
        solver = 'liblinear'
    else:  # l2
        solver = 'lbfgs'

    # Parse ngram_range
    ngram_range = tuple(map(int, ngram_range_str.split(',')))

    # Handle 'none' strings
    stop_words_val = None if stop_words == 'none' else stop_words
    class_weight_val = None if class_weight == 'none' else class_weight

    return {
        'tfidf': {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'min_df': min_df,
            'max_df': max_df,
            'sublinear_tf': sublinear_tf,
            'stop_words': stop_words_val,
            'strip_accents': 'unicode',
            'lowercase': True,
        },
        'logreg': {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'class_weight': class_weight_val,
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
    }


def build_pipeline_from_params(params: dict) -> Pipeline:
    """
    Build sklearn Pipeline from hyperparameter dict.

    Args:
        params: Dict with 'tfidf' and 'logreg' sub-dicts

    Returns:
        Configured sklearn Pipeline
    """
    tfidf_params = params['tfidf']
    logreg_params = {k: v for k, v in params['logreg'].items() if v is not None}

    return Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('classifier', LogisticRegression(**logreg_params))
    ])


def create_objective(X_train: np.ndarray, y_train: np.ndarray):
    """
    Create Optuna objective function with training data in closure.

    Args:
        X_train: Training text samples
        y_train: Training labels

    Returns:
        Callable objective function for Optuna
    """
    # Pre-create CV splitter for consistency across trials
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Custom scorer for F1 on refusal class
    f1_refusal_scorer = make_scorer(f1_score, pos_label='refusal')

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Returns:
            float: Mean cross-validation F1 score (refusal class)
        """
        # Sample hyperparameters
        params = sample_hyperparameters(trial)

        # Build pipeline with sampled params
        pipeline = build_pipeline_from_params(params)

        # Run cross-validation
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv,
                scoring=f1_refusal_scorer,
                n_jobs=1
            )

            # Store additional info for analysis
            trial.set_user_attr('cv_std', float(cv_scores.std()))
            trial.set_user_attr('cv_scores', cv_scores.tolist())

            return cv_scores.mean()

        except Exception as e:
            trial.set_user_attr('error', str(e))
            return 0.0

    return objective


def run_optimization(X_train: np.ndarray, y_train: np.ndarray) -> tuple[optuna.Study, Path]:
    """
    Run Optuna hyperparameter optimization.

    Args:
        X_train: Training text samples
        y_train: Training labels

    Returns:
        Tuple of (completed study, results directory path)
    """
    # Create study with SQLite storage (allows resumption if interrupted)
    storage_path = RESULTS_DIR / 'study.db'

    study = optuna.create_study(
        study_name='logreg_hpopt',
        direction='maximize',  # Maximize F1
        storage=f'sqlite:///{storage_path}',
        load_if_exists=True,  # Resume if interrupted
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    # Create objective function
    objective = create_objective(X_train, y_train)

    # Print optimization settings
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION SETTINGS")
    print("=" * 80)
    print(f"Metric:         F1 Score (refusal class)")
    print(f"Trials:         {N_TRIALS}")
    print(f"CV Folds:       {N_CV_FOLDS}-fold stratified")
    print(f"Optimizer:      Optuna (TPE sampler)")
    print(f"Random State:   {RANDOM_STATE}")
    print(f"Storage:        {storage_path}")
    print("-" * 80)

    # Run optimization
    print(f"\nStarting optimization ({N_TRIALS} trials)")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    return study, RESULTS_DIR


def evaluate_final_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate trained model on test set.

    Args:
        pipeline: Trained sklearn Pipeline
        X_test: Test text samples
        y_test: Test labels

    Returns:
        dict: Complete evaluation metrics
    """
    # Predictions
    start = time.time()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    inference_time = (time.time() - start) / len(X_test) * 1000

    # Get refusal class probability
    classes = pipeline.classes_
    refusal_idx = np.where(classes == 'refusal')[0][0]
    y_score = y_proba[:, refusal_idx]

    # Binary labels for ROC-AUC
    y_test_binary = (y_test == 'refusal').astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_refusal = f1_score(y_test, y_pred, pos_label='refusal')
    roc_auc = roc_auc_score(y_test_binary, y_score)

    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, labels=['refusal', 'compliance']
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['refusal', 'compliance'])

    return {
        'accuracy': float(accuracy),
        'f1_refusal': float(f1_refusal),
        'roc_auc': float(roc_auc),
        'precision': {
            'refusal': float(precision[0]),
            'compliance': float(precision[1])
        },
        'recall': {
            'refusal': float(recall[0]),
            'compliance': float(recall[1])
        },
        'f1_per_class': {
            'refusal': float(f1_per_class[0]),
            'compliance': float(f1_per_class[1])
        },
        'support': {
            'refusal': int(support[0]),
            'compliance': int(support[1])
        },
        'confusion_matrix': cm.tolist(),
        'inference_time_ms_per_sample': float(inference_time)
    }


def save_results(
    study: optuna.Study,
    results_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimization_time: float
) -> dict:
    """
    Save all optimization results and train final model.

    Args:
        study: Completed Optuna study
        results_dir: Directory to save results
        X_train, y_train: Training data
        X_test, y_test: Test data
        optimization_time: Time taken for optimization

    Returns:
        dict: Final model metrics
    """
    # 1. Extract best parameters
    best_trial = study.best_trial
    best_params_raw = best_trial.params

    # Reconstruct full params dict
    best_params = sample_hyperparameters_from_dict(best_params_raw)

    # Save best params
    best_params_path = results_dir / 'best_params.json'
    with open(best_params_path, 'w') as f:
        # Convert to JSON-serializable format
        json_params = {
            'tfidf': {k: v if not isinstance(v, tuple) else list(v)
                      for k, v in best_params['tfidf'].items()},
            'logreg': {k: v for k, v in best_params['logreg'].items() if v is not None}
        }
        json.dump(json_params, f, indent=2)
    print(f"Best parameters saved to {best_params_path}")

    # 2. Generate Optuna visualizations
    try:
        fig_history = plot_optimization_history(study)
        fig_history.write_image(str(results_dir / 'optimization_history.png'), scale=2)
        print("  - optimization_history.png")
    except Exception as e:
        print(f"  - optimization_history.png (skipped: {e})")

    try:
        fig_importance = plot_param_importances(study)
        fig_importance.write_image(str(results_dir / 'param_importances.png'), scale=2)
        print("  - param_importances.png")
    except Exception as e:
        print(f"  - param_importances.png (skipped: {e})")

    try:
        fig_parallel = plot_parallel_coordinate(study)
        fig_parallel.write_image(str(results_dir / 'parallel_coordinate.png'), scale=2)
        print("  - parallel_coordinate.png")
    except Exception as e:
        print(f"  - parallel_coordinate.png (skipped: {e})")

    try:
        fig_slice = plot_slice(study)
        fig_slice.write_image(str(results_dir / 'param_slice.png'), scale=2)
        print("  - param_slice.png")
    except Exception as e:
        print(f"  - param_slice.png (skipped: {e})")

    # 3. Train final model with best params on full training set
    final_pipeline = build_pipeline_from_params(best_params)

    train_start = time.time()
    final_pipeline.fit(X_train, y_train)
    final_training_time = time.time() - train_start

    # 4. Evaluate on test set
    metrics = evaluate_final_model(final_pipeline, X_test, y_test)

    # Add optimization metadata
    metrics['optimization'] = {
        'best_cv_f1': float(study.best_value),
        'best_cv_std': float(best_trial.user_attrs.get('cv_std', 0)),
        'n_trials': len(study.trials),
        'optimization_time_seconds': optimization_time,
        'final_training_time_seconds': final_training_time,
    }
    metrics['best_params'] = json_params
    metrics['model'] = 'Logistic Regression (TF-IDF) - Hyperparameter Optimized'

    # Save metrics
    metrics_path = results_dir / 'best_model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # 5. Save trained model
    model_path = results_dir / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_pipeline, f)
    print(f"Model saved to {model_path}")

    # 6. Save trials summary CSV
    trials_df = study.trials_dataframe()
    trials_path = results_dir / 'trials_summary.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"Trials summary saved to {trials_path}")

    # 7. Generate confusion matrix visualization
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, results_dir / 'confusion_matrix.png')

    # 8. Extract feature importances
    extract_feature_importances(final_pipeline, results_dir / 'feature_importances.txt')

    return metrics


def sample_hyperparameters_from_dict(params_dict: dict) -> dict:
    """
    Reconstruct full hyperparameters dict from Optuna trial params.

    Args:
        params_dict: Flat dict of trial parameters

    Returns:
        Nested dict with 'tfidf' and 'logreg' sub-dicts
    """
    # Parse ngram_range
    ngram_str = params_dict['tfidf_ngram_range']
    ngram_range = tuple(map(int, ngram_str.split(',')))

    # Handle 'none' strings
    stop_words = params_dict['tfidf_stop_words']
    stop_words_val = None if stop_words == 'none' else stop_words

    class_weight = params_dict['logreg_class_weight']
    class_weight_val = None if class_weight == 'none' else class_weight

    # Determine solver based on penalty
    penalty = params_dict['logreg_penalty']
    if penalty == 'l1':
        solver = 'liblinear'
    else:  # l2
        solver = 'lbfgs'

    return {
        'tfidf': {
            'max_features': params_dict['tfidf_max_features'],
            'ngram_range': ngram_range,
            'min_df': params_dict['tfidf_min_df'],
            'max_df': params_dict['tfidf_max_df'],
            'sublinear_tf': params_dict['tfidf_sublinear_tf'],
            'stop_words': stop_words_val,
            'strip_accents': 'unicode',
            'lowercase': True,
        },
        'logreg': {
            'C': params_dict['logreg_C'],
            'penalty': penalty,
            'solver': solver,
            'class_weight': class_weight_val,
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Refusal', 'Compliance'],
                yticklabels=['Refusal', 'Compliance'],
                cbar_kws={'label': 'Count'})
    plt.title('Optimized Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def extract_feature_importances(pipeline: Pipeline, output_path: Path, top_n: int = 20) -> None:
    """Extract and save most important TF-IDF features per class"""
    # Get feature names and coefficients
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coefficients = pipeline.named_steps['classifier'].coef_[0]

    # Sort features by coefficient
    feature_importance = list(zip(feature_names, coefficients))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Top features for refusal (positive coefficients)
    top_refusal = feature_importance[:top_n]

    # Top features for compliance (negative coefficients)
    top_compliance = feature_importance[-top_n:][::-1]

    # Save to file
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE IMPORTANCES (Optimized TF-IDF + Logistic Regression)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"TOP {top_n} FEATURES FOR REFUSAL CLASS:\n")
        f.write("-" * 80 + "\n")
        for rank, (word, coef) in enumerate(top_refusal, 1):
            f.write(f"{rank:2}. {word:<30} {coef:>10.4f}\n")

        f.write(f"\nTOP {top_n} FEATURES FOR COMPLIANCE CLASS:\n")
        f.write("-" * 80 + "\n")
        for rank, (word, coef) in enumerate(top_compliance, 1):
            f.write(f"{rank:2}. {word:<30} {coef:>10.4f}\n")

    print(f"  - feature_importances.txt")


def main():
    """Main hyperparameter optimization pipeline"""

    # Load data
    train_df = load_jsonl(TRAIN_FILE)
    test_df = load_jsonl(TEST_FILE)

    # Prepare features and labels
    X_train, y_train = prepare_data(train_df, text_field='response')
    X_test, y_test = prepare_data(test_df, text_field='response')


    # Print label distribution
    print(f"\nLabel distribution (training):")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count:,} ({count/len(y_train)*100:.1f}%)")

    # Run optimization
    start_time = time.time()
    study, results_dir = run_optimization(X_train, y_train)
    optimization_time = time.time() - start_time

    # Print optimization summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal optimization time: {optimization_time:.1f}s ({optimization_time/60:.1f} min)")
    print(f"Best CV F1 (refusal): {study.best_value:.4f}")
    print(f"Best trial: #{study.best_trial.number}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results and train final model
    metrics = save_results(
        study, results_dir,
        X_train, y_train,
        X_test, y_test,
        optimization_time
    )

    # Print final results
    print("\n" + "=" * 80)
    print("FINAL TEST SET RESULTS")
    print("=" * 80)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 (ref):  {metrics['f1_refusal']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 55)
    for cls in ['refusal', 'compliance']:
        print(f"{cls:<15} {metrics['precision'][cls]:<12.4f} {metrics['recall'][cls]:<12.4f} {metrics['f1_per_class'][cls]:<12.4f}")

    cm = np.array(metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Refusal  Compliance")
    print(f"Actual Refusal     {cm[0,0]:>4}      {cm[0,1]:>4}")
    print(f"    Compliance     {cm[1,0]:>4}      {cm[1,1]:>4}")

if __name__ == "__main__":
    main()
