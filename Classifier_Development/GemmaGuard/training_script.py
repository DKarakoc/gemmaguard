"""
Gemma-3-270M-IT Refusal Classifier Training Script
"""

import sys
import os

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['DATASETS_DISABLE_PROGRESS_BARS'] = '0'

import logging
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import json
import subprocess
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Setup proper logging
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILENAME = f"gemma3_270m_it_training_log_{RUN_TIMESTAMP}.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w'),
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Capture Python warnings in log file
logging.captureWarnings(True)

# Redirect tqdm to logger
class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            line = line.strip()
            if line and any(c.isalnum() for c in line):
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

tqdm_logger = TqdmToLogger(logger)

# GPU Memory Logging Helper
def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Custom Training Progress Callback
class TrainingProgressCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=50, total_steps=None):
        self.log_every_n_steps = log_every_n_steps
        self.total_steps = total_steps
        self.training_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = datetime.now()
        logger.info(f"Training progress will be logged every {self.log_every_n_steps} steps")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % self.log_every_n_steps == 0:
            current_time = datetime.now()

            # Calculate ETA
            eta_str = ""
            if self.total_steps and state.global_step > 0:
                elapsed = (current_time - self.training_start_time).total_seconds()
                steps_remaining = self.total_steps - state.global_step
                seconds_per_step = elapsed / state.global_step
                eta_seconds = steps_remaining * seconds_per_step
                eta = timedelta(seconds=int(eta_seconds))
                eta_str = f", ETA: {eta}"

            loss = logs.get('loss', 'N/A')
            lr = logs.get('learning_rate', 'N/A')

            loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
            lr_str = f"{lr:.2e}" if lr != 'N/A' else 'N/A'

            logger.info(
                f"Step {state.global_step}/{self.total_steps or '?'}: "
                f"loss={loss_str}, lr={lr_str}{eta_str}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {int(state.epoch)} complete (step {state.global_step})")
        log_gpu_memory(prefix="  ")

# Enhanced Early Stopping Callback
class VerboseEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience, early_stopping_threshold=0.0):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.best_metric_value = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get(args.metric_for_best_model)

        if metric_value is None:
            return

        logger.info(f"  Eval metric ({args.metric_for_best_model}): {metric_value:.4f}")

        if self.best_metric_value is None:
            self.best_metric_value = metric_value
            logger.info(f"  - Initial best metric: {metric_value:.4f}")
            self.patience_counter = 0
        else:
            if args.greater_is_better:
                improved = metric_value > (self.best_metric_value + self.early_stopping_threshold)
            else:
                improved = metric_value < (self.best_metric_value - self.early_stopping_threshold)

            if improved:
                improvement = abs(metric_value - self.best_metric_value)
                self.best_metric_value = metric_value
                self.patience_counter = 0
                logger.info(f"  - Improved by {improvement:.4f}! New best: {metric_value:.4f} (patience reset)")
            else:
                self.patience_counter += 1
                patience_remaining = self.early_stopping_patience - self.patience_counter
                logger.info(
                    f"  - No improvement (best: {self.best_metric_value:.4f}). "
                    f"Patience: {self.patience_counter}/{self.early_stopping_patience} "
                    f"({patience_remaining} evaluations remaining)"
                )

                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("")
                    logger.info("="*80)
                    logger.info("EARLY STOPPING TRIGGERED")
                    logger.info(f"  No improvement for {self.early_stopping_patience} consecutive evaluations")
                    logger.info(f"  Best {args.metric_for_best_model}: {self.best_metric_value:.4f}")
                    logger.info(f"  Stopping at step {state.global_step}")
                    logger.info("="*80)
                    logger.info("")

        return super().on_evaluate(args, state, control, metrics, **kwargs)


logger.info("="*80)
logger.info("GEMMA-3-270M-IT REFUSAL CLASSIFIER - SINGLE GPU TRAINING")
logger.info("="*80)
logger.info(f"Start time: {datetime.now()}")
logger.info(f"Run ID: {RUN_TIMESTAMP}")
logger.info(f"Log file: {LOG_FILENAME}")
logger.info("")

# GPU Detection
logger.info(f"GPU Configuration:")
logger.info(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
else:
    logger.warning("  NO GPU detected! Training will be very slow.")
logger.info("")

# Authenticate with Hugging Face
logger.info("Authenticating with Hugging Face...")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not set")
    raise ValueError("HF_TOKEN environment variable not set")

from huggingface_hub import login
login(token=HF_TOKEN)
logger.info("Authenticated with Hugging Face")
logger.info("")

# Auto-detect project
logger.info("Detecting GCP project")
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
if not PROJECT_ID:
    result = subprocess.run(['gcloud', 'config', 'get-value', 'project'],
                           capture_output=True, text=True)
    PROJECT_ID = result.stdout.strip()
BUCKET_NAME = f"{PROJECT_ID}-gemma-quickstart"

logger.info(f"Project ID: {PROJECT_ID}")
logger.info(f"Bucket: gs://{BUCKET_NAME}")
logger.info("")

# Configuration - GEMMA-3-270M-IT optimized for single L4 GPU
MODEL_NAME = "gemma3-270m-it"
MODEL_ID = "google/gemma-3-270m-it"

NUM_EPOCHS = 3
# Single GPU: effective batch = 2 × 4 = 8 (matching 270M base exactly)
# Using batch=2, grad_accum=4 (instead of 4/2) for better generalization
BATCH_SIZE = 2
GRADIENT_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
LORA_RANK = 16
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.03

# Early stopping configuration
EVAL_STEPS = 2000
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

GCS_DATA_DIR = f"gs://{BUCKET_NAME}/data"
LOCAL_OUTPUT_DIR = "/tmp/gemma3-270m-it-training"

effective_batch = BATCH_SIZE * GRADIENT_ACCUM_STEPS

logger.info("Configuration:")
logger.info(f"  Model: {MODEL_ID}")
logger.info(f"  Context window: 32k (using {MAX_SEQ_LENGTH})")
logger.info(f"  Epochs: {NUM_EPOCHS} (max, may stop early)")
logger.info(f"  Batch configuration:")
logger.info(f"    - Per-device batch size: {BATCH_SIZE}")
logger.info(f"    - Gradient accumulation: {GRADIENT_ACCUM_STEPS}")
logger.info(f"    - Effective batch size: {effective_batch}")
logger.info(f"  Learning rate: {LEARNING_RATE}")
logger.info(f"  LoRA rank: {LORA_RANK}")
logger.info(f"  Max sequence length: {MAX_SEQ_LENGTH}")
logger.info(f"  Warmup ratio: {WARMUP_RATIO}")
logger.info(f"  Evaluation frequency: every {EVAL_STEPS} steps")
logger.info(f"  Early stopping: patience={EARLY_STOPPING_PATIENCE}, threshold={EARLY_STOPPING_THRESHOLD}")
logger.info("")

# Load datasets
logger.info("Loading datasets from GCS")
train_dataset = load_dataset('json', data_files=f"{GCS_DATA_DIR}/full/full_training_data_yesno.jsonl", split='train')
val_dataset = load_dataset('json', data_files=f"{GCS_DATA_DIR}/full/full_validation_data_yesno.jsonl", split='train')
test_dataset = load_dataset('json', data_files=f"{GCS_DATA_DIR}/full/full_test_data_yesno.jsonl", split='train')

logger.info(f" Loaded datasets:")
logger.info(f"  Training: {len(train_dataset):,} samples")
logger.info(f"  Validation: {len(val_dataset):,} samples")
logger.info(f"  Test: {len(test_dataset):,} samples")
logger.info(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,} samples")
logger.info("")


def train_model(model_name, model_variant):
    """Train Gemma-3-270M-IT model with yes/no token classification"""
    logger.info("")
    logger.info("="*80)
    logger.info(f"TRAINING: {model_name.upper()} MODEL ({model_variant})")
    logger.info("="*80)

    start_time = datetime.now()
    logger.info(f"Training start time: {start_time}")

    # Load model
    logger.info("")
    logger.info(f"[1/7] Loading {model_variant}")
    tokenizer = AutoTokenizer.from_pretrained(model_variant)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LEFT truncation to preserve classification tokens
    tokenizer.truncation_side = 'left'

    # Use pretrained yes/no tokens from vocabulary
    logger.info("")
    logger.info("  Using pretrained yes/no tokens from vocabulary")
    # IMPORTANT: Use " yes" and " no" with leading space to match training data format
    # Training data: "Is this a refusal? yes" where space is part of token
    yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    logger.info(f"  Token IDs: ' yes'={yes_token_id}, ' no'={no_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_variant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    log_gpu_memory(prefix="  ")

    # Configure LoRA
    logger.info("")
    logger.info(f"[2/7] Configuring LoRA (rank={LORA_RANK})")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Enable gradients for embeddings
    logger.info("  Enabling gradients for input embeddings")
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True

    logger.info("  Enabling gradients for output layer")
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA configured: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Input embedding layer is trainable")
    logger.info(f"Output layer is trainable")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Tokenize
    logger.info("")
    logger.info(f"[3/7] Tokenizing datasets")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    logger.info(f"Tokenized {len(tokenized_train):,} train + {len(tokenized_val):,} val samples")

    # Setup training
    logger.info("")
    logger.info(f"[4/7] Setting up training")
    output_dir = f"{LOCAL_OUTPUT_DIR}/{model_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Calculate total training steps
    steps_per_epoch = len(tokenized_train) // (BATCH_SIZE * GRADIENT_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total training steps: {total_steps:,}")

    # Estimate training time (empirical from 270M base: ~1.7 sec/step)
    sec_per_step = 1.7
    estimated_time_min = total_steps * sec_per_step / 60
    estimated_time_hours = estimated_time_min / 60
    cost_per_hour = 0.70  # L4 @ $0.70/hr
    estimated_cost = estimated_time_hours * cost_per_hour

    logger.info(f"  Estimated time: ~{estimated_time_min:.0f} min ({estimated_time_hours:.1f} hours)")

    # Setup callbacks
    verbose_early_stopping = VerboseEarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )

    progress_callback = TrainingProgressCallback(
        log_every_n_steps=50,
        total_steps=total_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[verbose_early_stopping, progress_callback],
    )

    logger.info(f"Training setup complete")

    # Train
    logger.info("")
    logger.info(f"[5/7] Starting training ({NUM_EPOCHS} epochs, max {total_steps:,} steps)")
    logger.info("")
    log_gpu_memory(prefix="  ")
    logger.info("")

    train_result = trainer.train()

    logger.info("")
    logger.info(f"  Training complete!")
    logger.info(f"  Final train loss: {train_result.training_loss:.4f}")
    logger.info(f"  Training time: {train_result.metrics['train_runtime']/60:.1f} min")
    logger.info(f"  Samples per second: {train_result.metrics.get('train_samples_per_second', 'N/A')}")
    log_gpu_memory(prefix="  ")

    # Evaluate
    logger.info("")
    logger.info(f"[6/7] Evaluating on validation set")
    eval_result = trainer.evaluate()
    logger.info(f"Eval loss: {eval_result['eval_loss']:.4f}")

    # Save
    logger.info("")
    logger.info(f"[7/7] Saving model")
    local_model_path = f"{LOCAL_OUTPUT_DIR}/{model_name}_final"
    trainer.save_model(local_model_path)
    tokenizer.save_pretrained(local_model_path)

    # Upload to GCS
    gcs_path = f"gs://{BUCKET_NAME}/models/gemma3-270m-it/{RUN_TIMESTAMP}/{model_name}"
    subprocess.run(f"gcloud storage cp -r {local_model_path} {gcs_path}", shell=True, check=True)
    logger.info(f"Model saved to {gcs_path}")

    # Record results
    end_time = datetime.now()
    results = {
        "model_name": model_name,
        "model_id": model_variant,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result['eval_loss'],
        "training_time_seconds": train_result.metrics['train_runtime'],
        "total_time_seconds": (end_time - start_time).total_seconds(),
        "gcs_path": gcs_path,
        "effective_batch_size": effective_batch,
    }

    logger.info("")
    logger.info(f"✓ {model_name.upper()} training complete! ({(end_time - start_time).total_seconds()/60:.1f} min)")

    # Cleanup
    del model, trainer, tokenizer
    torch.cuda.empty_cache()

    return local_model_path, results, yes_token_id, no_token_id


def get_classification_probabilities(model, tokenizer, input_text, yes_token_id, no_token_id, max_seq_length, debug=False):
    """
    Direct logit extraction for probability-based classification.

    Uses yes/no token logits for classification.
    Returns calibrated probabilities for refusal (yes) vs compliance (no).
    """
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]

    # Extract yes/no logits (yes = refusal, no = compliance)
    yes_logit = next_token_logits[yes_token_id].item()
    no_logit = next_token_logits[no_token_id].item()

    classification_logits = torch.tensor(
        [yes_logit, no_logit],
        dtype=torch.float32
    )
    probs = torch.softmax(classification_logits, dim=0)

    refusal_prob = probs[0].item()  # yes = refusal
    compliance_prob = probs[1].item()  # no = compliance

    prediction = "refusal" if refusal_prob > compliance_prob else "compliance"
    confidence = max(refusal_prob, compliance_prob)

    if debug:
        logger.info(f"[DEBUG] Logits: yes={yes_logit:.4f}, no={no_logit:.4f}")
        logger.info(f"[DEBUG] Probabilities: P(refusal/yes)={refusal_prob:.4f}, P(compliance/no)={compliance_prob:.4f}")
        logger.info(f"[DEBUG] Prediction: {prediction} (confidence={confidence:.4f})")

    return {
        "refusal": refusal_prob,
        "compliance": compliance_prob,
        "prediction": prediction,
        "confidence": confidence
    }


def evaluate_model(model_name, model_variant, model_path, yes_token_id, no_token_id):
    """Evaluate model on test set with probability-based classification using yes/no tokens"""
    logger.info("")
    logger.info("="*80)
    logger.info(f"EVALUATING: {model_name.upper()} MODEL")
    logger.info("="*80)

    logger.info("")
    logger.info(f"Loading model")
    tokenizer = AutoTokenizer.from_pretrained(model_variant)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_variant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    logger.info(f"  Model loaded")
    log_gpu_memory(prefix="  ")

    logger.info(f"  Token IDs: yes={yes_token_id}, no={no_token_id}")

    logger.info("")
    logger.info(f"Evaluating on {len(test_dataset):,} test samples...")
    logger.info(f"Using DIRECT LOGIT EXTRACTION for probability-based classification")
    eval_start_time = datetime.now()
    predictions = []
    ground_truth = []
    probabilities = []

    logger.info("")
    logger.info("🔍 DEBUG MODE: Showing first 5 sample predictions")
    logger.info("")

    for idx, sample in enumerate(tqdm(test_dataset, desc="Testing", file=tqdm_logger, mininterval=30)):
        input_text = f"""Classify if this response is a refusal or compliance.

User Request: {sample['prompt']}

Assistant Response: {sample['response']}

Is this a refusal?"""

        debug_mode = (idx < 5)
        if debug_mode:
            logger.info(f"--- Sample {idx+1}/{len(test_dataset)} (Ground truth: {sample['label']}) ---")

        probs = get_classification_probabilities(
            model,
            tokenizer,
            input_text,
            yes_token_id,
            no_token_id,
            MAX_SEQ_LENGTH,
            debug=debug_mode
        )

        if debug_mode:
            logger.info("")

        predictions.append(probs['prediction'])
        ground_truth.append(sample['label'])
        probabilities.append(probs)

    eval_end_time = datetime.now()
    eval_duration = (eval_end_time - eval_start_time).total_seconds()

    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, pos_label="refusal", average="binary")
    samples_per_second = len(test_dataset) / eval_duration if eval_duration > 0 else 0

    y_true_binary = [1 if label == 'refusal' else 0 for label in ground_truth]
    y_scores = [p['refusal'] for p in probabilities]
    roc_auc = roc_auc_score(y_true_binary, y_scores)
    avg_confidence = np.mean([p['confidence'] for p in probabilities])

    logger.info("")
    logger.info(f"  Evaluation complete!")
    logger.info(f"  Evaluation time: {eval_duration/60:.1f} minutes ({eval_duration:.0f} seconds)")
    logger.info(f"  Samples/second: {samples_per_second:.2f}")
    logger.info("")
    logger.info(f"  PERFORMANCE METRICS:")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Average Confidence: {avg_confidence:.4f}")

    logger.info("")
    logger.info(f"Classification Report:")
    report = classification_report(ground_truth, predictions, target_names=["compliance", "refusal"])
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "avg_confidence": avg_confidence,
        "probabilities": probabilities,
        "predictions": predictions,
        "ground_truth": ground_truth
    }


def upload_logs_to_gcs():
    """Upload log file to GCS"""
    try:
        logger.info("")
        logger.info("Uploading logs to GCS")
        gcs_log_path = f"gs://{BUCKET_NAME}/models/gemma3-270m-it/{RUN_TIMESTAMP}/{LOG_FILENAME}"
        result = subprocess.run(
            f"gcloud storage cp {LOG_FILENAME} {gcs_log_path}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"  Logs uploaded to {gcs_log_path}")
        else:
            logger.error(f"Failed to upload logs: {result.stderr}")
    except Exception as e:
        logger.error(f"Exception during log upload: {e}")


# Main training loop
if __name__ == "__main__":
    try:
        logger.info("")
        logger.info("Starting Gemma-3-270M-IT YES/NO training pipeline...")
        model_path, training_results, yes_token_id, no_token_id = train_model(MODEL_NAME, MODEL_ID)

        # Evaluate model
        evaluation_results = evaluate_model(MODEL_NAME, MODEL_ID, model_path, yes_token_id, no_token_id)

        # Final results
        logger.info("")
        logger.info("="*80)
        logger.info("FINAL RESULTS - GEMMA-3-270M-IT")
        logger.info("="*80)

        logger.info("")
        logger.info(f"Model: {MODEL_ID}")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']*100:.2f}%")
        logger.info(f"  F1 Score: {evaluation_results['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        logger.info(f"  Avg Confidence: {evaluation_results['avg_confidence']:.4f}")
        logger.info(f"  Train Loss: {training_results['train_loss']:.4f}")
        logger.info(f"  Eval Loss: {training_results['eval_loss']:.4f}")
        logger.info(f"  Training Time: {training_results['training_time_seconds']/60:.1f} min")
        logger.info(f"  Effective Batch Size: {training_results['effective_batch_size']}")

        # Save results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'model': {
                'training': training_results,
                'evaluation': evaluation_results
            }
        }

        logger.info("")
        logger.info("Saving final results")
        results_file = f"{LOCAL_OUTPUT_DIR}/final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        subprocess.run(f"gcloud storage cp {results_file} gs://{BUCKET_NAME}/models/gemma3-270m-it/{RUN_TIMESTAMP}/", shell=True, check=True)

        logger.info(f"Results saved to GCS")
        logger.info("")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)

    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error(f"FATAL ERROR: {e}")
        logger.error("="*80)
        logger.exception("Full traceback:")
        raise

    finally:
        upload_logs_to_gcs()
