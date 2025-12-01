#!/bin/bash
# Gemma Classification on GCP

set -e

# Detect script location and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# 1. Check GCP authentication
if ! gcloud auth list 2>/dev/null | grep -q ACTIVE; then
    echo "ERROR: Not authenticated with GCP"
    echo ""
    echo "Run: gcloud auth login"
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: No GCP project configured"
    echo ""
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi


# 2. Check .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: .env file not found in $PROJECT_ROOT"
    echo ""
    echo "Create a .env file with your HF_TOKEN"
    exit 1
fi

# Check HF_TOKEN is set
if ! grep -q "^HF_TOKEN=" "$PROJECT_ROOT/.env"; then
    echo "ERROR: HF_TOKEN not found in .env file"
    exit 1
fi

# 3. Check service account key exists
SERVICE_ACCOUNT="gemma-guard-runner@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="$PROJECT_ROOT/customjob_vertexai/${PROJECT_ID}-key.json"

if [ ! -f "$KEY_FILE" ]; then
    # Try to find any key file
    KEY_FILE=$(find "$PROJECT_ROOT/customjob_vertexai" -maxdepth 1 -name "${PROJECT_ID}-*.json" -type f 2>/dev/null | head -n 1)

    if [ -z "$KEY_FILE" ]; then
        echo "ERROR: Service account key not found"
        echo ""
        echo "Expected: customjob_vertexai/${PROJECT_ID}-key.json"
        echo ""
        echo "Create with:"
        echo "  gcloud iam service-accounts keys create \\"
        echo "    customjob_vertexai/${PROJECT_ID}-key.json \\"
        echo "    --iam-account=${SERVICE_ACCOUNT}"
        exit 1
    fi
fi


# 4. Check WildGuard classified files exist
WILDGUARD_DIR="$PROJECT_ROOT/wildguard_classified"

if [ ! -d "$WILDGUARD_DIR" ]; then
    echo "ERROR: WildGuard classified directory not found"
    echo ""
    echo "Expected: $WILDGUARD_DIR"
    echo ""
    echo "Run WildGuard classification first:"
    echo "  uv run 03_classify_wildguard.py"
    exit 1
fi

# 5. Check classify_gcp directory exists
GCP_DIR="$SCRIPT_DIR/classify_gcp"

if [ ! -d "$GCP_DIR" ]; then
    echo "ERROR: GCP classification directory not found"
    echo ""
    echo "Expected: $GCP_DIR"
    echo ""
    echo "This directory should contain:"
    echo "  - Dockerfile"
    echo "  - classify_script.py"
    echo "  - submit.sh"
    exit 1
fi

read -p "Upload WildGuard files to GCS bucket? [y/N]: " -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to gs://${PROJECT_ID}-gemma-quickstart/wildguard_classified/"
    echo ""

    gcloud storage cp "$WILDGUARD_DIR"/*.json \
        "gs://${PROJECT_ID}-gemma-quickstart/wildguard_classified/"
else
    exit 1
fi

read -p "Submit Gemma classification job to GCP? [y/N]: " -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo "Navigating to GCP directory..."
cd "$GCP_DIR"

echo "Submitting job"
echo ""

./submit.sh
