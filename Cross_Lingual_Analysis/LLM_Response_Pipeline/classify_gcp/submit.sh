#!/bin/bash
# Submit Classification job to Vertex AI

set -e

# Detect script location and project root (gemmaguard/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables from .env file in project root
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
else
    echo "Error: .env file not found in $PROJECT_ROOT"
    echo "Create a .env file in the project root with your HF_TOKEN."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set in .env file"
    exit 1
fi

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-east4"
JOB_NAME="classification-$(date +%Y%m%d-%H%M%S)"
IMAGE_NAME="classification"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

# Auto-detect service account based on active project
SERVICE_ACCOUNT="gemma-guard-runner@${PROJECT_ID}.iam.gserviceaccount.com"

# Auto-detect service account key file
CUSTOMJOB_DIR="$(cd "$PROJECT_ROOT/customjob_vertexai" && pwd)"
KEY_FILE=""
if [ -f "$CUSTOMJOB_DIR/${PROJECT_ID}-key.json" ]; then
    KEY_FILE="$CUSTOMJOB_DIR/${PROJECT_ID}-key.json"
else
    # Try to find any key file matching the project ID
    KEY_FILE=$(find "$CUSTOMJOB_DIR" -maxdepth 1 -name "${PROJECT_ID}-*.json" -type f | head -n 1)
fi

if [ -z "$KEY_FILE" ]; then
    echo "Error: No service account key found for project ${PROJECT_ID}"
    echo "Expected: $CUSTOMJOB_DIR/${PROJECT_ID}-key.json"
    echo ""
    echo "Create a service account key with:"
    echo "  gcloud iam service-accounts keys create \\"
    echo "    customjob_vertexai/${PROJECT_ID}-key.json \\"
    echo "    --iam-account=${SERVICE_ACCOUNT}"
    exit 1
fi

KEY_FILENAME=$(basename "$KEY_FILE")

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo "Service Account: ${SERVICE_ACCOUNT}"
echo "Service Account Key: ${KEY_FILENAME}"
echo ""
echo "HARDWARE CONFIGURATION:"
echo "  Machine Type: g2-standard-8"
echo "  - 8 vCPU, 32GB RAM"
echo "  - Single NVIDIA L4 GPU (24GB)"
echo ""
echo "CLASSIFICATION CONFIGURATION:"
echo "  Refusal Classifier:"
echo "    - Model: Custom Gemma-3-270M-IT"
echo "    - Approach: Yes/No token classification"
echo "    - Threshold: 0.2"
echo ""
echo "  Harmfulness Classifier:"
echo "    - Model: WildGuard-7B"
echo "    - Approach: HuggingFace InferenceClient"
echo ""
echo "INPUT DATA:"
echo "  - Location: gs://${PROJECT_ID}-gemma-quickstart/wildguard_classified/"
echo ""
echo "OUTPUT DATA:"
echo "  - Location: gs://${PROJECT_ID}-gemma-quickstart/gemma_classified/"
echo ""

# Confirm before proceeding
read -p "Proceed with classification job submission? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Build Docker image
echo ""
echo "[1/3] Building Docker image"
cd "$SCRIPT_DIR"

# Temporarily copy the service account key to build context
cp "$KEY_FILE" "$SCRIPT_DIR/${KEY_FILENAME}"
trap "rm -f '$SCRIPT_DIR/${KEY_FILENAME}'" EXIT

docker build --build-arg KEY_FILE=${KEY_FILENAME} -f Dockerfile -t ${IMAGE_URI} .

# Step 2: Push to Google Container Registry
echo ""
echo "[2/3] Pushing image to GCR"
docker push ${IMAGE_URI}

# Step 3: Submit custom job with single L4 GPU
echo ""
echo "[3/3] Submitting classification job to Vertex AI"

# Create a temporary config file with single L4 GPU configuration
CONFIG_FILE=$(mktemp)
cat > ${CONFIG_FILE} <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: g2-standard-8
      acceleratorType: NVIDIA_L4
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskSizeGb: 100
      bootDiskType: pd-ssd
    containerSpec:
      imageUri: ${IMAGE_URI}
      env:
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
        - name: GOOGLE_CLOUD_PROJECT
          value: "${PROJECT_ID}"
EOF

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --service-account=${SERVICE_ACCOUNT} \
  --config=${CONFIG_FILE}

# Clean up
rm ${CONFIG_FILE}

echo "Monitor your job:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "When complete, download results:"
echo "  gcloud storage cp -r gs://${PROJECT_ID}-gemma-quickstart/gemma_classified/ ."
echo ""
echo "Next steps:"
echo "  1. Run LogReg classification:"
echo "     05_classify_logreg.py"
echo ""
echo "  2. Run analysis:"
echo "     06_analyze_all_models.py"
echo ""
