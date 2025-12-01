#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOMJOB_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$CUSTOMJOB_DIR/.." && pwd)"

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
JOB_NAME="gemma3-270m-it-yesno-training-$(date +%Y%m%d-%H%M%S)"
IMAGE_NAME="gemma3-270m-it-yesno-training"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

# Auto-detect service account based on active project
SERVICE_ACCOUNT="gemma-guard-runner@${PROJECT_ID}.iam.gserviceaccount.com"

# Auto-detect service account key file (look in customjob_vertexai root)
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
    echo "    ${PROJECT_ID}-key.json \\"
    echo "    --iam-account=${SERVICE_ACCOUNT}"
    exit 1
fi

KEY_FILENAME=$(basename "$KEY_FILE")

echo "======================================================="
echo "Vertex AI - Gemma-3-270M-IT Training"
echo "======================================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo "Service Account: ${SERVICE_ACCOUNT}"
echo "Service Account Key: ${KEY_FILENAME}"

read -p "Proceed with training job submission? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "[1/3] Building Docker image"
cd "$SCRIPT_DIR"

cp "$KEY_FILE" "$SCRIPT_DIR/${KEY_FILENAME}"
trap "rm -f '$SCRIPT_DIR/${KEY_FILENAME}'" EXIT

docker build --build-arg KEY_FILE=${KEY_FILENAME} -f Dockerfile -t ${IMAGE_URI} .

echo "[2/3] Pushing image to GCR"
docker push ${IMAGE_URI}

echo "[3/3] Submitting training job to Vertex AI"

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

rm ${CONFIG_FILE}

echo "============================================"
echo "✓ Job submitted successfully!"
echo "============================================"
echo "Monitor your job:"
echo "  gcloud ai custom-jobs list --region=${REGION}"
echo ""
echo "View logs (real-time):"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"
echo ""
echo "Or visit the Cloud Console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
