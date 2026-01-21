#!/bin/bash
set -euo pipefail

PYTHON_SCRIPT="/code/fengruixiang/verifier/general_judge_filter_batch.py"
K_VALUE=3
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM}"

MODEL_IPS=(
  "/code/fengruixiang/verifier/verifier_ip/compass-verifier-7b.txt"
  "/code/fengruixiang/verifier/verifier_ip/llama3.3-70b.txt"
  "/code/fengruixiang/verifier/verifier_ip/compass-verifier-32b.txt"
)

DATASET_PATHS=(
  "./CosineVerifier/data/sci-verify-bench"
  "./CosineVerifier/data/verifybench"
  "./CosineVerifier/data/verifybench_hard"
)

OUTPUT_BASE_DIR="./label_results"

echo "K=${K_VALUE}"
echo "OUTPUT=${OUTPUT_BASE_DIR}"
echo "RUN_TAG=${RUN_TAG}"
echo "-----------------------------------"

mkdir -p "${OUTPUT_BASE_DIR}"

for model_ip_file in "${MODEL_IPS[@]}"; do
  MODEL_NAME=$(basename "${model_ip_file}" .txt)
  MODEL_RUN_NAME="${MODEL_NAME}_${RUN_TAG}"

  if [[ "${MODEL_NAME}" == *compass* ]]; then
    CURRENT_VERIFY_TYPE="COMPASS_COT"
  elif [[ "${MODEL_NAME}" == *xverify* ]]; then
    CURRENT_VERIFY_TYPE="XVERIFY"
  else
    CURRENT_VERIFY_TYPE="STEM"
  fi

  MODEL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_RUN_NAME}"
  mkdir -p "${MODEL_OUTPUT_DIR}"
  LOG_FILE="${MODEL_OUTPUT_DIR}/${MODEL_RUN_NAME}_${CURRENT_VERIFY_TYPE}.log"

  echo "Launching: ${MODEL_NAME}"
  echo "  Verify: ${CURRENT_VERIFY_TYPE}"
  echo "  Out: ${MODEL_OUTPUT_DIR}"
  echo "  Log: ${LOG_FILE}"

  python3 "${PYTHON_SCRIPT}" \
    --K "${K_VALUE}" \
    --service_ip_files "${model_ip_file}" \
    --dataset_paths "${DATASET_PATHS[@]}" \
    --output_base_dir "${MODEL_OUTPUT_DIR}" \
    --verify_type "${CURRENT_VERIFY_TYPE}" &> "${LOG_FILE}" &

  echo "  PID: $!"
done

