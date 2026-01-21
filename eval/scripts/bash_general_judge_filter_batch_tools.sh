#!/bin/bash

PYTHON_SCRIPT="general_judge_filter_batch_tools.py"
K_VALUE=3

MODEL_IPS=(
  "YOUR MODEL ip"
)

DATASET_PATHS=(
  "./CosineVerifier/data/sci-verify-bench"
  "./CosineVerifier/data/verifybench"
  "./CosineVerifier/data/verifybench_hard"
)

OUTPUT_BASE_DIR="./batch_results_tools"

echo "K=${K_VALUE}"
echo "OUTPUT=${OUTPUT_BASE_DIR}"
echo "-----------------------------------"

mkdir -p "${OUTPUT_BASE_DIR}"

for model_ip_file in "${MODEL_IPS[@]}"; do
  MODEL_NAME=$(basename "${model_ip_file}" .txt)

  if [[ "${MODEL_NAME}" == *compass_verifier* ]]; then
    CURRENT_VERIFY_TYPE="COMPASS"
  elif [[ "${MODEL_NAME}" == *xverify* ]]; then
    CURRENT_VERIFY_TYPE="XVERIFY"
  else
    CURRENT_VERIFY_TYPE="STEM"
  fi

  TIMESTAMP=$(date +"%Y%m%dT%H%M%SZ")
  RAND_ID=$((1000 + RANDOM % 9000))
  RUN_SUFFIX="${TIMESTAMP}_${RAND_ID}"

  LOG_FILE="${OUTPUT_BASE_DIR}/${MODEL_NAME}_${CURRENT_VERIFY_TYPE}_${RUN_SUFFIX}_eval.log"

  echo "Launching: ${MODEL_NAME}"
  echo "  Verify: ${CURRENT_VERIFY_TYPE}"
  echo "  Log: ${LOG_FILE}"

  python3 "${PYTHON_SCRIPT}" \
    --K "${K_VALUE}" \
    --service_ip_files "${model_ip_file}" \
    --dataset_paths "${DATASET_PATHS[@]}" \
    --output_base_dir "${OUTPUT_BASE_DIR}" \
    --verify_type "${CURRENT_VERIFY_TYPE}" \
    --run_suffix "${RUN_SUFFIX}" &> "${LOG_FILE}" &

  echo "  PID: $!"
done

