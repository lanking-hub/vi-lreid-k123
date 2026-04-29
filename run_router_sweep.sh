#!/usr/bin/env bash
set -euo pipefail

# Sequentially run multiple router configurations on the server.
# Typical usage:
#   cd /data/sheng_hao_xuan_2025/code/my_work
#   nohup bash run_router_sweep.sh > /data/sheng_hao_xuan_2025/experiments/router_sweep_launcher.log 2>&1 &
#
# Optional overrides:
#   GPU=3 BASE_LOG_ROOT=/data/sheng_hao_xuan_2025/experiments bash run_router_sweep.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-3}"
SETTING="${SETTING:-6}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-/data/sheng_hao_xuan_2025/experiments}"
EPOCHS="${EPOCHS:-5}"
RUN_TAG="${RUN_TAG:-e${EPOCHS}}"

# Extra common arguments can be injected from the shell if needed, for example:
#   COMMON_ARGS="--debug_max_epoch 5" bash run_router_sweep.sh
COMMON_ARGS="${COMMON_ARGS:---debug_max_epoch ${EPOCHS}}"

# Format: run_name|train_debug.py extra args
RUNS=(
  "logs_debug_3stage_router_tau03_scale05_${RUN_TAG}|--route_tau 0.3 --train_k3_old_scale 0.5 --eval_k3_old_scale 0.5"
  "logs_debug_3stage_router_tau025_scale05_${RUN_TAG}|--route_tau 0.25 --train_k3_old_scale 0.5 --eval_k3_old_scale 0.5"
  "logs_debug_3stage_router_tau02_scale05_${RUN_TAG}|--route_tau 0.2 --train_k3_old_scale 0.5 --eval_k3_old_scale 0.5"
  "logs_debug_3stage_router_tau015_scale05_${RUN_TAG}|--route_tau 0.15 --train_k3_old_scale 0.5 --eval_k3_old_scale 0.5"
  "logs_debug_3stage_router_tau025_scale04_${RUN_TAG}|--route_tau 0.25 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.4"
  "logs_debug_3stage_router_tau02_scale04_${RUN_TAG}|--route_tau 0.2 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.4"
  "logs_debug_3stage_router_tau015_scale04_${RUN_TAG}|--route_tau 0.15 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.4"
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

count_gpus() {
  local gpu_csv="$1"
  awk -F',' '{print NF}' <<< "${gpu_csv}"
}

final_stage_name() {
  case "${SETTING}" in
    1)
      echo "vcm"
      ;;
    6)
      echo "llcm"
      ;;
    7)
      echo "regdb"
      ;;
    *)
      echo "Unsupported SETTING=${SETTING}" >&2
      exit 1
      ;;
  esac
}

echo "[$(timestamp)] Router sweep launcher started"
echo "[$(timestamp)] PROJECT_DIR=${PROJECT_DIR}"
echo "[$(timestamp)] GPU=${GPU} SETTING=${SETTING}"
echo "[$(timestamp)] BASE_LOG_ROOT=${BASE_LOG_ROOT}"
echo "[$(timestamp)] PYTHON_BIN=${PYTHON_BIN}"
echo "[$(timestamp)] EPOCHS=${EPOCHS} RUN_TAG=${RUN_TAG}"
if [[ -n "${COMMON_ARGS}" ]]; then
  echo "[$(timestamp)] COMMON_ARGS=${COMMON_ARGS}"
fi

cd "${PROJECT_DIR}"

read -r -a common_parts <<< "${COMMON_ARGS}"
NPROC="$(count_gpus "${GPU}")"
FINAL_STAGE="$(final_stage_name)"

if [[ "${NPROC}" -lt 1 ]]; then
  echo "Invalid GPU specification: ${GPU}" >&2
  exit 1
fi

echo "[$(timestamp)] NPROC=${NPROC}"
echo "[$(timestamp)] FINAL_STAGE=${FINAL_STAGE}"

for entry in "${RUNS[@]}"; do
  run_name="${entry%%|*}"
  run_args="${entry#*|}"
  logs_dir="${BASE_LOG_ROOT}/${run_name}"

  if [[ -f "${logs_dir}/${FINAL_STAGE}_stage.done" ]]; then
    echo "[$(timestamp)] Skip ${run_name}: final stage marker already exists"
    continue
  fi

  mkdir -p "${logs_dir}"
  read -r -a run_parts <<< "${run_args}"

  cmd=(
    "${PYTHON_BIN}"
  )
  if [[ "${NPROC}" -gt 1 ]]; then
    cmd+=( -m torch.distributed.run --nproc_per_node "${NPROC}" )
  fi
  cmd+=(
    train_debug.py
    --gpu "${GPU}"
    --logs-dir "${logs_dir}"
    --setting "${SETTING}"
  )
  cmd+=( "${common_parts[@]}" )
  cmd+=( "${run_parts[@]}" )

  echo "[$(timestamp)] Start ${run_name}"
  echo "[$(timestamp)] Command: ${cmd[*]}"
  "${cmd[@]}"
  echo "[$(timestamp)] Finished ${run_name}"
done

echo "[$(timestamp)] Router sweep launcher finished"
