#!/usr/bin/env bash
set -euo pipefail

# Small, high-information validation launcher for the current router framework.
#
# Typical usage:
#   cd /data/sheng_hao_xuan_2025/code/my_work
#   bash run_router_validate_minimal.sh
#
# Optional overrides:
#   GPU=0,2 EPOCHS=5 RUN_TAG=validate_e5 bash run_router_validate_minimal.sh
#   RUN_SET=group_a bash run_router_validate_minimal.sh
#   RUN_SET=group_b bash run_router_validate_minimal.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0,2}"
SETTING="${SETTING:-6}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-/data/sheng_hao_xuan_2025/experiments}"
EPOCHS="${EPOCHS:-5}"
RUN_TAG="${RUN_TAG:-validate_e${EPOCHS}}"
RUN_SET="${RUN_SET:-all}"

COMMON_ARGS="${COMMON_ARGS:---debug_max_epoch ${EPOCHS}}"

GROUP_A=(
  "router_off_proxy_${RUN_TAG}|--route_tau 0.25 --train_k3_old_scale 0.0 --eval_k3_old_scale 0.0"
  "router_tau025_scale04_${RUN_TAG}|--route_tau 0.25 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.4"
  "router_tau02_scale04_${RUN_TAG}|--route_tau 0.2 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.4"
  "router_tau015_scale04_${RUN_TAG}|--route_tau 0.15 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.4"
)

GROUP_B=(
  "router_train_only_tau025_${RUN_TAG}|--route_tau 0.25 --train_k3_old_scale 0.4 --eval_k3_old_scale 0.0"
  "router_eval_only_tau025_${RUN_TAG}|--route_tau 0.25 --train_k3_old_scale 0.0 --eval_k3_old_scale 0.4"
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

select_runs() {
  case "${RUN_SET}" in
    all)
      printf "%s\n" "${GROUP_A[@]}" "${GROUP_B[@]}"
      ;;
    group_a)
      printf "%s\n" "${GROUP_A[@]}"
      ;;
    group_b)
      printf "%s\n" "${GROUP_B[@]}"
      ;;
    *)
      echo "Unsupported RUN_SET=${RUN_SET}" >&2
      exit 1
      ;;
  esac
}

echo "[$(timestamp)] Minimal router validation launcher started"
echo "[$(timestamp)] PROJECT_DIR=${PROJECT_DIR}"
echo "[$(timestamp)] PYTHON_BIN=${PYTHON_BIN}"
echo "[$(timestamp)] GPU=${GPU}"
echo "[$(timestamp)] SETTING=${SETTING}"
echo "[$(timestamp)] BASE_LOG_ROOT=${BASE_LOG_ROOT}"
echo "[$(timestamp)] EPOCHS=${EPOCHS} RUN_TAG=${RUN_TAG}"
echo "[$(timestamp)] RUN_SET=${RUN_SET}"
echo "[$(timestamp)] COMMON_ARGS=${COMMON_ARGS}"

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

while IFS= read -r entry; do
  [[ -z "${entry}" ]] && continue

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
done < <(select_runs)

echo "[$(timestamp)] Minimal router validation launcher finished"
