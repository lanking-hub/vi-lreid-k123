#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for 4-stage VCM validation runs.
#
# Typical usage on the server:
#   cd /data/sheng_hao_xuan_2025/code/my_work
#   nohup env GPU=0,6 bash run_4stage_batch.sh > /data/sheng_hao_xuan_2025/experiments/4stage_batch_launcher.log 2>&1 < /dev/null &
#
# Useful overrides:
#   RUN_SET=lr bash run_4stage_batch.sh
#   RUN_SET=scale bash run_4stage_batch.sh
#   RUN_SET=all bash run_4stage_batch.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/sheng_hao_xuan_2025/envs/lreid/bin/python}"
GPU="${GPU:-0,6}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-/data/sheng_hao_xuan_2025/experiments}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_SET="${RUN_SET:-next}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29620}"

COMMON_ARGS=(
  --setting 1
  --debug_max_epoch "${EPOCHS}"
  --debug_batch_size "${BATCH_SIZE}"
  --route_tau 0.25
)

NEXT_RUNS=(
  "router_tau025_scale04_lr1e4_4stage_e5_2gpu|--train_k3_old_scale 0.4 --eval_k3_old_scale 0.4 BASE_LR 1e-4"
  "router_tau025_scale03_4stage_e5_2gpu|--train_k3_old_scale 0.3 --eval_k3_old_scale 0.3"
)

LR_RUNS=(
  "router_tau025_scale04_lr1e4_4stage_e5_2gpu|--train_k3_old_scale 0.4 --eval_k3_old_scale 0.4 BASE_LR 1e-4"
  "router_tau025_scale04_lr2e4_4stage_e5_2gpu|--train_k3_old_scale 0.4 --eval_k3_old_scale 0.4 BASE_LR 2e-4"
)

SCALE_RUNS=(
  "router_tau025_scale03_4stage_e5_2gpu|--train_k3_old_scale 0.3 --eval_k3_old_scale 0.3"
  "router_tau025_scale06_4stage_e5_2gpu|--train_k3_old_scale 0.6 --eval_k3_old_scale 0.6"
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

count_gpus() {
  local gpu_csv="$1"
  awk -F',' '{print NF}' <<< "${gpu_csv}"
}

select_runs() {
  case "${RUN_SET}" in
    next)
      printf "%s\n" "${NEXT_RUNS[@]}"
      ;;
    lr)
      printf "%s\n" "${LR_RUNS[@]}"
      ;;
    scale)
      printf "%s\n" "${SCALE_RUNS[@]}"
      ;;
    all)
      printf "%s\n" "${LR_RUNS[@]}" "${SCALE_RUNS[@]}"
      ;;
    *)
      echo "Unsupported RUN_SET=${RUN_SET}" >&2
      exit 1
      ;;
  esac
}

echo "[$(timestamp)] 4-stage batch launcher started"
echo "[$(timestamp)] PROJECT_DIR=${PROJECT_DIR}"
echo "[$(timestamp)] PYTHON_BIN=${PYTHON_BIN}"
echo "[$(timestamp)] GPU=${GPU}"
echo "[$(timestamp)] BASE_LOG_ROOT=${BASE_LOG_ROOT}"
echo "[$(timestamp)] EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE}"
echo "[$(timestamp)] RUN_SET=${RUN_SET}"
echo "[$(timestamp)] MASTER_PORT_BASE=${MASTER_PORT_BASE}"

cd "${PROJECT_DIR}"
mkdir -p "${BASE_LOG_ROOT}"

NPROC="$(count_gpus "${GPU}")"
if [[ "${NPROC}" -lt 1 ]]; then
  echo "Invalid GPU specification: ${GPU}" >&2
  exit 1
fi

run_idx=0
while IFS= read -r entry; do
  [[ -z "${entry}" ]] && continue

  run_name="${entry%%|*}"
  run_args="${entry#*|}"
  logs_dir="${BASE_LOG_ROOT}/${run_name}"
  stage_done_file="${logs_dir}/vcm_stage.done"
  master_port=$(( MASTER_PORT_BASE + run_idx ))
  run_idx=$(( run_idx + 1 ))

  if [[ -f "${stage_done_file}" ]]; then
    echo "[$(timestamp)] Skip ${run_name}: final stage marker already exists"
    continue
  fi

  mkdir -p "${logs_dir}"
  read -r -a run_parts <<< "${run_args}"

  cmd=(
    "${PYTHON_BIN}"
  )
  if [[ "${NPROC}" -gt 1 ]]; then
    cmd+=( -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${master_port}" )
  fi
  cmd+=(
    train_debug.py
    --gpu "${GPU}"
    --logs-dir "${logs_dir}"
  )
  cmd+=( "${COMMON_ARGS[@]}" )
  cmd+=( "${run_parts[@]}" )

  echo "[$(timestamp)] Start ${run_name}"
  echo "[$(timestamp)] Command: ${cmd[*]}"
  "${cmd[@]}"
  echo "[$(timestamp)] Finished ${run_name}"
done < <(select_runs)

echo "[$(timestamp)] 4-stage batch launcher finished"
