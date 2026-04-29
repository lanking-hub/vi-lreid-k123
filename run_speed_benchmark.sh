#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for quick single-vs-multi GPU speed checks with train_debug.py.
#
# Typical usage:
#   cd /data/sheng_hao_xuan_2025/code/my_work
#   bash run_speed_benchmark.sh
#
# Useful overrides:
#   MODE=fixed_global BASE_BATCH_SIZE=8 bash run_speed_benchmark.sh
#   MODE=fixed_local BASE_BATCH_SIZE=8 bash run_speed_benchmark.sh
#   RUNS="1:0;2:0,2;4:0,2,3,7" bash run_speed_benchmark.sh
#   COMMON_ARGS="--route_tau 0.25 --train_k3_old_scale 0.0 --eval_k3_old_scale 0.4" bash run_speed_benchmark.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-/data/sheng_hao_xuan_2025/experiments}"
SETTING="${SETTING:-7}"
EPOCHS="${EPOCHS:-1}"
MODE="${MODE:-fixed_global}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-8}"
RUN_TAG="${RUN_TAG:-speed_${MODE}_e${EPOCHS}_b${BASE_BATCH_SIZE}}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Format: nproc:gpu_ids ; nproc:gpu_ids ; ...
RUNS="${RUNS:-1:0;2:0,2;4:0,2,3,7}"

# Keep these defaults aligned with the recent smoke-test setup.
COMMON_ARGS="${COMMON_ARGS:---route_tau 0.25 --train_k3_old_scale 0.0 --eval_k3_old_scale 0.4}"

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

calc_global_batch_size() {
  local nproc="$1"
  if [[ "${MODE}" == "fixed_global" ]]; then
    echo "${BASE_BATCH_SIZE}"
  elif [[ "${MODE}" == "fixed_local" ]]; then
    echo $(( BASE_BATCH_SIZE * nproc ))
  else
    echo "Unsupported MODE=${MODE}" >&2
    exit 1
  fi
}

extract_summary() {
  local log_file="$1"
  local line
  line="$(grep 'Global Img/s:' "${log_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo $'NA\tNA'
    return
  fi

  local elapsed images_per_sec
  elapsed="$(sed -n 's/.*Time: \([0-9.]*\)s, Global Img\/s: \([0-9.]*\).*/\1/p' <<< "${line}")"
  images_per_sec="$(sed -n 's/.*Time: \([0-9.]*\)s, Global Img\/s: \([0-9.]*\).*/\2/p' <<< "${line}")"
  echo -e "${elapsed:-NA}\t${images_per_sec:-NA}"
}

echo "[$(timestamp)] Speed benchmark launcher started"
echo "[$(timestamp)] PROJECT_DIR=${PROJECT_DIR}"
echo "[$(timestamp)] PYTHON_BIN=${PYTHON_BIN}"
echo "[$(timestamp)] BASE_LOG_ROOT=${BASE_LOG_ROOT}"
echo "[$(timestamp)] SETTING=${SETTING} EPOCHS=${EPOCHS}"
echo "[$(timestamp)] MODE=${MODE} BASE_BATCH_SIZE=${BASE_BATCH_SIZE}"
echo "[$(timestamp)] COMMON_ARGS=${COMMON_ARGS}"
echo "[$(timestamp)] RUNS=${RUNS}"

mkdir -p "${BASE_LOG_ROOT}"
summary_file="${BASE_LOG_ROOT}/${RUN_TAG}_summary.tsv"
printf "nproc\tgpu_ids\tmode\tglobal_batch\tlocal_batch\telapsed_s\tglobal_img_s\tlogs_dir\n" > "${summary_file}"

cd "${PROJECT_DIR}"
IFS=';' read -r -a run_specs <<< "${RUNS}"
read -r -a common_parts <<< "${COMMON_ARGS}"
FINAL_STAGE="$(final_stage_name)"

echo "[$(timestamp)] FINAL_STAGE=${FINAL_STAGE}"

for spec in "${run_specs[@]}"; do
  nproc="${spec%%:*}"
  gpu_ids="${spec#*:}"
  gpu_count="$(count_gpus "${gpu_ids}")"
  if [[ "${gpu_count}" -ne "${nproc}" ]]; then
    echo "[$(timestamp)] Skip invalid spec ${spec}: gpu_count=${gpu_count}, nproc=${nproc}" >&2
    continue
  fi

  global_batch_size="$(calc_global_batch_size "${nproc}")"
  local_batch_size=$(( global_batch_size / nproc ))
  if (( global_batch_size % nproc != 0 )); then
    echo "[$(timestamp)] Skip ${spec}: global_batch_size=${global_batch_size} is not divisible by nproc=${nproc}" >&2
    continue
  fi

  run_name="${RUN_TAG}_${nproc}gpu"
  logs_dir="${BASE_LOG_ROOT}/${run_name}"
  log_file="${logs_dir}/log.txt"
  stage_done_file="${logs_dir}/${FINAL_STAGE}_stage.done"

  if [[ "${FORCE_RERUN}" != "1" && -f "${log_file}" && -f "${stage_done_file}" ]]; then
    echo "[$(timestamp)] Skip ${run_name}: final stage marker already exists"
    read -r elapsed_s global_img_s <<< "$(extract_summary "${log_file}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${nproc}" "${gpu_ids}" "${MODE}" "${global_batch_size}" "${local_batch_size}" \
      "${elapsed_s}" "${global_img_s}" "${logs_dir}" >> "${summary_file}"
    continue
  fi

  mkdir -p "${logs_dir}"
  cmd=(
    "${PYTHON_BIN}"
  )
  if [[ "${nproc}" -gt 1 ]]; then
    cmd+=( -m torch.distributed.run --nproc_per_node "${nproc}" )
  fi
  cmd+=(
    train_debug.py
    --gpu "${gpu_ids}"
    --logs-dir "${logs_dir}"
    --setting "${SETTING}"
    --debug_max_epoch "${EPOCHS}"
    --debug_batch_size "${global_batch_size}"
  )
  cmd+=( "${common_parts[@]}" )

  echo "[$(timestamp)] Start ${run_name}"
  echo "[$(timestamp)] Command: ${cmd[*]}"
  "${cmd[@]}"
  echo "[$(timestamp)] Finished ${run_name}"

  read -r elapsed_s global_img_s <<< "$(extract_summary "${log_file}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${nproc}" "${gpu_ids}" "${MODE}" "${global_batch_size}" "${local_batch_size}" \
    "${elapsed_s}" "${global_img_s}" "${logs_dir}" >> "${summary_file}"
done

echo "[$(timestamp)] Summary written to ${summary_file}"
cat "${summary_file}"
echo "[$(timestamp)] Speed benchmark launcher finished"
