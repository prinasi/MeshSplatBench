#!/usr/bin/env bash
set -euo pipefail

# single_train.sh — TriangleBench training & evaluation pipeline.
#
# Trains triangle-splatting on MipNeRF-360 / Tanks & Temples / DTU scenes,
# evaluates image-quality metrics, measures rendering FPS, and (for DTU)
# computes Chamfer distance.  Prints a formatted summary table at the end.
#
# Output columns:
#   MipNeRF-360 / T&T : PSNR | SSIM | LPIPS | FPS | TrainMem(MiB) | TrainTime(s)
#   DTU               : PSNR | SSIM | LPIPS | FPS | TrainMem(MiB) | TrainTime(s) | Chamfer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MIPNERF360_ROOT="data/MipNeRF360"
TANKS_AND_TEMPLES_ROOT="data/tandt"
DTU_ROOT="data/DTU"
DTU_OFFICIAL_ROOT="data/DTU_Official"
OUTPUT_PATH="./outputs"
LOG_DIR="logs"
METHOD="triangle-splatting"
DATASET_TYPE_M360="colmap"
DATASET_TYPE_TAT="colmap"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

TB_CMD="trianglebench"

TRAIN_ITERATIONS=30000
EVAL_SPLIT=1
RESOLUTION=""
WHITE_BACKGROUND=0
IMAGE_DIR_M360_OUTDOOR="images_4"
IMAGE_DIR_M360_INDOOR="images_2"
IMAGE_DIR_TAT="images"

SKIP_TRAINING=0
SKIP_RENDERING=0
SKIP_METRICS=0
REQUESTED_TRAINING=0
REQUESTED_RENDERING=0
REQUESTED_METRICS=0
HAS_STAGE_REQUEST=0
EXPLICIT_SKIP_TRAINING=0
EXPLICIT_SKIP_RENDERING=0
EXPLICIT_SKIP_METRICS=0

SUMMARY_ROWS=()
LAST_TRAIN_MEMORY="N/A"
LAST_TRAIN_TIME="N/A"
LAST_PSNR="N/A"
LAST_SSIM="N/A"
LAST_LPIPS="N/A"
LAST_RENDER_FPS="N/A"
LAST_CHAMFER="N/A"

# -- Scene lists --------------------------------------------------------
MIPNERF360_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIPNERF360_INDOOR_SCENES=(room counter kitchen bonsai)
MIPNERF360_SCENES=("${MIPNERF360_OUTDOOR_SCENES[@]}" "${MIPNERF360_INDOOR_SCENES[@]}")
TANKS_AND_TEMPLES_SCENES=(truck train)
DTU_SCENES=(scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122)
ALL_SCENES=(
    "${MIPNERF360_SCENES[@]}"
    "${TANKS_AND_TEMPLES_SCENES[@]}"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage:
  bash $0 <scene|all|dtu> [scene ...] <gpu_id> [options]

Examples:
  bash $0 garden 0
  bash $0 bicycle flowers 0 --mipnerf360 data/MipNeRF360
  bash $0 truck 0 --tanksandtemples data/tandt
  bash $0 scan105 0 --dtu data/DTU
  bash $0 all 0                             # all MipNeRF360 + T&T scenes
  bash $0 dtu 0                             # all DTU scenes
  bash $0 garden 0 --rendering --metrics --skip_training

Options:
  --mipnerf360 PATH        MipNeRF-360 root           (default: ${MIPNERF360_ROOT})
  --tanksandtemples PATH   Tanks & Temples root       (default: ${TANKS_AND_TEMPLES_ROOT})
  --dtu PATH               DTU preprocessed root      (default: ${DTU_ROOT})
  --DTU_Official PATH      DTU official GT for CD     (default: ${DTU_OFFICIAL_ROOT})
  --output_path PATH       Output root                (default: ${OUTPUT_PATH})
  --method NAME            TriangleBench method       (default: ${METHOD})
  --train_iterations N     Training iterations        (default: ${TRAIN_ITERATIONS})
  --resolution N           Image resolution / width   (default: auto)
  --white_background       White background
  --training               Run training only
  --rendering              Run rendering only
  --metrics                Run metrics only
  --skip_training          Skip training
  --skip_rendering         Skip rendering
  --skip_metrics           Skip metrics
  --no_eval                Do not hold out test split
  --python PATH            Python executable
  -h, --help               Show this help
EOF
}

contains_scene() {
    local needle="$1"; shift
    for item in "$@"; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

require_value() {
    if [[ "$#" -lt 2 ]]; then
        echo "Missing value for $1" >&2; exit 1
    fi
}

resolve_python_bin() {
    if [[ "${PYTHON_BIN}" == */* ]]; then
        [[ ! -x "${PYTHON_BIN}" ]] && { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
        return
    fi
    local resolved
    resolved="$(command -v "${PYTHON_BIN}" || true)"
    [[ -z "${resolved}" ]] && { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
    PYTHON_BIN="${resolved}"
}

# -- Scene classification -------------------------------------------------
is_mipnerf360() {
    contains_scene "$1" "${MIPNERF360_SCENES[@]}"
}
is_mipnerf360_outdoor() {
    contains_scene "$1" "${MIPNERF360_OUTDOOR_SCENES[@]}"
}
is_tanks_and_temples() {
    contains_scene "$1" "${TANKS_AND_TEMPLES_SCENES[@]}"
}
is_dtu_scene() {
    contains_scene "$1" "${DTU_SCENES[@]}" || [[ "$1" =~ ^scan[0-9]+$ ]]
}
dtu_scan_id() { echo "${1#scan}"; }

scene_source() {
    local target="$1"
    if is_mipnerf360 "${target}"; then
        echo "${MIPNERF360_ROOT}/${target}"
    elif is_tanks_and_temples "${target}"; then
        echo "${TANKS_AND_TEMPLES_ROOT}/${target}"
    elif is_dtu_scene "${target}"; then
        echo "${DTU_ROOT}/${target}"
    elif [[ -d "${target}" ]]; then
        echo "${target}"
    else
        echo "Unknown scene or missing path: ${target}" >&2; exit 1
    fi
}

scene_name() {
    local t="$1"
    if [[ -d "${t}" || "${t}" == */* ]]; then basename "${t}"; else echo "${t}"; fi
}

scene_image_dir() {
    local target="$1"
    if is_mipnerf360_outdoor "${target}"; then
        echo "${IMAGE_DIR_M360_OUTDOOR}"
    elif is_mipnerf360 "${target}"; then
        echo "${IMAGE_DIR_M360_INDOOR}"
    else
        echo "${IMAGE_DIR_TAT}"
    fi
}

scene_dataset_type() {
    local target="$1"
    if is_dtu_scene "${target}"; then echo "dtu"
    else echo "colmap"
    fi
}

# -- Metric readers -------------------------------------------------------
read_metrics_from_json() {
    local metrics_file="$1"
    local values
    values=$("${PYTHON_BIN}" - "${metrics_file}" <<'PY'
import json, sys

def fmt(v):
    try: return f"{float(v):.4f}"
    except: return "N/A"

try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    agg = d.get("aggregate", {})
    print(fmt(agg.get("psnr_mean")), fmt(agg.get("ssim_mean")), fmt(agg.get("lpips_mean")))
except Exception:
    print("N/A N/A N/A")
PY
)
    read -r LAST_PSNR LAST_SSIM LAST_LPIPS <<< "${values}"
}

read_chamfer_from_json() {
    local results_file="$1"
    LAST_CHAMFER=$("${PYTHON_BIN}" - "${results_file}" <<'PY'
import json, sys

def fmt(v):
    try: return f"{float(v):.4f}"
    except: return "N/A"

try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    # Try various keys the DTU eval might produce
    for key in ("overall", "chamfer_distance", "cd", "mean"):
        if key in d:
            print(fmt(d[key]))
            raise SystemExit
    # Try nested "metrics" dict
    metrics = d.get("metrics", {})
    for key in ("overall", "chamfer_distance", "cd", "mean"):
        if key in metrics:
            print(fmt(metrics[key]))
            raise SystemExit
    print("N/A")
except SystemExit:
    pass
except Exception:
    print("N/A")
PY
)
}

read_metric_cache() {
    local path="$1" fallback="${2:-N/A}"
    [[ -f "${path}" ]] && tr -d '\n' < "${path}" || echo "${fallback}"
}

# ---------------------------------------------------------------------------
# Training (with memory monitoring)
# ---------------------------------------------------------------------------
run_train_with_memory() {
    local scene="$1" source="$2" model_path="$3"
    local image_dir="$4"
    local train_log="${LOG_DIR}/${scene}_gpu${GPU_ID}_train.log"
    local mem_log="${LOG_DIR}/${scene}_gpu${GPU_ID}_mem.log"
    local max_mem_file="${LOG_DIR}/${scene}_gpu${GPU_ID}_max_mem.txt"
    local train_time_file="${LOG_DIR}/${scene}_gpu${GPU_ID}_train_time.txt"

    rm -f "${train_log}" "${mem_log}" "${max_mem_file}" "${train_time_file}"

    # Build training command
    local -a cmd=(
        "${TB_CMD}" train
        -m "${METHOD}"
        -d "${source}"
        -o "${model_path}"
        --max-steps "${TRAIN_ITERATIONS}"
        --images "${image_dir}"
    )
    [[ "${EVAL_SPLIT}" -eq 1 ]] && cmd+=(--eval)
    [[ -n "${RESOLUTION}" ]] && cmd+=(-r "${RESOLUTION}")
    [[ "${WHITE_BACKGROUND}" -eq 1 ]] && cmd+=(--white-background)

    echo "[${scene}] Training started.  Log: ${train_log}"
    printf '[%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${scene}" "${GPU_ID}" "${cmd[*]}" > "${train_log}"

    local start_time end_time elapsed train_pid exit_code max_mem cur_mem
    start_time=$(date +%s)

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${train_log}" 2>&1 &
    train_pid=$!

    # Poll GPU memory
    max_mem=0
    sleep 2
    if command -v nvidia-smi >/dev/null 2>&1; then
        while kill -0 "${train_pid}" 2>/dev/null; do
            cur_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                      -i "${GPU_ID}" 2>/dev/null | tr -d ' ')
            echo "$(date '+%H:%M:%S') MEM=${cur_mem} MiB" >> "${mem_log}"
            if [[ "${cur_mem}" =~ ^[0-9]+$ ]] && (( cur_mem > max_mem )); then
                max_mem="${cur_mem}"
            fi
            sleep 1
        done
    else
        echo "nvidia-smi not found; skip memory monitoring." > "${mem_log}"
    fi

    set +e; wait "${train_pid}"; exit_code=$?; set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    local reported_mem="${max_mem}"
    [[ "${reported_mem}" == "0" ]] && reported_mem="N/A"
    echo "${reported_mem}" > "${max_mem_file}"
    echo "${elapsed}" > "${train_time_file}"
    LAST_TRAIN_MEMORY="${reported_mem}"
    LAST_TRAIN_TIME="${elapsed}"

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${scene}] Training FAILED (exit ${exit_code}).  Last log:"
        tail -n 30 "${train_log}" || true
        return "${exit_code}"
    fi
    echo "[${scene}] Training done.  Mem=${LAST_TRAIN_MEMORY} MiB  Time=${LAST_TRAIN_TIME}s"
}

# ---------------------------------------------------------------------------
# Rendering (test split, timed for FPS)
# ---------------------------------------------------------------------------
run_render() {
    local scene="$1" source="$2" model_path="$3"
    local image_dir="$4" dataset_type="$5"
    local render_dir="${model_path}/renders"
    local render_log="${LOG_DIR}/${scene}_gpu${GPU_ID}_render.log"
    local render_fps_file="${LOG_DIR}/${scene}_gpu${GPU_ID}_render_fps.txt"

    rm -f "${render_fps_file}"
    LAST_RENDER_FPS="N/A"

    local checkpoint="${model_path}/point_cloud/iteration_${TRAIN_ITERATIONS}"

    local -a cmd=(
        "${TB_CMD}" render images
        -m "${METHOD}"
        -c "${checkpoint}"
        -d "${source}"
        --split test
        -o "${render_dir}"
        --image-dir "${image_dir}"
    )
    [[ "${dataset_type}" != "colmap" ]] && cmd+=(--dataset-type "${dataset_type}")

    echo "[${scene}] Rendering started.  Log: ${render_log}"
    printf '[%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${scene}" "${GPU_ID}" "${cmd[*]}" > "${render_log}"

    local start_time end_time elapsed exit_code num_frames
    start_time=$(date +%s)

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${render_log}" 2>&1
    exit_code=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${scene}] Rendering FAILED.  Last log:"
        tail -n 30 "${render_log}" || true
        return "${exit_code}"
    fi

    # Parse "Rendered N frames to ..." from stdout captured in log
    num_frames=$(grep -oP 'Rendered \K\d+' "${render_log}" 2>/dev/null || echo "")
    if [[ -n "${num_frames}" && "${num_frames}" -gt 0 && "${elapsed}" -gt 0 ]]; then
        LAST_RENDER_FPS=$("${PYTHON_BIN}" -c "print(f'{${num_frames}/${elapsed}:.2f}')")
    fi
    echo "${LAST_RENDER_FPS}" > "${render_fps_file}"

    echo "[${scene}] Rendering done.  FPS=${LAST_RENDER_FPS}  (${num_frames:-?} frames in ${elapsed}s)"
}

# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------
run_metrics() {
    local scene="$1" source="$2" model_path="$3"
    local image_dir="$4" dataset_type="$5"
    local metrics_file="${model_path}/metrics.json"
    local metrics_log="${LOG_DIR}/${scene}_gpu${GPU_ID}_metrics.log"
    local checkpoint="${model_path}/point_cloud/iteration_${TRAIN_ITERATIONS}"

    local -a cmd=(
        "${TB_CMD}" eval images
        -m "${METHOD}"
        -c "${checkpoint}"
        -d "${source}"
        --split test
        -o "${metrics_file}"
        --image-dir "${image_dir}"
    )
    [[ "${dataset_type}" != "colmap" ]] && cmd+=(--dataset-type "${dataset_type}")

    echo "[${scene}] Metrics started.  Log: ${metrics_log}"
    printf '[%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${scene}" "${GPU_ID}" "${cmd[*]}" > "${metrics_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${metrics_log}" 2>&1
    local exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${scene}] Metrics FAILED.  Last log:"
        tail -n 30 "${metrics_log}" || true
        return "${exit_code}"
    fi

    read_metrics_from_json "${metrics_file}"
    echo "[${scene}] Metrics done.  PSNR=${LAST_PSNR}  SSIM=${LAST_SSIM}  LPIPS=${LAST_LPIPS}"
}

# ---------------------------------------------------------------------------
# DTU Chamfer distance
# ---------------------------------------------------------------------------
run_dtu_chamfer() {
    local scene="$1" source="$2" model_path="$3"
    local scan_id chamfer_log chamfer_file exit_code
    local eval_dir results_path

    scan_id="$(dtu_scan_id "${scene}")"
    chamfer_log="${LOG_DIR}/${scene}_gpu${GPU_ID}_chamfer.log"
    chamfer_file="${LOG_DIR}/${scene}_gpu${GPU_ID}_chamfer.txt"
    eval_dir="${model_path}/dtu_eval"
    results_path="${eval_dir}/scan${scan_id}/results.json"

    LAST_CHAMFER="N/A"

    if [[ ! -d "${DTU_OFFICIAL_ROOT}" ]]; then
        echo "[${scene}] DTU official GT path missing: ${DTU_OFFICIAL_ROOT}"
        echo "[${scene}] Skipping Chamfer distance.  Pass --DTU_Official PATH to enable."
        return 0
    fi

    # Try trianglebench eval dtu-mesh first
    local mesh_path
    mesh_path=$(find "${model_path}" -name "fuse*.ply" -o -name "mesh*.ply" | head -1)
    if [[ -z "${mesh_path}" ]]; then
        echo "[${scene}] No mesh found for DTU Chamfer.  Run rendering with mesh export first."
        return 0
    fi

    echo "[${scene}] DTU Chamfer started.  Log: ${chamfer_log}"

    local -a cmd=(
        "${TB_CMD}" eval dtu-mesh
        --pred "${mesh_path}"
        --dtu-root "${DTU_OFFICIAL_ROOT}"
        --scan-id "${scan_id}"
        -o "${results_path}"
    )

    printf '[%s] Command: %s\n\n' "${scene}" "${cmd[*]}" > "${chamfer_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${chamfer_log}" 2>&1
    exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${scene}] DTU Chamfer FAILED.  Last log:"
        tail -n 30 "${chamfer_log}" || true
        return "${exit_code}"
    fi

    read_chamfer_from_json "${results_path}"
    echo "${LAST_CHAMFER}" > "${chamfer_file}"
    echo "[${scene}] DTU Chamfer done.  CD=${LAST_CHAMFER}"
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print_final_summary() {
    if [[ "${#SUMMARY_ROWS[@]}" -eq 0 ]]; then return; fi

    echo
    echo "================================================================================"
    echo "  TriangleBench Metrics Summary  (method: ${METHOD})"
    echo "================================================================================"
    echo "  FPS unit: frames/s   |   TrainMem unit: MiB   |   TrainTime unit: s"
    echo "  Chamfer unit: DTU official eval distance (only for DTU scenes)"
    echo "================================================================================"
    printf "  %-16s %8s %8s %8s %8s %12s %12s %10s\n" \
        "Scene" "PSNR" "SSIM" "LPIPS" "FPS" "TrainMem(MiB)" "TrainTime(s)" "Chamfer"
    printf "  %-16s %8s %8s %8s %8s %12s %12s %10s\n" \
        "----------------" "--------" "--------" "--------" "--------" "------------" "------------" "----------"

    local row scene psnr ssim lpips fps mem time chamfer
    for row in "${SUMMARY_ROWS[@]}"; do
        IFS='|' read -r scene psnr ssim lpips fps mem time chamfer <<< "${row}"
        printf "  %-16s %8s %8s %8s %8s %12s %12s %10s\n" \
            "${scene}" "${psnr}" "${ssim}" "${lpips}" "${fps}" "${mem}" "${time}" "${chamfer}"
    done
    echo "================================================================================"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then usage; exit 1; fi
[[ "$1" == "-h" || "$1" == "--help" ]] && { usage; exit 0; }

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mipnerf360|-m360)  require_value "$@"; MIPNERF360_ROOT="$2"; shift 2 ;;
        --tanksandtemples|-tat) require_value "$@"; TANKS_AND_TEMPLES_ROOT="$2"; shift 2 ;;
        --dtu)               require_value "$@"; DTU_ROOT="$2"; shift 2 ;;
        --DTU_Official|-DTU) require_value "$@"; DTU_OFFICIAL_ROOT="$2"; shift 2 ;;
        --output_path)       require_value "$@"; OUTPUT_PATH="$2"; shift 2 ;;
        --method)            require_value "$@"; METHOD="$2"; shift 2 ;;
        --train_iterations|--iterations) require_value "$@"; TRAIN_ITERATIONS="$2"; shift 2 ;;
        --resolution|-r)     require_value "$@"; RESOLUTION="$2"; shift 2 ;;
        --white_background)  WHITE_BACKGROUND=1; shift ;;
        --training)          REQUESTED_TRAINING=1; HAS_STAGE_REQUEST=1; shift ;;
        --rendering)         REQUESTED_RENDERING=1; HAS_STAGE_REQUEST=1; shift ;;
        --metrics)           REQUESTED_METRICS=1; HAS_STAGE_REQUEST=1; shift ;;
        --skip_training)     EXPLICIT_SKIP_TRAINING=1; shift ;;
        --skip_rendering)    EXPLICIT_SKIP_RENDERING=1; shift ;;
        --skip_metrics)      EXPLICIT_SKIP_METRICS=1; shift ;;
        --no_eval)           EVAL_SPLIT=0; shift ;;
        --python)            require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)
            if [[ "$1" == --* ]]; then
                echo "Unknown option: $1" >&2; usage; exit 1
            fi
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ "${#POSITIONAL_ARGS[@]}" -lt 2 ]]; then
    usage; exit 1
fi

GPU_ID="${POSITIONAL_ARGS[$((${#POSITIONAL_ARGS[@]} - 1))]}"
TARGETS=("${POSITIONAL_ARGS[@]:0:$((${#POSITIONAL_ARGS[@]} - 1))}")

# Resolve stage flags
if [[ "${HAS_STAGE_REQUEST}" -eq 1 ]]; then
    SKIP_TRAINING=1; SKIP_RENDERING=1; SKIP_METRICS=1
    [[ "${REQUESTED_TRAINING}"  -eq 1 ]] && SKIP_TRAINING=0
    [[ "${REQUESTED_RENDERING}" -eq 1 ]] && SKIP_RENDERING=0
    [[ "${REQUESTED_METRICS}"   -eq 1 ]] && SKIP_METRICS=0
fi
[[ "${EXPLICIT_SKIP_TRAINING}"  -eq 1 ]] && SKIP_TRAINING=1
[[ "${EXPLICIT_SKIP_RENDERING}" -eq 1 ]] && SKIP_RENDERING=1
[[ "${EXPLICIT_SKIP_METRICS}"   -eq 1 ]] && SKIP_METRICS=1

if [[ "${SKIP_TRAINING}" -eq 1 && "${SKIP_RENDERING}" -eq 1 && "${SKIP_METRICS}" -eq 1 ]]; then
    echo "No stage selected." >&2; exit 1
fi

# Expand special targets
if contains_scene "all" "${TARGETS[@]}"; then
    SELECTED_TARGETS=("${ALL_SCENES[@]}")
elif contains_scene "dtu" "${TARGETS[@]}" || contains_scene "all_dtu" "${TARGETS[@]}"; then
    SELECTED_TARGETS=("${DTU_SCENES[@]}")
else
    SELECTED_TARGETS=("${TARGETS[@]}")
fi

resolve_python_bin
mkdir -p "${LOG_DIR}" "${OUTPUT_PATH}"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for TARGET in "${SELECTED_TARGETS[@]}"; do
    SCENE="$(scene_name "${TARGET}")"
    SOURCE="$(scene_source "${TARGET}")"
    MODEL_PATH="${OUTPUT_PATH}/${METHOD}/${SCENE}"
    IMAGE_DIR="$(scene_image_dir "${TARGET}")"
    DTYPE="$(scene_dataset_type "${TARGET}")"

    [[ ! -d "${SOURCE}" ]] && { echo "[${SCENE}] Dataset missing: ${SOURCE}" >&2; exit 1; }

    # Restore cached values
    LAST_TRAIN_MEMORY="$(read_metric_cache "${LOG_DIR}/${SCENE}_gpu${GPU_ID}_max_mem.txt")"
    LAST_TRAIN_TIME="$(read_metric_cache "${LOG_DIR}/${SCENE}_gpu${GPU_ID}_train_time.txt")"
    LAST_RENDER_FPS="$(read_metric_cache "${LOG_DIR}/${SCENE}_gpu${GPU_ID}_render_fps.txt")"
    LAST_CHAMFER="$(read_metric_cache "${LOG_DIR}/${SCENE}_gpu${GPU_ID}_chamfer.txt")"
    read_metrics_from_json "${MODEL_PATH}/metrics.json"

    # -- Train --
    if [[ "${SKIP_TRAINING}" -eq 0 ]]; then
        run_train_with_memory "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}"
    fi

    # -- Render --
    if [[ "${SKIP_RENDERING}" -eq 0 ]]; then
        run_render "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}" "${DTYPE}"
    fi

    # -- Metrics --
    if [[ "${SKIP_METRICS}" -eq 0 ]]; then
        run_metrics "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}" "${DTYPE}"
        if is_dtu_scene "${SCENE}"; then
            run_dtu_chamfer "${SCENE}" "${SOURCE}" "${MODEL_PATH}"
        fi
    fi

    echo "[${SCENE}] Summary: PSNR=${LAST_PSNR} SSIM=${LAST_SSIM} LPIPS=${LAST_LPIPS} FPS=${LAST_RENDER_FPS} Mem=${LAST_TRAIN_MEMORY}MiB Time=${LAST_TRAIN_TIME}s Chamfer=${LAST_CHAMFER}"

    SUMMARY_ROWS+=("${SCENE}|${LAST_PSNR}|${LAST_SSIM}|${LAST_LPIPS}|${LAST_RENDER_FPS}|${LAST_TRAIN_MEMORY}|${LAST_TRAIN_TIME}|${LAST_CHAMFER}")
done

print_final_summary
