#!/usr/bin/env bash
set -euo pipefail

# single_train.sh - one-scene TriBench training/evaluation pipeline.
#
# Primary interface:
#   bash single_train.sh <method> <dataset/scene|dataset/all|all> <gpu_id> [options]
#
# Default result layout:
#   outputs/<method>/<dataset>/<scene>/
#
# Pipeline:
#   train -> render train/test splits -> compute test metrics -> export video
#   -> optional DTU Chamfer -> formatted per-scene summary

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MIPNERF360_ROOT="data/MipNeRF360"
TANKS_AND_TEMPLES_ROOT="data/tandt"
DTU_ROOT="data/DTU"
DTU_OFFICIAL_ROOT="data/DTU_Official"
OUTPUT_PATH="outputs"
LOG_ROOT=""
LOG_DIR=""
METHOD=""

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

TB_CMD="${TRIBENCH_CMD:-tribench}"
TB_CMD_ARR=()

TRAIN_ITERATIONS=30000
EVAL_SPLIT=1
EVAL_EVERY=8
RESOLUTION="-1"
RENDER_RESOLUTION="1"
VIDEO_FRAMES=240
VIDEO_FPS=30
VIDEO_ZOOM="1.0"
VIDEO_Z_VARIATION="0.0"
VIDEO_Z_PHASE="0.0"
WHITE_BACKGROUND=0
IMAGE_DIR_M360_OUTDOOR="images_4"
IMAGE_DIR_M360_INDOOR="images_2"
IMAGE_DIR_TAT="images"
IMAGE_DIR_DTU="images"

SKIP_TRAINING=0
SKIP_RENDERING=0
SKIP_METRICS=0
SKIP_VIDEO=0
REQUESTED_TRAINING=0
REQUESTED_RENDERING=0
REQUESTED_METRICS=0
REQUESTED_VIDEO=0
HAS_STAGE_REQUEST=0
EXPLICIT_SKIP_TRAINING=0
EXPLICIT_SKIP_RENDERING=0
EXPLICIT_SKIP_METRICS=0
EXPLICIT_SKIP_VIDEO=0

SUMMARY_ROWS=()
LAST_TRAIN_MEMORY="N/A"
LAST_TRAIN_TIME="N/A"
LAST_PSNR="N/A"
LAST_SSIM="N/A"
LAST_LPIPS="N/A"
LAST_RENDER_FPS="N/A"
LAST_VIDEO="N/A"
LAST_CHAMFER="N/A"

# -- Scene lists ------------------------------------------------------------
MIPNERF360_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIPNERF360_INDOOR_SCENES=(room counter kitchen bonsai)
MIPNERF360_SCENES=("${MIPNERF360_OUTDOOR_SCENES[@]}" "${MIPNERF360_INDOOR_SCENES[@]}")
TANKS_AND_TEMPLES_SCENES=(truck train)
DTU_SCENES=(scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage:
  bash $0 <method> <dataset/scene|dataset/all|all> <gpu_id> [options]

Examples:
  bash $0 triangle-splatting mipnerf360/garden 0
  bash $0 triangle_splatting m360/room 0 --mipnerf360 data/MipNeRF360
  bash $0 triangle-splatting tandt/truck 0 --tanksandtemples data/tandt
  bash $0 triangle-splatting dtu/scan105 0 --dtu data/DTU
  bash $0 triangle-splatting mipnerf360/all 0
  bash $0 triangle-splatting all 0

Legacy form still works:
  bash $0 garden 0 --method triangle-splatting

Options:
  --mipnerf360 PATH        MipNeRF-360 root           (default: ${MIPNERF360_ROOT})
  --tanksandtemples PATH   Tanks & Temples root       (default: ${TANKS_AND_TEMPLES_ROOT})
  --dtu PATH               DTU preprocessed root      (default: ${DTU_ROOT})
  --DTU_Official PATH      DTU official GT for CD     (default: ${DTU_OFFICIAL_ROOT})
  --output_path PATH       Output root                (default: ${OUTPUT_PATH})
  --log_dir PATH           Optional shared log root   (default: per-scene logs/)
  --method NAME            Method for legacy form
  --train_iterations N     Training iterations        (default: ${TRAIN_ITERATIONS})
  --resolution N           Training resolution        (default: ${RESOLUTION})
  --render_resolution N    Render/eval/video resolution (default: ${RENDER_RESOLUTION})
  --eval_every N           Holdout stride             (default: ${EVAL_EVERY})
  --video_frames N         Video frames               (default: ${VIDEO_FRAMES})
  --video_fps N            Video FPS                  (default: ${VIDEO_FPS})
  --white_background       White background
  --training               Run training only
  --rendering              Run rendering only
  --metrics                Run metrics only
  --video                  Run video export only
  --skip_training          Skip training
  --skip_rendering         Skip render train/test
  --skip_metrics           Skip image metrics
  --skip_video             Skip video export
  --no_eval                Do not hold out test split during training
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
        echo "Missing value for $1" >&2
        exit 1
    fi
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

canonical_method() {
    local name
    name="$(lower "$1")"
    name="${name//_/-}"
    case "${name}" in
        trianglesplatting) echo "triangle-splatting" ;;
        meshsplatting) echo "mesh-splatting" ;;
        2dts|d2ts) echo "2dts" ;;
        *) echo "${name}" ;;
    esac
}

normalize_dataset() {
    local name
    name="$(lower "$1")"
    name="${name//_/-}"
    case "${name}" in
        mipnerf360|mipnerf-360|m360|360|nerf360) echo "mipnerf360" ;;
        tanksandtemples|tanks-and-temples|tandt|tat|tanks) echo "tandt" ;;
        dtu) echo "dtu" ;;
        custom|path) echo "custom" ;;
        *) echo "${name}" ;;
    esac
}

dataset_label() {
    case "$(normalize_dataset "$1")" in
        mipnerf360) echo "mipnerf360" ;;
        tandt) echo "tandt" ;;
        dtu) echo "dtu" ;;
        custom) echo "custom" ;;
        *) echo "$(normalize_dataset "$1")" ;;
    esac
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

resolve_tb_cmd() {
    if command -v "${TB_CMD}" >/dev/null 2>&1; then
        TB_CMD_ARR=("${TB_CMD}")
    else
        TB_CMD_ARR=("${PYTHON_BIN}" -m tribench.cli.main)
    fi
}

# -- Scene classification --------------------------------------------------
is_mipnerf360() { contains_scene "$1" "${MIPNERF360_SCENES[@]}"; }
is_mipnerf360_outdoor() { contains_scene "$1" "${MIPNERF360_OUTDOOR_SCENES[@]}"; }
is_tanks_and_temples() { contains_scene "$1" "${TANKS_AND_TEMPLES_SCENES[@]}"; }
is_dtu_scene() { contains_scene "$1" "${DTU_SCENES[@]}" || [[ "$1" =~ ^scan[0-9]+$ ]]; }
dtu_scan_id() { echo "${1#scan}"; }

infer_dataset_for_scene() {
    local scene="$1"
    if is_mipnerf360 "${scene}"; then
        echo "mipnerf360"
    elif is_tanks_and_temples "${scene}"; then
        echo "tandt"
    elif is_dtu_scene "${scene}"; then
        echo "dtu"
    else
        echo "custom"
    fi
}

dataset_root() {
    case "$(normalize_dataset "$1")" in
        mipnerf360) echo "${MIPNERF360_ROOT}" ;;
        tandt) echo "${TANKS_AND_TEMPLES_ROOT}" ;;
        dtu) echo "${DTU_ROOT}" ;;
        custom) echo "" ;;
        *) echo "" ;;
    esac
}

scene_image_dir() {
    local dataset="$1" scene="$2"
    case "$(normalize_dataset "${dataset}")" in
        mipnerf360)
            if is_mipnerf360_outdoor "${scene}"; then
                echo "${IMAGE_DIR_M360_OUTDOOR}"
            else
                echo "${IMAGE_DIR_M360_INDOOR}"
            fi
            ;;
        tandt) echo "${IMAGE_DIR_TAT}" ;;
        dtu) echo "${IMAGE_DIR_DTU}" ;;
        *) echo "images" ;;
    esac
}

scene_dataset_type() {
    case "$(normalize_dataset "$1")" in
        dtu) echo "dtu" ;;
        mipnerf360|tandt) echo "colmap" ;;
        *) echo "auto" ;;
    esac
}

scene_source() {
    local dataset="$1" scene="$2" raw_target="${3:-}"
    if [[ "$(normalize_dataset "${dataset}")" == "custom" ]]; then
        if [[ -d "${raw_target}" ]]; then
            echo "${raw_target}"
        elif [[ -d "${scene}" ]]; then
            echo "${scene}"
        else
            echo "Unknown scene or missing path: ${raw_target:-${scene}}" >&2
            exit 1
        fi
        return
    fi

    local root
    root="$(dataset_root "${dataset}")"
    [[ -z "${root}" ]] && { echo "Unknown dataset: ${dataset}" >&2; exit 1; }
    echo "${root}/${scene}"
}

parse_target() {
    local target="$1"
    if [[ -d "${target}" ]]; then
        PARSED_DATASET="custom"
        PARSED_SCENE="$(basename "${target}")"
        PARSED_RAW_TARGET="${target}"
        return
    fi

    if [[ "${target}" == */* ]]; then
        PARSED_DATASET="$(dataset_label "${target%%/*}")"
        PARSED_SCENE="${target#*/}"
        PARSED_RAW_TARGET="${target}"
        return
    fi

    PARSED_DATASET="$(infer_dataset_for_scene "${target}")"
    PARSED_SCENE="${target}"
    PARSED_RAW_TARGET="${target}"
}

expand_targets() {
    local target dataset scene
    EXPANDED_TARGETS=()
    for target in "$@"; do
        if [[ "${target}" == "all" ]]; then
            for scene in "${MIPNERF360_SCENES[@]}"; do EXPANDED_TARGETS+=("mipnerf360/${scene}"); done
            for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do EXPANDED_TARGETS+=("tandt/${scene}"); done
            continue
        fi

        if [[ "${target}" == "dtu" || "${target}" == "dtu/all" || "${target}" == "all_dtu" ]]; then
            for scene in "${DTU_SCENES[@]}"; do EXPANDED_TARGETS+=("dtu/${scene}"); done
            continue
        fi

        if [[ "${target}" == */all ]]; then
            dataset="$(dataset_label "${target%%/*}")"
            case "${dataset}" in
                mipnerf360) for scene in "${MIPNERF360_SCENES[@]}"; do EXPANDED_TARGETS+=("mipnerf360/${scene}"); done ;;
                tandt) for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do EXPANDED_TARGETS+=("tandt/${scene}"); done ;;
                dtu) for scene in "${DTU_SCENES[@]}"; do EXPANDED_TARGETS+=("dtu/${scene}"); done ;;
                *) echo "Cannot expand unknown dataset target: ${target}" >&2; exit 1 ;;
            esac
            continue
        fi

        EXPANDED_TARGETS+=("${target}")
    done
}

checkpoint_path() {
    local model_path="$1"
    local expected="${model_path}/point_cloud/iteration_${TRAIN_ITERATIONS}"
    if [[ -f "${expected}/point_cloud_state_dict.pt" || -f "${expected}" ]]; then
        echo "${expected}"
        return
    fi

    local latest
    latest="$(find "${model_path}/point_cloud" -maxdepth 1 -type d -name 'iteration_*' 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
        return
    fi

    echo "${expected}"
}

log_prefix() {
    local dataset="$1" scene="$2"
    echo "$(dataset_label "${dataset}")_${scene}_gpu${GPU_ID}"
}

# -- Metric readers --------------------------------------------------------
read_metrics_from_json() {
    local metrics_file="$1"
    local values
    values=$("${PYTHON_BIN}" - "${metrics_file}" <<'PY'
import json, sys

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "N/A"

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
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "N/A"

try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    for key in ("overall", "chamfer_distance", "chamfer", "cd", "mean"):
        if key in d:
            print(fmt(d[key]))
            raise SystemExit
    metrics = d.get("metrics", {})
    for key in ("overall", "chamfer_distance", "chamfer", "cd", "mean"):
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

parse_num_frames() {
    local log_file="$1"
    grep -oE 'Rendered [0-9]+' "${log_file}" 2>/dev/null | awk '{print $2}' | tail -1 || true
}

# ---------------------------------------------------------------------------
# Training (with memory monitoring)
# ---------------------------------------------------------------------------
run_train_with_memory() {
    local dataset="$1" scene="$2" source="$3" model_path="$4" image_dir="$5"
    local prefix train_log mem_log max_mem_file train_time_file
    prefix="$(log_prefix "${dataset}" "${scene}")"
    train_log="${LOG_DIR}/${prefix}_train.log"
    mem_log="${LOG_DIR}/${prefix}_mem.log"
    max_mem_file="${LOG_DIR}/${prefix}_max_mem.txt"
    train_time_file="${LOG_DIR}/${prefix}_train_time.txt"

    rm -f "${train_log}" "${mem_log}" "${max_mem_file}" "${train_time_file}"
    mkdir -p "${model_path}"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" train
        -m "${METHOD_ID}"
        -d "${source}"
        -o "${model_path}"
        --max-steps "${TRAIN_ITERATIONS}"
        --save-interval 1000
        --images "${image_dir}"
        --resolution "${RESOLUTION}"
    )
    [[ "${EVAL_SPLIT}" -eq 1 ]] && cmd+=(--eval) || cmd+=(--no-eval)
    [[ "${WHITE_BACKGROUND}" -eq 1 ]] && cmd+=(--white-background)
    if [[ "$(normalize_dataset "${dataset}")" == "mipnerf360" ]] && is_mipnerf360_outdoor "${scene}"; then
        cmd+=(--outdoor)
    fi

    echo "[${dataset}/${scene}] Training started. Log: ${train_log}"
    printf '[%s/%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${dataset}" "${scene}" "${GPU_ID}" "${cmd[*]}" > "${train_log}"

    local start_time end_time elapsed train_pid exit_code max_mem cur_mem
    start_time=$(date +%s)

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${train_log}" 2>&1 &
    train_pid=$!

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

    set +e
    wait "${train_pid}"
    exit_code=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    local reported_mem="${max_mem}"
    [[ "${reported_mem}" == "0" ]] && reported_mem="N/A"
    echo "${reported_mem}" > "${max_mem_file}"
    echo "${elapsed}" > "${train_time_file}"
    LAST_TRAIN_MEMORY="${reported_mem}"
    LAST_TRAIN_TIME="${elapsed}"

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] Training FAILED (exit ${exit_code}). Last log:"
        tail -n 30 "${train_log}" || true
        return "${exit_code}"
    fi
    echo "[${dataset}/${scene}] Training done. Mem=${LAST_TRAIN_MEMORY} MiB Time=${LAST_TRAIN_TIME}s"
}

# ---------------------------------------------------------------------------
# Rendering (train/test splits, timed on test split for FPS)
# ---------------------------------------------------------------------------
run_render_split() {
    local dataset="$1" scene="$2" source="$3" model_path="$4" image_dir="$5" dataset_type="$6" split="$7"
    local render_dir="${model_path}/renders/${split}"
    local prefix render_log checkpoint
    prefix="$(log_prefix "${dataset}" "${scene}")"
    render_log="${LOG_DIR}/${prefix}_render_${split}.log"
    checkpoint="$(checkpoint_path "${model_path}")"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" render images
        -m "${METHOD_ID}"
        -c "${checkpoint}"
        -d "${source}"
        --split "${split}"
        -o "${render_dir}"
        --image-dir "${image_dir}"
        --resolution "${RENDER_RESOLUTION}"
        --eval-every "${EVAL_EVERY}"
        --save-gt
    )
    [[ "${dataset_type}" != "auto" ]] && cmd+=(--dataset-type "${dataset_type}")

    echo "[${dataset}/${scene}] Rendering ${split} started. Log: ${render_log}"
    printf '[%s/%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${dataset}" "${scene}" "${GPU_ID}" "${cmd[*]}" > "${render_log}"

    local start_time end_time elapsed exit_code num_frames
    start_time=$(date +%s)

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${render_log}" 2>&1
    exit_code=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] Rendering ${split} FAILED. Last log:"
        tail -n 30 "${render_log}" || true
        return "${exit_code}"
    fi

    num_frames="$(parse_num_frames "${render_log}")"
    if [[ "${split}" == "test" ]]; then
        if [[ -n "${num_frames}" && "${num_frames}" -gt 0 && "${elapsed}" -gt 0 ]]; then
            LAST_RENDER_FPS=$("${PYTHON_BIN}" -c "print(f'{${num_frames}/${elapsed}:.2f}')")
        else
            LAST_RENDER_FPS="N/A"
        fi
        echo "${LAST_RENDER_FPS}" > "${LOG_DIR}/${prefix}_render_fps.txt"
    fi

    echo "[${dataset}/${scene}] Rendering ${split} done. Frames=${num_frames:-?} Time=${elapsed}s"
}

run_render() {
    local dataset="$1" scene="$2" source="$3" model_path="$4" image_dir="$5" dataset_type="$6"
    LAST_RENDER_FPS="N/A"
    run_render_split "${dataset}" "${scene}" "${source}" "${model_path}" "${image_dir}" "${dataset_type}" "train"
    run_render_split "${dataset}" "${scene}" "${source}" "${model_path}" "${image_dir}" "${dataset_type}" "test"
    echo "[${dataset}/${scene}] Rendering done. Test FPS=${LAST_RENDER_FPS}"
}

# ---------------------------------------------------------------------------
# Image quality metrics (test split)
# ---------------------------------------------------------------------------
run_metrics() {
    local dataset="$1" scene="$2" source="$3" model_path="$4" image_dir="$5" dataset_type="$6"
    local metrics_file="${model_path}/metrics.json"
    local render_dir="${model_path}/renders/test_metrics"
    local prefix metrics_log checkpoint
    prefix="$(log_prefix "${dataset}" "${scene}")"
    metrics_log="${LOG_DIR}/${prefix}_metrics.log"
    checkpoint="$(checkpoint_path "${model_path}")"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" eval images
        -m "${METHOD_ID}"
        -c "${checkpoint}"
        -d "${source}"
        --split test
        -o "${metrics_file}"
        --render-dir "${render_dir}"
        --image-dir "${image_dir}"
        --resolution "${RENDER_RESOLUTION}"
        --eval-every "${EVAL_EVERY}"
    )
    [[ "${dataset_type}" != "auto" ]] && cmd+=(--dataset-type "${dataset_type}")

    echo "[${dataset}/${scene}] Metrics started. Log: ${metrics_log}"
    printf '[%s/%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${dataset}" "${scene}" "${GPU_ID}" "${cmd[*]}" > "${metrics_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${metrics_log}" 2>&1
    local exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] Metrics FAILED. Last log:"
        tail -n 30 "${metrics_log}" || true
        return "${exit_code}"
    fi

    read_metrics_from_json "${metrics_file}"
    echo "[${dataset}/${scene}] Metrics done. PSNR=${LAST_PSNR} SSIM=${LAST_SSIM} LPIPS=${LAST_LPIPS}"
}

# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------
run_video() {
    local dataset="$1" scene="$2" source="$3" model_path="$4" image_dir="$5" dataset_type="$6"
    local video_dir="${model_path}/videos"
    local prefix video_log checkpoint
    prefix="$(log_prefix "${dataset}" "${scene}")"
    video_log="${LOG_DIR}/${prefix}_video.log"
    checkpoint="$(checkpoint_path "${model_path}")"
    LAST_VIDEO="N/A"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" render video
        -m "${METHOD_ID}"
        -c "${checkpoint}"
        -d "${source}"
        --split train
        -o "${video_dir}"
        --image-dir "${image_dir}"
        --resolution "${RENDER_RESOLUTION}"
        --eval-every "${EVAL_EVERY}"
        --frames "${VIDEO_FRAMES}"
        --fps "${VIDEO_FPS}"
        --zoom "${VIDEO_ZOOM}"
        --z-variation "${VIDEO_Z_VARIATION}"
        --z-phase "${VIDEO_Z_PHASE}"
    )
    [[ "${dataset_type}" != "auto" ]] && cmd+=(--dataset-type "${dataset_type}")

    echo "[${dataset}/${scene}] Video export started. Log: ${video_log}"
    printf '[%s/%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${dataset}" "${scene}" "${GPU_ID}" "${cmd[*]}" > "${video_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${video_log}" 2>&1
    local exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] Video export FAILED. Last log:"
        tail -n 30 "${video_log}" || true
        return "${exit_code}"
    fi

    LAST_VIDEO="${video_dir}/render_traj.mp4"
    echo "${LAST_VIDEO}" > "${LOG_DIR}/${prefix}_video.txt"
    echo "[${dataset}/${scene}] Video export done. ${LAST_VIDEO}"
}

# ---------------------------------------------------------------------------
# DTU Chamfer distance
# ---------------------------------------------------------------------------
run_dtu_chamfer() {
    local dataset="$1" scene="$2" model_path="$3"
    local scan_id chamfer_log chamfer_file eval_dir results_path prefix

    prefix="$(log_prefix "${dataset}" "${scene}")"
    scan_id="$(dtu_scan_id "${scene}")"
    chamfer_log="${LOG_DIR}/${prefix}_chamfer.log"
    chamfer_file="${LOG_DIR}/${prefix}_chamfer.txt"
    eval_dir="${model_path}/dtu_eval"
    results_path="${eval_dir}/scan${scan_id}/results.json"

    LAST_CHAMFER="N/A"

    if [[ ! -d "${DTU_OFFICIAL_ROOT}" ]]; then
        echo "[${dataset}/${scene}] DTU official GT path missing: ${DTU_OFFICIAL_ROOT}"
        echo "[${dataset}/${scene}] Skipping Chamfer distance. Pass --DTU_Official PATH to enable."
        return 0
    fi

    local mesh_path
    mesh_path=$(find "${model_path}" \( -name "fuse*.ply" -o -name "mesh*.ply" \) 2>/dev/null | head -1)
    if [[ -z "${mesh_path}" ]]; then
        echo "[${dataset}/${scene}] No mesh found for DTU Chamfer. Skipping."
        return 0
    fi

    mkdir -p "$(dirname "${results_path}")"
    echo "[${dataset}/${scene}] DTU Chamfer started. Log: ${chamfer_log}"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" eval dtu-mesh
        --pred "${mesh_path}"
        --dtu-root "${DTU_OFFICIAL_ROOT}"
        --scan-id "${scan_id}"
        -o "${results_path}"
    )

    printf '[%s/%s] Command: %s\n\n' "${dataset}" "${scene}" "${cmd[*]}" > "${chamfer_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${chamfer_log}" 2>&1
    local exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] DTU Chamfer FAILED. Last log:"
        tail -n 30 "${chamfer_log}" || true
        return "${exit_code}"
    fi

    read_chamfer_from_json "${results_path}"
    echo "${LAST_CHAMFER}" > "${chamfer_file}"
    echo "[${dataset}/${scene}] DTU Chamfer done. CD=${LAST_CHAMFER}"
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print_final_summary() {
    if [[ "${#SUMMARY_ROWS[@]}" -eq 0 ]]; then return; fi

    echo
    echo "=============================================================================================================="
    echo "  TriBench Summary  (method: ${METHOD_ID})"
    echo "=============================================================================================================="
    echo "  Output root: ${OUTPUT_PATH}/${METHOD_ID}/<dataset>/<scene>"
    echo "  FPS: test-split render frames/s | TrainMem: MiB | TrainTime: s | Metrics: test split"
    echo "=============================================================================================================="
    printf "  %-24s %8s %8s %8s %8s %12s %12s %10s %s\n" \
        "Dataset/Scene" "PSNR" "SSIM" "LPIPS" "FPS" "TrainMem" "TrainTime" "Chamfer" "Video"
    printf "  %-24s %8s %8s %8s %8s %12s %12s %10s %s\n" \
        "------------------------" "--------" "--------" "--------" "--------" "------------" "------------" "----------" "-----"

    local row label psnr ssim lpips fps mem time chamfer video
    for row in "${SUMMARY_ROWS[@]}"; do
        IFS='|' read -r label psnr ssim lpips fps mem time chamfer video <<< "${row}"
        printf "  %-24s %8s %8s %8s %8s %12s %12s %10s %s\n" \
            "${label}" "${psnr}" "${ssim}" "${lpips}" "${fps}" "${mem}" "${time}" "${chamfer}" "${video}"
    done
    echo "=============================================================================================================="
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
        --output_path|--output-dir|--outputs) require_value "$@"; OUTPUT_PATH="$2"; shift 2 ;;
        --log_dir|--log-dir) require_value "$@"; LOG_ROOT="$2"; shift 2 ;;
        --method)            require_value "$@"; METHOD="$2"; shift 2 ;;
        --train_iterations|--iterations|--max-steps) require_value "$@"; TRAIN_ITERATIONS="$2"; shift 2 ;;
        --resolution|-r)     require_value "$@"; RESOLUTION="$2"; shift 2 ;;
        --render_resolution|--render-resolution) require_value "$@"; RENDER_RESOLUTION="$2"; shift 2 ;;
        --eval_every|--eval-every) require_value "$@"; EVAL_EVERY="$2"; shift 2 ;;
        --video_frames|--video-frames) require_value "$@"; VIDEO_FRAMES="$2"; shift 2 ;;
        --video_fps|--video-fps) require_value "$@"; VIDEO_FPS="$2"; shift 2 ;;
        --video_zoom|--video-zoom) require_value "$@"; VIDEO_ZOOM="$2"; shift 2 ;;
        --video_z_variation|--video-z-variation) require_value "$@"; VIDEO_Z_VARIATION="$2"; shift 2 ;;
        --video_z_phase|--video-z-phase) require_value "$@"; VIDEO_Z_PHASE="$2"; shift 2 ;;
        --white_background|--white-background) WHITE_BACKGROUND=1; shift ;;
        --training)          REQUESTED_TRAINING=1; HAS_STAGE_REQUEST=1; shift ;;
        --rendering|--render) REQUESTED_RENDERING=1; HAS_STAGE_REQUEST=1; shift ;;
        --metrics|--eval)    REQUESTED_METRICS=1; HAS_STAGE_REQUEST=1; shift ;;
        --video)             REQUESTED_VIDEO=1; HAS_STAGE_REQUEST=1; shift ;;
        --skip_training|--skip-training) EXPLICIT_SKIP_TRAINING=1; shift ;;
        --skip_rendering|--skip-rendering) EXPLICIT_SKIP_RENDERING=1; shift ;;
        --skip_metrics|--skip-metrics) EXPLICIT_SKIP_METRICS=1; shift ;;
        --skip_video|--skip-video) EXPLICIT_SKIP_VIDEO=1; shift ;;
        --no_eval|--no-eval) EVAL_SPLIT=0; shift ;;
        --python)            require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)
            if [[ "$1" == --* ]]; then
                echo "Unknown option: $1" >&2
                usage
                exit 1
            fi
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${METHOD}" ]]; then
    if [[ "${#POSITIONAL_ARGS[@]}" -lt 3 ]]; then
        usage
        exit 1
    fi
    METHOD="${POSITIONAL_ARGS[0]}"
    GPU_ID="${POSITIONAL_ARGS[$((${#POSITIONAL_ARGS[@]} - 1))]}"
    TARGETS=("${POSITIONAL_ARGS[@]:1:$((${#POSITIONAL_ARGS[@]} - 2))}")
else
    if [[ "${#POSITIONAL_ARGS[@]}" -lt 2 ]]; then
        usage
        exit 1
    fi
    GPU_ID="${POSITIONAL_ARGS[$((${#POSITIONAL_ARGS[@]} - 1))]}"
    TARGETS=("${POSITIONAL_ARGS[@]:0:$((${#POSITIONAL_ARGS[@]} - 1))}")
fi

METHOD_ID="$(canonical_method "${METHOD}")"

# Resolve stage flags. With no explicit stage request, run the full pipeline.
if [[ "${HAS_STAGE_REQUEST}" -eq 1 ]]; then
    SKIP_TRAINING=1
    SKIP_RENDERING=1
    SKIP_METRICS=1
    SKIP_VIDEO=1
    [[ "${REQUESTED_TRAINING}" -eq 1 ]] && SKIP_TRAINING=0
    [[ "${REQUESTED_RENDERING}" -eq 1 ]] && SKIP_RENDERING=0
    [[ "${REQUESTED_METRICS}" -eq 1 ]] && SKIP_METRICS=0
    [[ "${REQUESTED_VIDEO}" -eq 1 ]] && SKIP_VIDEO=0
fi
[[ "${EXPLICIT_SKIP_TRAINING}" -eq 1 ]] && SKIP_TRAINING=1
[[ "${EXPLICIT_SKIP_RENDERING}" -eq 1 ]] && SKIP_RENDERING=1
[[ "${EXPLICIT_SKIP_METRICS}" -eq 1 ]] && SKIP_METRICS=1
[[ "${EXPLICIT_SKIP_VIDEO}" -eq 1 ]] && SKIP_VIDEO=1

if [[ "${SKIP_TRAINING}" -eq 1 && "${SKIP_RENDERING}" -eq 1 && "${SKIP_METRICS}" -eq 1 && "${SKIP_VIDEO}" -eq 1 ]]; then
    echo "No stage selected." >&2
    exit 1
fi

resolve_python_bin
resolve_tb_cmd
expand_targets "${TARGETS[@]}"
mkdir -p "${OUTPUT_PATH}"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for TARGET in "${EXPANDED_TARGETS[@]}"; do
    parse_target "${TARGET}"
    DATASET="$(dataset_label "${PARSED_DATASET}")"
    SCENE="$(basename "${PARSED_SCENE}")"
    SOURCE="$(scene_source "${DATASET}" "${PARSED_SCENE}" "${PARSED_RAW_TARGET}")"
    MODEL_PATH="${OUTPUT_PATH}/${METHOD_ID}/${DATASET}/${SCENE}"
    IMAGE_DIR="$(scene_image_dir "${DATASET}" "${SCENE}")"
    DTYPE="$(scene_dataset_type "${DATASET}")"
    if [[ -n "${LOG_ROOT}" ]]; then
        LOG_DIR="${LOG_ROOT}/${METHOD_ID}/${DATASET}/${SCENE}"
    else
        LOG_DIR="${MODEL_PATH}/logs"
    fi
    PREFIX="$(log_prefix "${DATASET}" "${SCENE}")"

    [[ ! -d "${SOURCE}" ]] && { echo "[${DATASET}/${SCENE}] Dataset missing: ${SOURCE}" >&2; exit 1; }
    mkdir -p "${LOG_DIR}" "${MODEL_PATH}"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "[${DATASET}/${SCENE}] method=${METHOD_ID} gpu=${GPU_ID}"
    echo "[${DATASET}/${SCENE}] source=${SOURCE}"
    echo "[${DATASET}/${SCENE}] output=${MODEL_PATH}"
    echo "[${DATASET}/${SCENE}] logs=${LOG_DIR}"
    echo "--------------------------------------------------------------------------------"

    LAST_TRAIN_MEMORY="$(read_metric_cache "${LOG_DIR}/${PREFIX}_max_mem.txt")"
    LAST_TRAIN_TIME="$(read_metric_cache "${LOG_DIR}/${PREFIX}_train_time.txt")"
    LAST_RENDER_FPS="$(read_metric_cache "${LOG_DIR}/${PREFIX}_render_fps.txt")"
    LAST_CHAMFER="$(read_metric_cache "${LOG_DIR}/${PREFIX}_chamfer.txt")"
    LAST_VIDEO="$(read_metric_cache "${LOG_DIR}/${PREFIX}_video.txt")"
    read_metrics_from_json "${MODEL_PATH}/metrics.json"

    if [[ "${SKIP_TRAINING}" -eq 0 ]]; then
        run_train_with_memory "${DATASET}" "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}"
    fi

    if [[ "${SKIP_RENDERING}" -eq 0 ]]; then
        run_render "${DATASET}" "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}" "${DTYPE}"
    fi

    if [[ "${SKIP_METRICS}" -eq 0 ]]; then
        run_metrics "${DATASET}" "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}" "${DTYPE}"
        if [[ "$(normalize_dataset "${DATASET}")" == "dtu" ]]; then
            run_dtu_chamfer "${DATASET}" "${SCENE}" "${MODEL_PATH}"
        fi
    fi

    if [[ "${SKIP_VIDEO}" -eq 0 ]]; then
        run_video "${DATASET}" "${SCENE}" "${SOURCE}" "${MODEL_PATH}" "${IMAGE_DIR}" "${DTYPE}"
    fi

    echo "[${DATASET}/${SCENE}] Summary: PSNR=${LAST_PSNR} SSIM=${LAST_SSIM} LPIPS=${LAST_LPIPS} FPS=${LAST_RENDER_FPS} Mem=${LAST_TRAIN_MEMORY}MiB Time=${LAST_TRAIN_TIME}s Chamfer=${LAST_CHAMFER} Video=${LAST_VIDEO}"
    SUMMARY_ROWS+=("${DATASET}/${SCENE}|${LAST_PSNR}|${LAST_SSIM}|${LAST_LPIPS}|${LAST_RENDER_FPS}|${LAST_TRAIN_MEMORY}|${LAST_TRAIN_TIME}|${LAST_CHAMFER}|${LAST_VIDEO}")
done

print_final_summary
