#!/usr/bin/env bash
set -euo pipefail

# single_train.sh - config-driven TriBench scene/dataset pipeline.
#
# Primary interface:
#   bash single_train.sh <method> <dataset/scene|dataset/all|all> <gpu_id> [options]
#
# Config lookup:
#   configs/<method>/<dataset>/<scene>.yaml
#
# Pipeline:
#   train -> render configured image split -> compute image metrics
#   -> optional DTU mesh metrics -> export video -> formatted summary

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_ROOT="configs"
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
FORMAT_METRICS="${FORMAT_METRICS:-tools/format_metrics.py}"

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
FORCE_RETRAIN=0
FORCE_RERUN_EXISTING=0

SUMMARY_ROWS=()
SUMMARY_CONFIGS=()
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
  bash $0 triangle_splatting m360/room 0
  bash $0 triangle-splatting tandt/truck 0
  bash $0 triangle-splatting dtu/scan105 0
  bash $0 triangle-splatting mipnerf360/all 0
  bash $0 triangle-splatting all 0

Legacy form still works:
  bash $0 garden 0 --method triangle-splatting

Options:
  --config_root PATH       Config root                (default: ${CONFIG_ROOT})
  --log_dir PATH           Optional shared log root   (default: per-scene logs/)
  --format_metrics PATH    Metrics formatter script   (default: ${FORMAT_METRICS})
  --method NAME            Method for legacy form
  --training               Run training only
  --rendering              Run configured image rendering only
  --metrics                Run metrics only
  --video                  Run video export only
  --skip_training          Skip training
  --skip_rendering         Skip configured image rendering
  --skip_metrics           Skip image metrics
  --skip_video             Skip video export
  --retrain                Delete existing scene output and train again
  --force_retrain          Alias for --retrain
  --rerun_existing         Rerun render/metrics/video/mesh even if outputs exist
  --python PATH            Python executable
  -h, --help               Show this help

Each stage reads configs/<method>/<dataset>/<scene>.yaml. Dataset roots,
image dirs, resolutions, max steps, eval stride, output paths, and video
settings should be set in the YAML config files. When training is selected,
scenes with existing training outputs are skipped by default; pass --retrain
to remove the old scene output directory before training again. Rendering,
metrics, video, and DTU mesh metrics are also skipped when their outputs
already exist; pass --rerun_existing to rebuild those outputs.
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

scene_config_path() {
    local method="$1" dataset="$2" scene="$3"
    echo "${CONFIG_ROOT}/${method}/$(dataset_label "${dataset}")/${scene}.yaml"
}

config_value() {
    local config_file="$1" dotted_key="$2"
    "${PYTHON_BIN}" - "${config_file}" "${dotted_key}" <<'PY'
import sys
from pathlib import Path

from tribench.core.config import Config

cfg = Config.fromfile(sys.argv[1])
value = cfg
for part in sys.argv[2].split("."):
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        print("")
        raise SystemExit
print(value)
PY
}

config_output_dir() {
    local config_file="$1"
    local out_dir
    out_dir="$(config_value "${config_file}" "output.dir")"
    [[ -n "${out_dir}" ]] && echo "${out_dir}" || echo ""
}

first_config_value() {
    local config_file="$1"; shift
    local value key
    for key in "$@"; do
        value="$(config_value "${config_file}" "${key}")"
        if [[ -n "${value}" ]]; then
            echo "${value}"
            return 0
        fi
    done
    return 1
}

path_has_contents() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        [[ -s "${path}" ]]
        return
    fi
    if [[ -d "${path}" ]]; then
        [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
        return
    fi
    return 1
}

training_complete_marker() {
    local config_file="$1" model_path="$2"
    local max_steps checkpoint candidate

    candidate="${model_path}/train_stats.json"
    if path_has_contents "${candidate}"; then
        echo "${candidate}"
        return 0
    fi

    checkpoint="$(config_value "${config_file}" "adapter.checkpoint")"
    if [[ -n "${checkpoint}" && "${checkpoint}" != "${model_path}" ]] && path_has_contents "${checkpoint}"; then
        echo "${checkpoint}"
        return 0
    fi

    max_steps="$(config_value "${config_file}" "trainer.max_steps")"
    if [[ -n "${max_steps}" ]]; then
        for candidate in \
            "${model_path}/ckpt/point_cloud/iteration_${max_steps}" \
            "${model_path}/ckpt/point_cloud/${max_steps}.ply" \
            "${model_path}/ckpt/${max_steps}.ckpt" \
            "${model_path}/mesh/${max_steps}_pcd.ply" \
            "${model_path}/mesh/${max_steps}_mesh.ply"
        do
            if path_has_contents "${candidate}"; then
                echo "${candidate}"
                return 0
            fi
        done
    fi


    return 1
}

render_output_dir() {
    local config_file="$1" model_path="$2"
    local out_dir
    out_dir="$(first_config_value "${config_file}" \
        "render.images.output_dir" \
        "render.images.dir" \
        "render.output_dir" \
        "render.dir" || true)"
    [[ -n "${out_dir}" ]] && echo "${out_dir}" || echo "${model_path}"
}

rendering_complete_marker() {
    local config_file="$1" model_path="$2"
    local out_dir split_dir
    out_dir="$(render_output_dir "${config_file}" "${model_path}")"
    # Unified layout renders into <run_dir>/renders/<split>/{renders,gt}.
    split_dir="$(first_config_value "${config_file}" "render.split" "render.images.split" || true)"
    [[ -n "${split_dir}" ]] || split_dir="test"
    for candidate in "${out_dir}/renders/${split_dir}" "${out_dir}"; do
        if path_has_contents "${candidate}/manifest.json" && path_has_contents "${candidate}/renders"; then
            echo "${candidate}/manifest.json"
            return 0
        fi
        if path_has_contents "${candidate}/renders"; then
            echo "${candidate}/renders"
            return 0
        fi
    done
    return 1
}


metrics_output_file() {
    local config_file="$1" model_path="$2"
    local metrics_file
    metrics_file="$(first_config_value "${config_file}" \
        "eval.output" \
        "eval.metrics_file" \
        "output.metrics_file" || true)"
    [[ -n "${metrics_file}" ]] && echo "${metrics_file}" || echo "${model_path}/metrics.json"
}

metrics_complete_marker() {
    local config_file="$1" model_path="$2"
    local metrics_file
    metrics_file="$(metrics_output_file "${config_file}" "${model_path}")"
    if path_has_contents "${metrics_file}"; then
        echo "${metrics_file}"
        return 0
    fi
    return 1
}

video_output_file() {
    local config_file="$1" model_path="$2"
    local video_dir
    video_dir="$(first_config_value "${config_file}" \
        "render.video.output_dir" \
        "render.video.dir" || true)"
    [[ -n "${video_dir}" ]] || video_dir="${model_path}/video"
    echo "${video_dir}/render_traj.mp4"
}

video_complete_marker() {
    local config_file="$1" model_path="$2"
    local video_file
    video_file="$(video_output_file "${config_file}" "${model_path}")"
    if path_has_contents "${video_file}"; then
        echo "${video_file}"
        return 0
    fi
    return 1
}

mesh_output_file() {
    local config_file="$1" model_path="$2"
    local mesh_file
    mesh_file="$(first_config_value "${config_file}" \
        "mesh.output" \
        "output.mesh_file" || true)"
    [[ -n "${mesh_file}" ]] && echo "${mesh_file}" || echo "${model_path}/mesh.ply"
}

mesh_metrics_output_file() {
    local config_file="$1" model_path="$2"
    local mesh_metrics_file
    mesh_metrics_file="$(first_config_value "${config_file}" \
        "eval.dtu_mesh.output" \
        "eval.mesh.output" \
        "dtu_mesh.output" \
        "mesh_eval.output" \
        "output.mesh_metrics_file" \
        "output.dtu_mesh_metrics_file" || true)"
    [[ -n "${mesh_metrics_file}" ]] && echo "${mesh_metrics_file}" || echo "${model_path}/mesh_metrics.json"
}

mesh_metrics_complete_marker() {
    local config_file="$1" model_path="$2"
    local metrics_file mesh_file
    metrics_file="$(mesh_metrics_output_file "${config_file}" "${model_path}")"
    mesh_file="$(mesh_output_file "${config_file}" "${model_path}")"
    if path_has_contents "${metrics_file}" && path_has_contents "${mesh_file}"; then
        echo "${metrics_file}"
        return 0
    fi
    return 1
}

delete_existing_training_output() {
    local model_path="$1"
    case "${model_path}" in
        ""|"/"|".")
            echo "Refusing to delete unsafe output path: '${model_path}'" >&2
            exit 1
            ;;
    esac
    if [[ -e "${model_path}" ]]; then
        rm -rf -- "${model_path}"
    fi
}

expand_targets() {
    local target dataset scene
    EXPANDED_TARGETS=()
    for target in "$@"; do
        if [[ "${target}" == "all" ]]; then
            for scene in "${MIPNERF360_SCENES[@]}"; do EXPANDED_TARGETS+=("mipnerf360/${scene}"); done
            for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do EXPANDED_TARGETS+=("tandt/${scene}"); done
            for scene in "${DTU_SCENES[@]}"; do EXPANDED_TARGETS+=("dtu/${scene}"); done
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

def fmt_mib(v):
    try:
        return str(int(round(float(v))))
    except Exception:
        return "N/A"

def fmt_seconds(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "N/A"

try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    agg = d.get("aggregate", {})
    inference = d.get("inference", {}) or {}
    training = d.get("training", {}) or {}
    fps = d.get("inference_fps", inference.get("fps"))
    train_mem = d.get("training_peak_gpu_memory_mib", training.get("peak_gpu_memory_mib"))
    train_time = d.get("training_time_s", training.get("total_time_s"))
    print(
        fmt(agg.get("psnr_mean")),
        fmt(agg.get("ssim_mean")),
        fmt(agg.get("lpips_mean")),
        fmt(fps),
        fmt_mib(train_mem),
        fmt_seconds(train_time),
    )
except Exception:
    print("N/A N/A N/A N/A N/A N/A")
PY
)
    local metric_fps metric_train_memory metric_train_time
    read -r LAST_PSNR LAST_SSIM LAST_LPIPS metric_fps metric_train_memory metric_train_time <<< "${values}"
    [[ "${metric_fps}" != "N/A" ]] && LAST_RENDER_FPS="${metric_fps}"
    [[ "${metric_train_memory}" != "N/A" ]] && LAST_TRAIN_MEMORY="${metric_train_memory}"
    [[ "${metric_train_time}" != "N/A" ]] && LAST_TRAIN_TIME="${metric_train_time}"
    return 0
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
# Training
# ---------------------------------------------------------------------------
run_train_config() {
    local dataset="$1" scene="$2" config_file="$3"
    local prefix train_log
    prefix="$(log_prefix "${dataset}" "${scene}")"
    train_log="${LOG_DIR}/${prefix}_train.log"

    rm -f "${train_log}"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" train
        --config "${config_file}"
    )

    echo "[${dataset}/${scene}] Training started. Log: ${train_log}"
    printf '[%s/%s] Command: CUDA_VISIBLE_DEVICES=%s %s\n\n' \
        "${dataset}" "${scene}" "${GPU_ID}" "${cmd[*]}" > "${train_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${train_log}" 2>&1
    local exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] Training FAILED (exit ${exit_code}). Last log:"
        tail -n 30 "${train_log}" || true
        return "${exit_code}"
    fi
    echo "[${dataset}/${scene}] Training done."
}

# ---------------------------------------------------------------------------
# Rendering (train/test splits, timed on test split for FPS)
# ---------------------------------------------------------------------------
run_render_split() {
    local dataset="$1" scene="$2" config_file="$3"
    local prefix render_log
    prefix="$(log_prefix "${dataset}" "${scene}")"
    render_log="${LOG_DIR}/${prefix}_render.log"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" render images
        --config "${config_file}"
    )

    echo "[${dataset}/${scene}] Rendering started. Log: ${render_log}"
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
        echo "[${dataset}/${scene}] Rendering FAILED. Last log:"
        tail -n 30 "${render_log}" || true
        return "${exit_code}"
    fi

    num_frames="$(parse_num_frames "${render_log}")"

    echo "[${dataset}/${scene}] Rendering done. Frames=${num_frames:-?} Time=${elapsed}s"
}

run_render() {
    local dataset="$1" scene="$2" config_file="$3"
    run_render_split "${dataset}" "${scene}" "${config_file}"
}

# ---------------------------------------------------------------------------
# Image quality metrics (test split)
# ---------------------------------------------------------------------------
run_metrics() {
    local dataset="$1" scene="$2" config_file="$3" model_path="$4"
    local metrics_file
    local prefix metrics_log
    metrics_file="$(metrics_output_file "${config_file}" "${model_path}")"
    prefix="$(log_prefix "${dataset}" "${scene}")"
    metrics_log="${LOG_DIR}/${prefix}_metrics.log"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" eval images
        --config "${config_file}"
    )

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
    local dataset="$1" scene="$2" config_file="$3" model_path="$4"
    local video_file
    local prefix video_log
    video_file="$(video_output_file "${config_file}" "${model_path}")"
    prefix="$(log_prefix "${dataset}" "${scene}")"
    video_log="${LOG_DIR}/${prefix}_video.log"
    LAST_VIDEO="N/A"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" render video
        --config "${config_file}"
    )

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

    LAST_VIDEO="${video_file}"
    echo "${LAST_VIDEO}" > "${LOG_DIR}/${prefix}_video.txt"
    echo "[${dataset}/${scene}] Video export done. ${LAST_VIDEO}"
}

# ---------------------------------------------------------------------------
# DTU mesh distance
# ---------------------------------------------------------------------------
run_dtu_chamfer() {
    local dataset="$1" scene="$2" config_file="$3" model_path="$4"
    local chamfer_log chamfer_file results_path prefix

    prefix="$(log_prefix "${dataset}" "${scene}")"
    chamfer_log="${LOG_DIR}/${prefix}_chamfer.log"
    chamfer_file="${LOG_DIR}/${prefix}_chamfer.txt"
    results_path="$(mesh_metrics_output_file "${config_file}" "${model_path}")"

    LAST_CHAMFER="N/A"

    echo "[${dataset}/${scene}] DTU mesh metrics started. Log: ${chamfer_log}"

    local -a cmd=(
        "${TB_CMD_ARR[@]}" eval mesh
        --config "${config_file}"
    )

    printf '[%s/%s] Command: %s\n\n' "${dataset}" "${scene}" "${cmd[*]}" > "${chamfer_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" >> "${chamfer_log}" 2>&1
    local exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${dataset}/${scene}] DTU mesh metrics FAILED. Last log:"
        tail -n 30 "${chamfer_log}" || true
        return "${exit_code}"
    fi

    read_chamfer_from_json "${results_path}"
    echo "${LAST_CHAMFER}" > "${chamfer_file}"
    echo "[${dataset}/${scene}] DTU mesh metrics done. CD=${LAST_CHAMFER}"
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print_final_summary() {
    if [[ "${#SUMMARY_CONFIGS[@]}" -eq 0 ]]; then return; fi

    echo
    echo "=============================================================================================================="
    echo "  TriBench Summary  (method: ${METHOD_ID})"
    echo "=============================================================================================================="
    "${PYTHON_BIN}" "${FORMAT_METRICS}" --config "${SUMMARY_CONFIGS[@]}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then usage; exit 1; fi
[[ "$1" == "-h" || "$1" == "--help" ]] && { usage; exit 0; }

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_root|--config-root) require_value "$@"; CONFIG_ROOT="$2"; shift 2 ;;
        --log_dir|--log-dir) require_value "$@"; LOG_ROOT="$2"; shift 2 ;;
        --format_metrics|--format-metrics) require_value "$@"; FORMAT_METRICS="$2"; shift 2 ;;
        --method)            require_value "$@"; METHOD="$2"; shift 2 ;;
        --training)          REQUESTED_TRAINING=1; HAS_STAGE_REQUEST=1; shift ;;
        --rendering|--render) REQUESTED_RENDERING=1; HAS_STAGE_REQUEST=1; shift ;;
        --metrics|--eval)    REQUESTED_METRICS=1; HAS_STAGE_REQUEST=1; shift ;;
        --video)             REQUESTED_VIDEO=1; HAS_STAGE_REQUEST=1; shift ;;
        --skip_training|--skip-training) EXPLICIT_SKIP_TRAINING=1; shift ;;
        --skip_rendering|--skip-rendering) EXPLICIT_SKIP_RENDERING=1; shift ;;
        --skip_metrics|--skip-metrics) EXPLICIT_SKIP_METRICS=1; shift ;;
        --skip_video|--skip-video) EXPLICIT_SKIP_VIDEO=1; shift ;;
        --retrain|--force_retrain|--force-retrain) FORCE_RETRAIN=1; shift ;;
        --rerun_existing|--rerun-existing|--force_existing|--force-existing) FORCE_RERUN_EXISTING=1; shift ;;
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

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for TARGET in "${EXPANDED_TARGETS[@]}"; do
    parse_target "${TARGET}"
    DATASET="$(dataset_label "${PARSED_DATASET}")"
    SCENE="$(basename "${PARSED_SCENE}")"
    CONFIG_FILE="$(scene_config_path "${METHOD_ID}" "${DATASET}" "${SCENE}")"
    [[ ! -f "${CONFIG_FILE}" ]] && {
        echo "[${DATASET}/${SCENE}] Config missing: ${CONFIG_FILE}" >&2
        exit 1
    }
    MODEL_PATH="$(config_output_dir "${CONFIG_FILE}")"
    [[ -z "${MODEL_PATH}" ]] && {
        echo "[${DATASET}/${SCENE}] Config missing output.dir: ${CONFIG_FILE}" >&2
        exit 1
    }
    if [[ -n "${LOG_ROOT}" ]]; then
        LOG_DIR="${LOG_ROOT}/${METHOD_ID}/${DATASET}/${SCENE}"
    else
        LOG_DIR="${MODEL_PATH}/logs"
    fi
    PREFIX="$(log_prefix "${DATASET}" "${SCENE}")"

    TRAINING_ALREADY_DONE=0
    TRAINING_MARKER=""
    if [[ "${SKIP_TRAINING}" -eq 0 ]] && TRAINING_MARKER="$(training_complete_marker "${CONFIG_FILE}" "${MODEL_PATH}")"; then
        if [[ "${FORCE_RETRAIN}" -eq 1 ]]; then
            echo
            echo "[${DATASET}/${SCENE}] Existing training output found: ${TRAINING_MARKER}"
            echo "[${DATASET}/${SCENE}] --retrain set; deleting ${MODEL_PATH} before training."
            delete_existing_training_output "${MODEL_PATH}"
            TRAINING_MARKER=""
        else
            TRAINING_ALREADY_DONE=1
        fi
    fi

    RENDERING_ALREADY_DONE=0
    RENDERING_MARKER=""
    if [[ "${SKIP_RENDERING}" -eq 0 && "${FORCE_RERUN_EXISTING}" -eq 0 ]] && RENDERING_MARKER="$(rendering_complete_marker "${CONFIG_FILE}" "${MODEL_PATH}")"; then
        RENDERING_ALREADY_DONE=1
    fi

    METRICS_ALREADY_DONE=0
    METRICS_MARKER=""
    if [[ "${SKIP_METRICS}" -eq 0 && "${FORCE_RERUN_EXISTING}" -eq 0 ]] && METRICS_MARKER="$(metrics_complete_marker "${CONFIG_FILE}" "${MODEL_PATH}")"; then
        METRICS_ALREADY_DONE=1
    fi

    MESH_METRICS_ALREADY_DONE=0
    MESH_METRICS_MARKER=""
    if [[ "${SKIP_METRICS}" -eq 0 && "$(normalize_dataset "${DATASET}")" == "dtu" && "${FORCE_RERUN_EXISTING}" -eq 0 ]] && MESH_METRICS_MARKER="$(mesh_metrics_complete_marker "${CONFIG_FILE}" "${MODEL_PATH}")"; then
        MESH_METRICS_ALREADY_DONE=1
    fi

    VIDEO_ALREADY_DONE=0
    VIDEO_MARKER=""
    if [[ "${SKIP_VIDEO}" -eq 0 && "${FORCE_RERUN_EXISTING}" -eq 0 ]] && VIDEO_MARKER="$(video_complete_marker "${CONFIG_FILE}" "${MODEL_PATH}")"; then
        VIDEO_ALREADY_DONE=1
    fi

    mkdir -p "${LOG_DIR}" "${MODEL_PATH}"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "[${DATASET}/${SCENE}] method=${METHOD_ID} gpu=${GPU_ID}"
    echo "[${DATASET}/${SCENE}] config=${CONFIG_FILE}"
    echo "[${DATASET}/${SCENE}] output=${MODEL_PATH}"
    echo "[${DATASET}/${SCENE}] logs=${LOG_DIR}"
    if [[ "${TRAINING_ALREADY_DONE}" -eq 1 ]]; then
        echo "[${DATASET}/${SCENE}] training=skipped existing output (${TRAINING_MARKER})"
    elif [[ "${SKIP_TRAINING}" -eq 0 && "${FORCE_RETRAIN}" -eq 1 ]]; then
        echo "[${DATASET}/${SCENE}] training=retrain enabled"
    fi
    [[ "${RENDERING_ALREADY_DONE}" -eq 1 ]] && echo "[${DATASET}/${SCENE}] rendering=skipped existing output (${RENDERING_MARKER})"
    [[ "${METRICS_ALREADY_DONE}" -eq 1 ]] && echo "[${DATASET}/${SCENE}] metrics=skipped existing output (${METRICS_MARKER})"
    [[ "${MESH_METRICS_ALREADY_DONE}" -eq 1 ]] && echo "[${DATASET}/${SCENE}] mesh_metrics=skipped existing output (${MESH_METRICS_MARKER})"
    [[ "${VIDEO_ALREADY_DONE}" -eq 1 ]] && echo "[${DATASET}/${SCENE}] video=skipped existing output (${VIDEO_MARKER})"
    [[ "${FORCE_RERUN_EXISTING}" -eq 1 ]] && echo "[${DATASET}/${SCENE}] rerun_existing=enabled"
    echo "--------------------------------------------------------------------------------"

    LAST_TRAIN_MEMORY="N/A"
    LAST_TRAIN_TIME="N/A"
    LAST_RENDER_FPS="N/A"
    LAST_CHAMFER="$(read_metric_cache "${LOG_DIR}/${PREFIX}_chamfer.txt")"
    LAST_VIDEO="$(read_metric_cache "${LOG_DIR}/${PREFIX}_video.txt")"
    read_metrics_from_json "$(metrics_output_file "${CONFIG_FILE}" "${MODEL_PATH}")"

    if [[ "${SKIP_TRAINING}" -eq 0 ]]; then
        if [[ "${TRAINING_ALREADY_DONE}" -eq 1 ]]; then
            echo "[${DATASET}/${SCENE}] Training skipped; existing output detected at ${TRAINING_MARKER}. Use --retrain to rebuild."
        else
            run_train_config "${DATASET}" "${SCENE}" "${CONFIG_FILE}"
        fi
    fi

    if [[ "${SKIP_RENDERING}" -eq 0 ]]; then
        if [[ "${RENDERING_ALREADY_DONE}" -eq 1 ]]; then
            echo "[${DATASET}/${SCENE}] Rendering skipped; existing output detected at ${RENDERING_MARKER}. Use --rerun_existing to rebuild."
        else
            run_render "${DATASET}" "${SCENE}" "${CONFIG_FILE}"
        fi
    fi

    if [[ "${SKIP_METRICS}" -eq 0 ]]; then
        if [[ "${METRICS_ALREADY_DONE}" -eq 1 ]]; then
            read_metrics_from_json "${METRICS_MARKER}"
            echo "[${DATASET}/${SCENE}] Metrics skipped; existing output detected at ${METRICS_MARKER}. Use --rerun_existing to rebuild."
        else
            run_metrics "${DATASET}" "${SCENE}" "${CONFIG_FILE}" "${MODEL_PATH}"
        fi
        if [[ "$(normalize_dataset "${DATASET}")" == "dtu" ]]; then
            if [[ "${MESH_METRICS_ALREADY_DONE}" -eq 1 ]]; then
                read_chamfer_from_json "${MESH_METRICS_MARKER}"
                echo "${LAST_CHAMFER}" > "${LOG_DIR}/${PREFIX}_chamfer.txt"
                echo "[${DATASET}/${SCENE}] DTU mesh metrics skipped; existing output detected at ${MESH_METRICS_MARKER}. Use --rerun_existing to rebuild."
            else
                run_dtu_chamfer "${DATASET}" "${SCENE}" "${CONFIG_FILE}" "${MODEL_PATH}"
            fi
        fi
    fi

    if [[ "${SKIP_VIDEO}" -eq 0 ]]; then
        if [[ "${VIDEO_ALREADY_DONE}" -eq 1 ]]; then
            LAST_VIDEO="${VIDEO_MARKER}"
            echo "${LAST_VIDEO}" > "${LOG_DIR}/${PREFIX}_video.txt"
            echo "[${DATASET}/${SCENE}] Video export skipped; existing output detected at ${VIDEO_MARKER}. Use --rerun_existing to rebuild."
        else
            run_video "${DATASET}" "${SCENE}" "${CONFIG_FILE}" "${MODEL_PATH}"
        fi
    fi

    read_metrics_from_json "$(metrics_output_file "${CONFIG_FILE}" "${MODEL_PATH}")"
    echo "[${DATASET}/${SCENE}] Summary: PSNR=${LAST_PSNR} SSIM=${LAST_SSIM} LPIPS=${LAST_LPIPS} FPS=${LAST_RENDER_FPS} Mem=${LAST_TRAIN_MEMORY}MiB Time=${LAST_TRAIN_TIME}s Chamfer=${LAST_CHAMFER} Video=${LAST_VIDEO}"
    SUMMARY_ROWS+=("${DATASET}/${SCENE}|${LAST_PSNR}|${LAST_SSIM}|${LAST_LPIPS}|${LAST_RENDER_FPS}|${LAST_TRAIN_MEMORY}|${LAST_TRAIN_TIME}|${LAST_CHAMFER}|${LAST_VIDEO}")
    SUMMARY_CONFIGS+=("${CONFIG_FILE}")
done

print_final_summary
