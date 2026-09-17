#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Render camera trajectory videos in Unity using either method-aware procedural
# splatting or general-purpose Unity Mesh baselines.
# Uses the exact same PCA ellipse camera trajectory as "msbench render video".

CONFIG_ROOT="configs"
ASSET_SUBDIR="unity_native"
METHOD=""
GPU_ID=""
UNITY_BIN="${UNITY:-}"
UNITY_PROJECT_DIR="${PROJECT:-${UNITY_PROJECT:-}}"
DATASETS_ROOT_OVERRIDE="${DATASETS:-}"
OUTPUT_NAME=""
CONDITION="method-aware"
TOPOLOGY="indexed"
INDEXED_MESH_METHOD_AWARE=0
TRAJECTORY_PATH=""
FRAMES=240
FPS=30
ZOOM="1.0"
Z_VARIATION="0.0"
Z_PHASE="0.0"
SPLIT="train"
WRITE_FRAMES=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
REFERENCE_IMAGE_DIR_OVERRIDE=""

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x "/opt/miniconda3/envs/torch/bin/python" ]]; then
    PYTHON_BIN="/opt/miniconda3/envs/torch/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

MIPNERF360_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIPNERF360_INDOOR_SCENES=(room counter kitchen bonsai)
MIPNERF360_SCENES=("${MIPNERF360_OUTDOOR_SCENES[@]}" "${MIPNERF360_INDOOR_SCENES[@]}")
TANKS_AND_TEMPLES_SCENES=(truck train)
DTU_SCENES=(scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122)
NERF_SYNTHETIC_SCENES=(chair drums ficus hotdog lego materials mic ship)

usage() {
    cat <<EOF
Usage:
  bash $0 <method> <dataset/scene|dataset/all|all> [gpu_id] [options]
  bash $0 <target> [target ...] [gpu_id] --method <method> [options]

Examples:
  bash $0 2dts mipnerf360/bicycle 0 --method-aware
  bash $0 triangle-splatting mipnerf360/garden 0 --general-purpose
  bash $0 mesh-splatting mipnerf360/all 0 --frames 120 --fps 30

Options:
  --unity PATH             Unity executable (default: auto-detected or \$UNITY)
  --unity-project PATH     Unity project root (default: unity/ or \$PROJECT)
  --method-aware           Use method-aware Unity renderer (default)
  --general-purpose        Use ordinary Unity Mesh baseline
  --topology T             Mesh layout: indexed or soup (default: indexed)
  --indexed-mesh-method-aware
                           For general-purpose mesh-splatting, use indexed Mesh with method-aware appearance
  --frames N               Number of frames in ellipse trajectory (default: 240)
  --fps N                  Video framerate (default: 30)
  --zoom F                 Focal length multiplier (default: 1.0)
  --z-variation F          Vertical oscillation amplitude (default: 0.0)
  --z-phase F              Vertical oscillation phase [0, 1] (default: 0.0)
  --split S                Dataset split for trajectory (default: train)
  --trajectory PATH        Use existing trajectory JSON instead of computing ellipse
  --write-frames           Keep rendered PNG frames alongside MP4
  --output-name NAME       Subdirectory name (default: unity_method_aware/video or unity_general_purpose/video)
  --config-root PATH       Config root (default: configs)
  --datasets-root PATH     Override dataset root with PATH/<scene>
  --continue-on-error      Keep running subsequent scenes when a scene fails
  --dry-run                Print commands without launching Unity
  -h, --help               Show this help message
EOF
}

require_value() {
    if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Option $1 requires a value" >&2
        usage
        exit 1
    fi
}

canonical_method() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-')" in
        triangle-splatting|trianglesplatting|triangle_splatting|ts) echo "triangle-splatting" ;;
        mesh-splatting|meshsplatting|mesh_splatting|ms) echo "mesh-splatting" ;;
        2dts|d2ts) echo "2dts" ;;
        diffsoup|diff-soup) echo "diffsoup" ;;
        *) echo "$1" ;;
    esac
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method|-m) require_value "$@"; METHOD="$2"; shift 2 ;;
        --unity) require_value "$@"; UNITY_BIN="$2"; shift 2 ;;
        --unity-project|--unity_project) require_value "$@"; UNITY_PROJECT_DIR="$2"; shift 2 ;;
        --datasets-root|--datasets_root) require_value "$@"; DATASETS_ROOT_OVERRIDE="$2"; shift 2 ;;
        --config-root|--config_root) require_value "$@"; CONFIG_ROOT="$2"; shift 2 ;;
        --output-name|--output_name) require_value "$@"; OUTPUT_NAME="$2"; shift 2 ;;
        --method-aware|--method_aware|--method-specific|--method_specific) CONDITION="method-aware"; shift ;;
        --general-purpose|--general_purpose|--standard-mesh|--standard_mesh) CONDITION="general-purpose"; shift ;;
        --topology) require_value "$@"; TOPOLOGY="$2"; shift 2 ;;
        --indexed-mesh-method-aware|--indexed_mesh_method_aware) INDEXED_MESH_METHOD_AWARE=1; shift ;;
        --frames) require_value "$@"; FRAMES="$2"; shift 2 ;;
        --fps) require_value "$@"; FPS="$2"; shift 2 ;;
        --zoom) require_value "$@"; ZOOM="$2"; shift 2 ;;
        --z-variation|--z_variation) require_value "$@"; Z_VARIATION="$2"; shift 2 ;;
        --z-phase|--z_phase) require_value "$@"; Z_PHASE="$2"; shift 2 ;;
        --split) require_value "$@"; SPLIT="$2"; shift 2 ;;
        --trajectory|-t) require_value "$@"; TRAJECTORY_PATH="$2"; shift 2 ;;
        --write-frames|--write_frames) WRITE_FRAMES=1; shift ;;
        --reference-image-dir|--reference_image_dir) require_value "$@"; REFERENCE_IMAGE_DIR_OVERRIDE="$2"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        --dry-run|--dry_run) DRY_RUN=1; shift ;;
        --python) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ -z "${METHOD}" ]]; then
    [[ "${#POSITIONAL[@]}" -ge 2 ]] || { usage; exit 1; }
    METHOD="${POSITIONAL[0]}"
    if [[ "${#POSITIONAL[@]}" -ge 3 && "${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}" =~ ^[0-9]+$ || "${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}" == "cpu" ]]; then
        GPU_ID="${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}"
        TARGETS=("${POSITIONAL[@]:1:$((${#POSITIONAL[@]} - 2))}")
    else
        TARGETS=("${POSITIONAL[@]:1}")
    fi
else
    [[ "${#POSITIONAL[@]}" -ge 1 ]] || { usage; exit 1; }
    if [[ "${#POSITIONAL[@]}" -ge 2 && "${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}" =~ ^[0-9]+$ || "${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}" == "cpu" ]]; then
        GPU_ID="${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}"
        TARGETS=("${POSITIONAL[@]:0:$((${#POSITIONAL[@]} - 1))}")
    else
        TARGETS=("${POSITIONAL[@]}")
    fi
fi

METHOD_ID="$(canonical_method "${METHOD}")"
if [[ -z "${OUTPUT_NAME}" ]]; then
    OUTPUT_NAME="unity_method_aware"
    [[ "${CONDITION}" == "general-purpose" ]] && OUTPUT_NAME="unity_general_purpose"
fi

if [[ -z "${UNITY_BIN}" ]]; then
    UNITY_BIN="$("${PYTHON_BIN}" -c 'try:
    from msbench.unity_video import find_unity_executable
except ImportError:
    from tools.run_unity_triasset_video import find_unity_executable
print(find_unity_executable() or "")')"
fi

if [[ -z "${UNITY_PROJECT_DIR}" ]]; then
    UNITY_PROJECT_DIR="$("${PYTHON_BIN}" -c 'try:
    from msbench.unity_video import find_unity_project
except ImportError:
    from tools.run_unity_triasset_video import find_unity_project
print(find_unity_project() or "")')"
fi

resolve_scenes() {
    local target="$1"
    case "${target}" in
        all)
            for s in "${MIPNERF360_SCENES[@]}"; do echo "mipnerf360/${s}"; done
            for s in "${TANKS_AND_TEMPLES_SCENES[@]}"; do echo "tandt/${s}"; done
            for s in "${DTU_SCENES[@]}"; do echo "dtu/${s}"; done
            for s in "${NERF_SYNTHETIC_SCENES[@]}"; do echo "nerf_synthetic/${s}"; done
            ;;
        mipnerf360/all) for s in "${MIPNERF360_SCENES[@]}"; do echo "mipnerf360/${s}"; done ;;
        tandt/all) for s in "${TANKS_AND_TEMPLES_SCENES[@]}"; do echo "tandt/${s}"; done ;;
        dtu/all) for s in "${DTU_SCENES[@]}"; do echo "dtu/${s}"; done ;;
        nerf_synthetic/all) for s in "${NERF_SYNTHETIC_SCENES[@]}"; do echo "nerf_synthetic/${s}"; done ;;
        *) echo "${target}" ;;
    esac
}

RESOLVED_TARGETS=()
for t in "${TARGETS[@]}"; do
    while IFS= read -r s; do
        [[ -n "${s}" ]] && RESOLVED_TARGETS+=("${s}")
    done < <(resolve_scenes "${t}")
done

echo "==============================================================================="
echo "MeshSplatBench Unity Video Trajectory Renderer"
echo "Method:     ${METHOD_ID}"
echo "Condition:  ${CONDITION} (topology=${TOPOLOGY})"
echo "Targets:    ${#RESOLVED_TARGETS[@]} scene(s)"
echo "Frames:     ${FRAMES} @ ${FPS} fps (zoom=${ZOOM}, z_var=${Z_VARIATION}, z_phase=${Z_PHASE})"
echo "Unity Bin:  ${UNITY_BIN:-<auto>}"
echo "Project:    ${UNITY_PROJECT_DIR:-<auto>}"
echo "==============================================================================="

FAILED=0
for target in "${RESOLVED_TARGETS[@]}"; do
    dset="$(dirname "${target}")"
    scene="$(basename "${target}")"
    cfg="${CONFIG_ROOT}/${METHOD_ID}/${dset}/${scene}.yaml"
    if [[ ! -f "${cfg}" ]]; then
        cfg="${CONFIG_ROOT}/${METHOD_ID}/${dset}.yaml"
    fi

    echo ""
    echo "[${METHOD_ID}/${target}] Preparing video rendering..."

    CMD=("${PYTHON_BIN}" "tools/run_unity_triasset_video.py"
        --method "${METHOD_ID}"
        --frames "${FRAMES}"
        --fps "${FPS}"
        --zoom "${ZOOM}"
        --z-variation "${Z_VARIATION}"
        --z-phase "${Z_PHASE}"
        --split "${SPLIT}"
        --topology "${TOPOLOGY}")

    [[ -n "${UNITY_BIN}" ]] && CMD+=(--unity "${UNITY_BIN}")
    [[ -n "${UNITY_PROJECT_DIR}" ]] && CMD+=(--unity-project "${UNITY_PROJECT_DIR}")
    [[ "${CONDITION}" == "general-purpose" ]] && CMD+=(--general-purpose) || CMD+=(--method-aware)
    [[ "${INDEXED_MESH_METHOD_AWARE}" -eq 1 ]] && CMD+=(--indexed-mesh-method-aware)
    [[ "${WRITE_FRAMES}" -eq 1 ]] && CMD+=(--write-frames)
    [[ -n "${TRAJECTORY_PATH}" ]] && CMD+=(--trajectory "${TRAJECTORY_PATH}")
    [[ -n "${REFERENCE_IMAGE_DIR_OVERRIDE}" ]] && CMD+=(--image-dir "${REFERENCE_IMAGE_DIR_OVERRIDE}")

    if [[ -f "${cfg}" ]]; then
        CMD+=(--config "${cfg}")
    else
        CMD+=(--outputs-root "outputs/${METHOD_ID}/${dset}" --scenes "${scene}")
        if [[ -n "${DATASETS_ROOT_OVERRIDE}" ]]; then
            CMD+=(--dataset "${DATASETS_ROOT_OVERRIDE}/${scene}")
        fi
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY RUN] ${CMD[*]}"
        continue
    fi

    if ! "${CMD[@]}"; then
        echo "[ERROR] Failed video rendering for ${target}" >&2
        FAILED=1
        if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
            exit 1
        fi
    fi
done

if [[ "${FAILED}" -ne 0 ]]; then
    echo ""
    echo "[MeshSplatBench] Finished with errors." >&2
    exit 1
fi

echo ""
echo "[MeshSplatBench] All video trajectories successfully rendered!"
exit 0
