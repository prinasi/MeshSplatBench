#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Batch-export feature-preserving Unity-native TriAssets from completed runs.
#
# Usage:
#   bash single_export_unity.sh <method> <target> [target ...] <gpu_id|cpu> [options]
#
# The exporter itself is CPU-only. Pass "cpu" to hide CUDA devices explicitly;
# a numeric GPU id remains accepted for compatibility with single_train.sh.

CONFIG_ROOT="configs"
OUTPUT_SUBDIR="unity_native"
EXPORT_TOPOLOGY="indexed"
METHOD=""
GPU_ID=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi
TB_CMD="${MSBENCH_CMD:-msbench}"
TB_CMD_ARR=()
TB_EXPORT_ARGS=()
FORCE=0
DRY_RUN=0
CONTINUE_ON_ERROR=0
EXPORTED=0
SKIPPED=0

MIPNERF360_OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
MIPNERF360_INDOOR_SCENES=(room counter kitchen bonsai)
MIPNERF360_SCENES=("${MIPNERF360_OUTDOOR_SCENES[@]}" "${MIPNERF360_INDOOR_SCENES[@]}")
TANKS_AND_TEMPLES_SCENES=(truck train)
DTU_SCENES=(scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122)
NERF_SYNTHETIC_SCENES=(chair drums ficus hotdog lego materials mic ship)

usage() {
    cat <<EOF
Usage:
  bash $0 <method> <target> [target ...] <gpu_id|cpu> [options]
  bash $0 <target> [target ...] <gpu_id|cpu> --method <method> [options]

Examples:
  bash $0 triangle-splatting mipnerf360/garden cpu
  bash $0 mesh-splatting mipnerf360/all cpu
  bash $0 2dts mipnerf360/bicycle mipnerf360/garden 0
  bash $0 2dts dtu/scan105 cpu
  bash $0 mipnerf360/garden 0 --method triangle-splatting
  bash $0 triangle-splatting all cpu --continue-on-error

Options:
  --config-root PATH       Config root (default: ${CONFIG_ROOT})
  --output-subdir NAME     Per-run asset directory (default: ${OUTPUT_SUBDIR})
  --export-topology T      MeshSplatting export layout: indexed/mesh or soup/materialized-soup (default: ${EXPORT_TOPOLOGY})
  --method NAME            Method when using the target-first form
  --force                  Replace an existing .triasset package
  --continue-on-error      Export later scenes after an individual failure
  --dry-run                Print resolved commands without exporting
  --python PATH            Python executable for config loading/fallback CLI
  -h, --help               Show this help

The script resolves configs as:
  <config-root>/<method>/<dataset>/<scene>.yaml

Each package is written to:
  <output.dir>/<output-subdir>/<method>.triasset/
EOF
}

require_value() {
    [[ "$#" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
}

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

canonical_method() {
    local name
    name="$(lower "$1")"
    name="${name//_/-}"
    case "${name}" in
        trianglesplatting) echo "triangle-splatting" ;;
        meshsplatting) echo "mesh-splatting" ;;
        2dts|d2ts) echo "2dts" ;;
        diffsoup) echo "diffsoup" ;;
        *) echo "${name}" ;;
    esac
}

dataset_label() {
    local name
    name="$(lower "$1")"
    name="${name//_/-}"
    case "${name}" in
        mipnerf360|mipnerf-360|m360|360|nerf360) echo "mipnerf360" ;;
        tanksandtemples|tanks-and-temples|tandt|tat|tanks) echo "tandt" ;;
        nerfsynthetic|nerf-synthetic|blender|synthetic) echo "nerf_synthetic" ;;
        dtu) echo "dtu" ;;
        *) echo "${name}" ;;
    esac
}

contains_scene() {
    local needle="$1"; shift
    local item
    for item in "$@"; do [[ "${item}" == "${needle}" ]] && return 0; done
    return 1
}

infer_dataset_for_scene() {
    local scene="$1"
    if contains_scene "${scene}" "${MIPNERF360_SCENES[@]}"; then echo "mipnerf360"
    elif contains_scene "${scene}" "${TANKS_AND_TEMPLES_SCENES[@]}"; then echo "tandt"
    elif [[ "${scene}" =~ ^scan[0-9]+$ ]]; then echo "dtu"
    elif contains_scene "${scene}" "${NERF_SYNTHETIC_SCENES[@]}"; then echo "nerf_synthetic"
    else echo ""; fi
}

expand_targets() {
    local target dataset scene
    EXPANDED_TARGETS=()
    for target in "$@"; do
        if [[ "${target}" == "all" ]]; then
            for scene in "${MIPNERF360_SCENES[@]}"; do EXPANDED_TARGETS+=("mipnerf360/${scene}"); done
            for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do EXPANDED_TARGETS+=("tandt/${scene}"); done
            for scene in "${DTU_SCENES[@]}"; do EXPANDED_TARGETS+=("dtu/${scene}"); done
            for scene in "${NERF_SYNTHETIC_SCENES[@]}"; do EXPANDED_TARGETS+=("nerf_synthetic/${scene}"); done
            continue
        fi
        if [[ "${target}" == "dtu" || "${target}" == "dtu/all" || "${target}" == "all_dtu" ]]; then
            for scene in "${DTU_SCENES[@]}"; do EXPANDED_TARGETS+=("dtu/${scene}"); done
            continue
        fi
        if [[ "$(dataset_label "${target}")" == "nerf_synthetic" && "${target}" != */* ]]; then
            for scene in "${NERF_SYNTHETIC_SCENES[@]}"; do EXPANDED_TARGETS+=("nerf_synthetic/${scene}"); done
            continue
        fi
        if [[ "${target}" == */all ]]; then
            dataset="$(dataset_label "${target%%/*}")"
            case "${dataset}" in
                mipnerf360) for scene in "${MIPNERF360_SCENES[@]}"; do EXPANDED_TARGETS+=("mipnerf360/${scene}"); done ;;
                tandt) for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do EXPANDED_TARGETS+=("tandt/${scene}"); done ;;
                dtu) for scene in "${DTU_SCENES[@]}"; do EXPANDED_TARGETS+=("dtu/${scene}"); done ;;
                nerf_synthetic) for scene in "${NERF_SYNTHETIC_SCENES[@]}"; do EXPANDED_TARGETS+=("nerf_synthetic/${scene}"); done ;;
                *) echo "Cannot expand unknown dataset target: ${target}" >&2; exit 1 ;;
            esac
            continue
        fi
        if [[ "${target}" != */* ]]; then
            dataset="$(infer_dataset_for_scene "${target}")"
            [[ -n "${dataset}" ]] || { echo "Cannot infer dataset for scene: ${target}" >&2; exit 1; }
            EXPANDED_TARGETS+=("${dataset}/${target}")
        else
            EXPANDED_TARGETS+=("${target}")
        fi
    done
}

resolve_python_bin() {
    if [[ "${PYTHON_BIN}" == */* ]]; then
        [[ -x "${PYTHON_BIN}" ]] || { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
    else
        PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
        [[ -n "${PYTHON_BIN}" ]] || { echo "Python not found" >&2; exit 1; }
    fi
}

resolve_tb_cmd() {
    if command -v "${TB_CMD}" >/dev/null 2>&1; then
        TB_CMD_ARR=("${TB_CMD}")
        TB_EXPORT_ARGS=(export-unity)
    else
        TB_CMD_ARR=("${PYTHON_BIN}" "${REPO_ROOT}/tools/export_unity_triasset.py")
        TB_EXPORT_ARGS=()
    fi
}

config_value() {
    local config_file="$1" dotted_key="$2"
    "${PYTHON_BIN}" - "${config_file}" "${dotted_key}" <<'PY'
import sys
from msbench.core.config import Config

value = Config.fromfile(sys.argv[1])
for part in sys.argv[2].split("."):
    if not isinstance(value, dict) or part not in value:
        print("")
        raise SystemExit
    value = value[part]
print(value)
PY
}

asset_export_topology_matches() {
    local package="$1" expected="$2"
    "${PYTHON_BIN}" - "${package}" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1]) / "manifest.json"
expected = sys.argv[2]
try:
    rendering = json.loads(manifest.read_text()).get("rendering", {})
except Exception:
    raise SystemExit(1)
actual = rendering.get("export_topology", "indexed")
if expected == "soup":
    raise SystemExit(0 if actual == "materialized-soup" else 1)
raise SystemExit(0 if actual in ("indexed", None) else 1)
PY
}

if [[ "$#" -lt 1 || "${1}" == "-h" || "${1}" == "--help" ]]; then usage; exit 0; fi

POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config-root|--config_root) require_value "$@"; CONFIG_ROOT="$2"; shift 2 ;;
        --output-subdir|--output_subdir) require_value "$@"; OUTPUT_SUBDIR="$2"; shift 2 ;;
        --export-topology|--export_topology) require_value "$@"; EXPORT_TOPOLOGY="$2"; shift 2 ;;
        --method) require_value "$@"; METHOD="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --python) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ -z "${METHOD}" ]]; then
    [[ "${#POSITIONAL[@]}" -ge 3 ]] || { usage; exit 1; }
    METHOD="${POSITIONAL[0]}"
    GPU_ID="${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}"
    TARGETS=("${POSITIONAL[@]:1:$((${#POSITIONAL[@]} - 2))}")
else
    [[ "${#POSITIONAL[@]}" -ge 2 ]] || { usage; exit 1; }
    GPU_ID="${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}"
    TARGETS=("${POSITIONAL[@]:0:$((${#POSITIONAL[@]} - 1))}")
fi
METHOD_ID="$(canonical_method "${METHOD}")"
GPU_ID="$(lower "${GPU_ID}")"
EXPORT_TOPOLOGY="$(lower "${EXPORT_TOPOLOGY}")"
EXPORT_TOPOLOGY="${EXPORT_TOPOLOGY//_/-}"
case "${EXPORT_TOPOLOGY}" in
    indexed|mesh|indexed-mesh) EXPORT_TOPOLOGY="indexed" ;;
    soup|materialized-soup|triangle-soup|deindexed|deindexed-soup) EXPORT_TOPOLOGY="soup" ;;
    *) echo "--export-topology must be indexed/mesh or soup/materialized-soup: ${EXPORT_TOPOLOGY}" >&2; exit 1 ;;
esac
if [[ "${EXPORT_TOPOLOGY}" != "indexed" && "${METHOD_ID}" != "mesh-splatting" ]]; then
    echo "--export-topology ${EXPORT_TOPOLOGY} is only supported for mesh-splatting" >&2
    exit 1
fi
if [[ "${GPU_ID}" != "cpu" && ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid gpu_id: ${GPU_ID}. Expected a non-negative integer or 'cpu'." >&2
    exit 1
fi

resolve_python_bin
resolve_tb_cmd
expand_targets "${TARGETS[@]}"

FAILED=0
for target in "${EXPANDED_TARGETS[@]}"; do
    DATASET="$(dataset_label "${target%%/*}")"
    SCENE="${target#*/}"
    CONFIG_FILE="${CONFIG_ROOT}/${METHOD_ID}/${DATASET}/${SCENE}.yaml"
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[${DATASET}/${SCENE}] Config missing: ${CONFIG_FILE}" >&2
        FAILED=1
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] && continue || break
    fi
    MODEL_PATH="$(config_value "${CONFIG_FILE}" "output.dir")"
    CHECKPOINT="$(config_value "${CONFIG_FILE}" "adapter.checkpoint")"
    BACKGROUND_COLOR="$(config_value "${CONFIG_FILE}" "adapter.render_params.bg_color")"
    if [[ -z "${MODEL_PATH}" ]]; then
        echo "[${DATASET}/${SCENE}] Config missing output.dir: ${CONFIG_FILE}" >&2
        FAILED=1
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] && continue || break
    fi
    [[ -n "${CHECKPOINT}" ]] || CHECKPOINT="${MODEL_PATH}/ckpt"
    OUTPUT="${MODEL_PATH}/${OUTPUT_SUBDIR}/${METHOD_ID}"
    PACKAGE="${OUTPUT}.triasset"
    REPLACE_INCOMPLETE=0
    if [[ -f "${PACKAGE}/manifest.json" && "${FORCE}" -eq 0 ]]; then
        VALIDATOR="${REPO_ROOT}/tools/validate_unity_triasset_cpu.py"
        if "${PYTHON_BIN}" "${VALIDATOR}" "${PACKAGE}" --max-faces 10000 >/dev/null && asset_export_topology_matches "${PACKAGE}" "${EXPORT_TOPOLOGY}"; then
            echo "[${DATASET}/${SCENE}] Valid Unity asset exists; skipping: ${PACKAGE}"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        echo "[${DATASET}/${SCENE}] Re-exporting stale or topology-mismatched Unity asset: ${PACKAGE}"
        REPLACE_INCOMPLETE=1
    fi
    if [[ -e "${PACKAGE}" && ! -f "${PACKAGE}/manifest.json" ]]; then
        # A failed exporter creates the package directory before validating all
        # checkpoint fields.  Let the exporter atomically replace this known
        # incomplete package on the next normal invocation.
        echo "[${DATASET}/${SCENE}] Replacing incomplete Unity asset: ${PACKAGE}"
        REPLACE_INCOMPLETE=1
    fi
    if [[ ! -e "${CHECKPOINT}" ]]; then
        echo "[${DATASET}/${SCENE}] Checkpoint missing: ${CHECKPOINT}" >&2
        FAILED=1
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] && continue || break
    fi

    CMD=("${TB_CMD_ARR[@]}")
    if [[ "${#TB_EXPORT_ARGS[@]}" -gt 0 ]]; then
        CMD+=("${TB_EXPORT_ARGS[@]}")
    fi
    CMD+=(--method "${METHOD_ID}" --checkpoint "${CHECKPOINT}" --output "${OUTPUT}")
    CMD+=(--export-topology "${EXPORT_TOPOLOGY}")
    [[ -n "${BACKGROUND_COLOR}" ]] && CMD+=(--background-color "${BACKGROUND_COLOR}")
    [[ "${FORCE}" -eq 1 || "${REPLACE_INCOMPLETE}" -eq 1 ]] && CMD+=(--force)
    if [[ "${GPU_ID}" == "cpu" ]]; then
        DEVICE_LABEL="cpu (CUDA hidden)"
        CUDA_DEVICE_VALUE=""
    else
        DEVICE_LABEL="gpu=${GPU_ID}"
        CUDA_DEVICE_VALUE="${GPU_ID}"
    fi
    echo "[${DATASET}/${SCENE}] device=${DEVICE_LABEL} config=${CONFIG_FILE}"
    printf '[%s/%s] Command:' "${DATASET}" "${SCENE}"; printf ' %q' "${CMD[@]}"; printf '\n'
    [[ "${DRY_RUN}" -eq 1 ]] && continue
    mkdir -p "${MODEL_PATH}/logs"
    LOG_EXPORT_SUBDIR="${OUTPUT_SUBDIR//[^A-Za-z0-9_.-]/_}"
    if [[ "${EXPORT_TOPOLOGY}" == "indexed" ]]; then
        EXPORT_LOG="${MODEL_PATH}/logs/unity_export_${METHOD_ID}_${LOG_EXPORT_SUBDIR}.log"
    else
        EXPORT_LOG="${MODEL_PATH}/logs/unity_export_${METHOD_ID}_${LOG_EXPORT_SUBDIR}_${EXPORT_TOPOLOGY}.log"
    fi
    if ! CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_VALUE}" "${CMD[@]}" 2>&1 | tee "${EXPORT_LOG}"; then
        echo "[${DATASET}/${SCENE}] Unity asset export failed" >&2
        FAILED=1
        [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] && continue || break
    fi
    EXPORTED=$((EXPORTED + 1))
done

[[ "${FAILED}" -eq 0 ]] || exit 1
echo "Unity export complete: method=${METHOD_ID} exported=${EXPORTED} skipped=${SKIPPED} targets=${#EXPANDED_TARGETS[@]}"
