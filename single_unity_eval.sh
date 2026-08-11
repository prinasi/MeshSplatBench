#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Export Unity TriAssets when needed, capture held-out Unity frames, profile
# no-I/O runtime performance, score image metrics, and write paper-ready tables.

CONFIG_ROOT="configs"
ASSET_SUBDIR="unity_native"
EXPORT_TOPOLOGY="indexed"
METHOD=""
GPU_ID=""
UNITY_BIN="${UNITY:-}"
UNITY_PROJECT_DIR="${PROJECT:-${UNITY_PROJECT:-}}"
UNITY_PLAYER_BIN="${UNITY_PLAYER:-}"
UNITY_PLAYER_OUTPUT="${UNITY_PLAYER_OUTPUT:-outputs/unity_player/linux/TriBenchProfilePlayer.x86_64}"
UNITY_PLAYER_TARGET="${UNITY_PLAYER_TARGET:-linux64}"
UNITY_DISPLAY="${UNITY_DISPLAY:-}"
PROFILE_RUNTIME="${UNITY_PROFILE_RUNTIME:-player}"
GPU_TIMING_MIN_FRACTION="0.5"
PLAYER_BATCHMODE=0
SHOW_PLAYER_OUTPUT=0
FORCE_PLAYER_BUILD=0
SKIP_PLAYER_BUILD=0
DATASETS_ROOT_OVERRIDE="${DATASETS:-}"
OUTPUT_NAME="unity_method_aware"
CONDITION="method-aware"
TOPOLOGY="indexed"
INDEXED_MESH_METHOD_AWARE=0
REPORT_DIR=""
NATIVE_RENDER_SUBDIR="renders/test/renders"
NATIVE_SHAPE_POLICY="crop"
REFERENCE_IMAGE_DIR_OVERRIDE=""
PROFILE_RUNS=3
PROFILE_VIEWS=3
PROFILE_WARMUP=60
PROFILE_FRAMES=180
FPS_WARMUP=30
FPS_FRAMES=120
LPIPS_DEVICE=""
LPIPS_TILE=0
WITH_LPIPS=1
SKIP_EXPORT=0
SKIP_CAPTURE=0
SKIP_PROFILE=0
SKIP_METRICS=0
SKIP_UNITY_PATCH=0
FORCE_EXPORT=0
FORCE_CAPTURE=0
FORCE_PROFILE=0
FORCE_METRICS=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
REPORT_NAME="unity_metrics_report"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
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
RESULT_ARGS=()
SCENE_RESULT_COUNT=0
FAILED=0

usage() {
    cat <<EOF
Usage:
  bash $0 <method> <dataset/scene|dataset/all|all> [target ...] <gpu_id|cpu> [options]
  bash $0 <target> [target ...] <gpu_id|cpu> --method <method> [options]

Examples:
  bash $0 2dts mipnerf360/bicycle 0 --unity "\$UNITY" --unity-project "\$PROJECT"
  bash $0 triangle-splatting mipnerf360/all 0 --output-name unity_method_aware_vulkan
  bash $0 mipnerf360/garden 0 --method mesh-splatting --report-only

Options:
  --unity PATH             Unity executable (default: \$UNITY)
  --unity-project PATH     Unity project root (default: \$PROJECT or \$UNITY_PROJECT)
  --datasets-root PATH     Override all dataset roots with PATH/<scene>
  --config-root PATH       Config root (default: ${CONFIG_ROOT})
  --asset-subdir NAME      Per-run asset directory (default: ${ASSET_SUBDIR})
  --export-topology T      MeshSplatting export layout: indexed/mesh or soup/materialized-soup (default: ${EXPORT_TOPOLOGY})
  --output-name NAME       Unity output folder under each scene output.dir
  --method-aware           Use method-aware Unity renderer (default)
  --general-purpose        Use ordinary Unity Mesh baseline
  --topology T             Mesh layout for Unity renderer: indexed/mesh or soup (default: ${TOPOLOGY})
  --indexed-mesh-method-aware
                           With --general-purpose mesh-splatting, use Unity indexed MeshRenderer plus method-aware SH shader
  --profile-runs N         Independent profile runs per scene (default: ${PROFILE_RUNS})
  --profile-views N        Views per profile run (default: ${PROFILE_VIEWS})
  --profile-warmup N       Warmup frames per profiled view (default: ${PROFILE_WARMUP})
  --profile-frames N       Timed frames per profiled view (default: ${PROFILE_FRAMES})
  --fps-warmup N           Warmup frames for image-capture FPS CSV (default: ${FPS_WARMUP})
  --fps-frames N           Timed frames for image-capture FPS CSV (default: ${FPS_FRAMES})
  --profile-runtime R      Profile runtime: player or editor (default: ${PROFILE_RUNTIME})
  --unity-player PATH      Existing standalone Player executable
  --unity-player-output P  Player build executable path (default: ${UNITY_PLAYER_OUTPUT})
  --unity-player-target T  Player build target (default: ${UNITY_PLAYER_TARGET})
  --display DISPLAY        Linux display for Unity Editor/Player (default: \$DISPLAY or :0)
  --force-player-build     Rebuild the standalone Player before profiling
  --skip-player-build      Require --unity-player; do not build a Player
  --player-batchmode       Launch Player with -batchmode; never uses -nographics
  --show-player-output     Print raw Unity Player stdout/stderr instead of saving it to *.console.log
  --gpu-timing-min-fraction F
                           Required GPU timing sample fraction for Player profile (default: ${GPU_TIMING_MIN_FRACTION})
  --lpips-device DEVICE    LPIPS device (default: cuda:0 for numeric GPU, cpu for cpu)
  --lpips-tile N           Non-canonical tiled LPIPS size; 0 means full frame
  --no-lpips               Score PSNR/SSIM only
  --native-render-subdir P Native render directory under output.dir
  --native-shape-policy P  Native fidelity shape policy: crop, strict, or skip (default: ${NATIVE_SHAPE_POLICY})
  --reference-image-dir P  Reference image dir inside each dataset root
  --report-dir PATH        Report output directory
  --report-name NAME       Report file prefix (default: ${REPORT_NAME})
  --skip-export            Do not export missing Unity assets
  --skip-capture           Do not capture Unity PNGs/FPS CSV
  --skip-profile           Do not run Unity no-I/O profiles
  --skip-metrics           Do not compute PSNR/SSIM/LPIPS
  --report-only            Only format already completed scene outputs
  --force                  Rebuild export/capture/profile/metrics stages
  --force-export           Re-export TriAssets
  --force-capture          Re-capture Unity images
  --force-profile          Re-run Unity profiles
  --force-metrics          Recompute image metrics
  --skip-unity-patch       Do not patch legacy Unity project for Linux/Vulkan
  --continue-on-error      Continue with later scenes after a failure
  --dry-run                Print resolved commands without running them
  --python PATH            Python executable
  -h, --help               Show this help

This wrapper currently requires COLMAP sparse binaries for Unity camera replay.
It will skip/fail early on DTU or synthetic layouts that do not provide
sparse/0/cameras.bin and sparse/0/images.bin.
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
        nerfsynthetic|nerf-synthetic|nerf_synthetic|blender|synthetic) echo "nerf_synthetic" ;;
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
            EXPANDED_TARGETS+=("$(dataset_label "${target%%/*}")/${target#*/}")
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

config_value() {
    local config_file="$1" dotted_key="$2"
    "${PYTHON_BIN}" - "${config_file}" "${dotted_key}" <<'PY'
import sys
from tribench.core.config import Config

value = Config.fromfile(sys.argv[1])
for part in sys.argv[2].split("."):
    if not isinstance(value, dict) or part not in value:
        print("")
        raise SystemExit
    value = value[part]
print(value)
PY
}

abs_path() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pathlib import Path
value = sys.argv[1]
print(Path(value).expanduser().resolve() if value else "")
PY
}

has_images() {
    local path="$1"
    [[ -d "${path}" ]] || return 1
    [[ -n "$(find "${path}" -maxdepth 2 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print -quit)" ]]
}

capture_complete() {
    local result_dir="$1"
    [[ -f "${result_dir}/fps_per_test_view.csv" ]] || return 1
    has_images "${result_dir}"
}

metrics_complete() {
    local path="$1"
    "${PYTHON_BIN}" - "${path}" "${WITH_LPIPS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with_lpips = sys.argv[2] == "1"
if not path.is_file():
    raise SystemExit(1)
try:
    test = json.loads(path.read_text()).get("test", {})
except Exception:
    raise SystemExit(1)
required = {"psnr", "ssim"}
if with_lpips:
    required.add("lpips_vgg")
raise SystemExit(0 if required.issubset(test) else 1)
PY
}

profile_complete() {
    local path="$1" require_gpu="$2"
    "${PYTHON_BIN}" - "${path}" "${PROFILE_VIEWS}" "${PROFILE_FRAMES}" "${require_gpu}" "${GPU_TIMING_MIN_FRACTION}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_views = int(sys.argv[2])
expected_frames = int(sys.argv[3])
require_gpu = sys.argv[4] == "1"
gpu_fraction = float(sys.argv[5])
if not path.is_file():
    raise SystemExit(1)
try:
    views = json.loads(path.read_text()).get("views", [])
except Exception:
    raise SystemExit(1)
if len(views) != expected_views:
    raise SystemExit(1)
samples = sum(len(view.get("cpu_frame_samples_ms", [])) for view in views)
expected_samples = expected_views * expected_frames
if samples < expected_samples:
    raise SystemExit(1)
gpu_samples = sum(
    1
    for view in views
    for value in view.get("gpu_frame_samples_ms", [])
    if isinstance(value, (int, float)) and 0.01 < value < 1000.0
)
if require_gpu and gpu_samples < expected_samples * gpu_fraction:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

resolve_reference_image_dir() {
    local dataset="$1" scene="$2" dataset_path="$3" image_dir="$4" resolution="$5" override="$6"
    local candidates=()
    local candidate existing seen
    local SEEN_CANDIDATES=()
    if [[ -n "${override}" ]]; then
        printf '%s\n' "${override}"
        return 0
    fi
    if [[ "${resolution}" =~ ^[2-9][0-9]*$ && -n "${image_dir}" ]]; then
        if [[ "${image_dir}" == *_"${resolution}" ]]; then
            candidates+=("${image_dir}")
        else
            candidates+=("${image_dir}_${resolution}" "${image_dir}")
        fi
    elif [[ -n "${image_dir}" ]]; then
        candidates+=("${image_dir}")
    fi
    if [[ "${dataset}" == "mipnerf360" ]]; then
        if contains_scene "${scene}" "${MIPNERF360_INDOOR_SCENES[@]}"; then
            candidates+=(images_2 images)
        else
            candidates+=(images_4 images)
        fi
    fi
    candidates+=(images)

    for candidate in "${candidates[@]}"; do
        seen=0
        for existing in "${SEEN_CANDIDATES[@]}"; do
            [[ "${existing}" == "${candidate}" ]] && seen=1 && break
        done
        [[ "${seen}" -eq 1 ]] && continue
        SEEN_CANDIDATES+=("${candidate}")
        if has_images "${dataset_path}/${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    printf '%s\n' "${image_dir}"
}

run_command() {
    local label="$1"; shift
    printf '[%s] Command:' "${label}"; printf ' %q' "$@"; printf '\n'
    [[ "${DRY_RUN}" -eq 0 ]] || return 0
    "$@"
}

handle_failure() {
    local label="$1" message="$2"
    echo "[${label}] ${message}" >&2
    FAILED=1
    [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]
}

positive_int() {
    local name="$1" value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer: ${value}" >&2
        exit 1
    fi
}

if [[ "$#" -lt 1 || "${1}" == "-h" || "${1}" == "--help" ]]; then usage; exit 0; fi

POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --unity) require_value "$@"; UNITY_BIN="$2"; shift 2 ;;
        --unity-project|--unity_project|--project) require_value "$@"; UNITY_PROJECT_DIR="$2"; shift 2 ;;
        --datasets-root|--datasets_root) require_value "$@"; DATASETS_ROOT_OVERRIDE="$2"; shift 2 ;;
        --config-root|--config_root) require_value "$@"; CONFIG_ROOT="$2"; shift 2 ;;
        --asset-subdir|--asset_subdir|--output-subdir|--output_subdir) require_value "$@"; ASSET_SUBDIR="$2"; shift 2 ;;
        --export-topology|--export_topology) require_value "$@"; EXPORT_TOPOLOGY="$2"; shift 2 ;;
        --output-name|--output_name) require_value "$@"; OUTPUT_NAME="$2"; shift 2 ;;
        --method) require_value "$@"; METHOD="$2"; shift 2 ;;
        --method-aware|--method_aware|--method-specific|--method_specific) CONDITION="method-aware"; shift ;;
        --general-purpose|--general_purpose|--standard-mesh|--standard_mesh) CONDITION="general-purpose"; shift ;;
        --topology|--mesh-topology|--mesh_topology) require_value "$@"; TOPOLOGY="$2"; shift 2 ;;
        --indexed-mesh-method-aware|--indexed_mesh_method_aware) INDEXED_MESH_METHOD_AWARE=1; shift ;;
        --profile-runs|--profile_runs) require_value "$@"; PROFILE_RUNS="$2"; shift 2 ;;
        --profile-views|--profile_views) require_value "$@"; PROFILE_VIEWS="$2"; shift 2 ;;
        --profile-warmup|--profile_warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
        --profile-frames|--profile_frames) require_value "$@"; PROFILE_FRAMES="$2"; shift 2 ;;
        --fps-warmup|--fps_warmup) require_value "$@"; FPS_WARMUP="$2"; shift 2 ;;
        --fps-frames|--fps_frames) require_value "$@"; FPS_FRAMES="$2"; shift 2 ;;
        --profile-runtime|--profile_runtime) require_value "$@"; PROFILE_RUNTIME="$2"; shift 2 ;;
        --unity-player|--unity_player) require_value "$@"; UNITY_PLAYER_BIN="$2"; shift 2 ;;
        --unity-player-output|--unity_player_output) require_value "$@"; UNITY_PLAYER_OUTPUT="$2"; shift 2 ;;
        --unity-player-target|--unity_player_target) require_value "$@"; UNITY_PLAYER_TARGET="$2"; shift 2 ;;
        --display) require_value "$@"; UNITY_DISPLAY="$2"; shift 2 ;;
        --force-player-build|--force_player_build) FORCE_PLAYER_BUILD=1; shift ;;
        --skip-player-build|--skip_player_build) SKIP_PLAYER_BUILD=1; shift ;;
        --player-batchmode|--player_batchmode) PLAYER_BATCHMODE=1; shift ;;
        --show-player-output|--show_player_output) SHOW_PLAYER_OUTPUT=1; shift ;;
        --gpu-timing-min-fraction|--gpu_timing_min_fraction) require_value "$@"; GPU_TIMING_MIN_FRACTION="$2"; shift 2 ;;
        --lpips-device|--lpips_device) require_value "$@"; LPIPS_DEVICE="$2"; shift 2 ;;
        --lpips-tile|--lpips_tile) require_value "$@"; LPIPS_TILE="$2"; shift 2 ;;
        --no-lpips|--skip-lpips) WITH_LPIPS=0; shift ;;
        --native-render-subdir|--native_render_subdir) require_value "$@"; NATIVE_RENDER_SUBDIR="$2"; shift 2 ;;
        --native-shape-policy|--native_shape_policy) require_value "$@"; NATIVE_SHAPE_POLICY="$2"; shift 2 ;;
        --reference-image-dir|--reference_image_dir) require_value "$@"; REFERENCE_IMAGE_DIR_OVERRIDE="$2"; shift 2 ;;
        --report-dir|--report_dir) require_value "$@"; REPORT_DIR="$2"; shift 2 ;;
        --report-name|--report_name) require_value "$@"; REPORT_NAME="$2"; shift 2 ;;
        --skip-export|--skip_export) SKIP_EXPORT=1; shift ;;
        --skip-capture|--skip_capture) SKIP_CAPTURE=1; shift ;;
        --skip-profile|--skip_profile) SKIP_PROFILE=1; shift ;;
        --skip-metrics|--skip_metrics) SKIP_METRICS=1; shift ;;
        --report-only|--report_only) SKIP_EXPORT=1; SKIP_CAPTURE=1; SKIP_PROFILE=1; SKIP_METRICS=1; shift ;;
        --force) FORCE_EXPORT=1; FORCE_CAPTURE=1; FORCE_PROFILE=1; FORCE_METRICS=1; shift ;;
        --force-export|--force_export) FORCE_EXPORT=1; shift ;;
        --force-capture|--force_capture) FORCE_CAPTURE=1; shift ;;
        --force-profile|--force_profile) FORCE_PROFILE=1; shift ;;
        --force-metrics|--force_metrics) FORCE_METRICS=1; shift ;;
        --skip-unity-patch|--skip_unity_patch) SKIP_UNITY_PATCH=1; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        --dry-run|--dry_run) DRY_RUN=1; shift ;;
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
PROFILE_RUNTIME="$(lower "${PROFILE_RUNTIME}")"
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
TOPOLOGY="$(lower "${TOPOLOGY}")"
TOPOLOGY="${TOPOLOGY//_/-}"
case "${TOPOLOGY}" in
    indexed|mesh) TOPOLOGY="indexed" ;;
    soup|deindexed|deindexed-soup|triangle-soup) TOPOLOGY="soup" ;;
    *) echo "--topology must be indexed/mesh or soup: ${TOPOLOGY}" >&2; exit 1 ;;
esac
if [[ "${GPU_ID}" != "cpu" && ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid gpu_id: ${GPU_ID}. Expected a non-negative integer or 'cpu'." >&2
    exit 1
fi
case "${PROFILE_RUNTIME}" in
    player|standalone|standalone-player) PROFILE_RUNTIME="player" ;;
    editor|editor-development-only) PROFILE_RUNTIME="editor" ;;
    *) echo "--profile-runtime must be player or editor: ${PROFILE_RUNTIME}" >&2; exit 1 ;;
esac
positive_int "--profile-runs" "${PROFILE_RUNS}"
positive_int "--profile-views" "${PROFILE_VIEWS}"
positive_int "--profile-warmup" "${PROFILE_WARMUP}"
positive_int "--profile-frames" "${PROFILE_FRAMES}"
positive_int "--fps-warmup" "${FPS_WARMUP}"
positive_int "--fps-frames" "${FPS_FRAMES}"
if [[ ! "${LPIPS_TILE}" =~ ^[0-9]+$ ]]; then
    echo "--lpips-tile must be a non-negative integer: ${LPIPS_TILE}" >&2
    exit 1
fi
case "${NATIVE_SHAPE_POLICY}" in
    strict|crop|skip) ;;
    *) echo "--native-shape-policy must be strict, crop, or skip: ${NATIVE_SHAPE_POLICY}" >&2; exit 1 ;;
esac

resolve_python_bin
if ! "${PYTHON_BIN}" - "${GPU_TIMING_MIN_FRACTION}" <<'PY'
import sys
value = float(sys.argv[1])
if not 0.0 < value <= 1.0:
    raise SystemExit(1)
PY
then
    echo "--gpu-timing-min-fraction must be in (0, 1]: ${GPU_TIMING_MIN_FRACTION}" >&2
    exit 1
fi
expand_targets "${TARGETS[@]}"

if [[ "${GPU_ID}" == "cpu" ]]; then
    CUDA_DEVICE_VALUE=""
    [[ -n "${LPIPS_DEVICE}" ]] || LPIPS_DEVICE="cpu"
else
    CUDA_DEVICE_VALUE="${GPU_ID}"
    [[ -n "${LPIPS_DEVICE}" ]] || LPIPS_DEVICE="cuda:0"
fi

if [[ -z "${REPORT_DIR}" ]]; then
    REPORT_DIR="outputs/unity_reports/${METHOD_ID}/${OUTPUT_NAME}"
fi

NEEDS_UNITY=0
[[ "${SKIP_CAPTURE}" -eq 0 || "${SKIP_PROFILE}" -eq 0 ]] && NEEDS_UNITY=1
if [[ "${NEEDS_UNITY}" -eq 1 ]]; then
    [[ -n "${UNITY_BIN}" ]] || { echo "Unity executable is required; pass --unity or set UNITY." >&2; exit 1; }
    [[ -n "${UNITY_PROJECT_DIR}" ]] || { echo "Unity project is required; pass --unity-project or set PROJECT." >&2; exit 1; }
    UNITY_BIN="$(abs_path "${UNITY_BIN}")"
    UNITY_PROJECT_DIR="$(abs_path "${UNITY_PROJECT_DIR}")"
fi

if [[ "${SKIP_EXPORT}" -eq 0 ]]; then
    EXPORT_CMD=("bash" "${REPO_ROOT}/single_export_unity.sh" "${METHOD_ID}")
    EXPORT_CMD+=("${EXPANDED_TARGETS[@]}" "${GPU_ID}")
    EXPORT_CMD+=(--config-root "${CONFIG_ROOT}" --output-subdir "${ASSET_SUBDIR}" --python "${PYTHON_BIN}")
    EXPORT_CMD+=(--export-topology "${EXPORT_TOPOLOGY}")
    [[ "${FORCE_EXPORT}" -eq 1 ]] && EXPORT_CMD+=(--force)
    [[ "${CONTINUE_ON_ERROR}" -eq 1 ]] && EXPORT_CMD+=(--continue-on-error)
    [[ "${DRY_RUN}" -eq 1 ]] && EXPORT_CMD+=(--dry-run)
    if ! run_command "export" "${EXPORT_CMD[@]}"; then
        echo "[export] Unity asset export failed" >&2
        exit 1
    fi
fi

if [[ "${NEEDS_UNITY}" -eq 1 && "${SKIP_UNITY_PATCH}" -eq 0 && "$(uname -s)" == "Linux" ]]; then
    PATCH_CMD=("${PYTHON_BIN}" "tools/patch_legacy_unity_vulkan.py" --unity-project "${UNITY_PROJECT_DIR}")
    if ! run_command "unity-patch" "${PATCH_CMD[@]}"; then
        echo "[unity-patch] Vulkan compatibility patch failed" >&2
        exit 1
    fi
fi

REQUIRE_GPU_TIMING=0
PROFILE_RUNTIME_LABEL="editor-development-only"
if [[ "${PROFILE_RUNTIME}" == "player" ]]; then
    PROFILE_RUNTIME_LABEL="standalone-player"
fi
if [[ "${SKIP_PROFILE}" -eq 0 && "${PROFILE_RUNTIME}" == "player" ]]; then
    REQUIRE_GPU_TIMING=1
    REQUIRES_CURRENT_PLAYER_BUILD=0
    PLAYER_REBUILD_REASON=""
    PLAYER_REBUILD_SOURCES=()
    if [[ "${METHOD_ID}" == "mesh-splatting" && "${TOPOLOGY}" == "soup" && "${SKIP_PLAYER_BUILD}" -eq 0 ]]; then
        REQUIRES_CURRENT_PLAYER_BUILD=1
        PLAYER_REBUILD_REASON="MeshSplatting shader-level soup topology"
        PLAYER_REBUILD_SOURCES+=("${UNITY_PROJECT_DIR}/Assets/TriBench/Scripts/MethodSpecificSplatRenderer.cs")
    fi
    if [[ "${METHOD_ID}" == "mesh-splatting" && "${INDEXED_MESH_METHOD_AWARE}" -eq 1 && "${SKIP_PLAYER_BUILD}" -eq 0 ]]; then
        REQUIRES_CURRENT_PLAYER_BUILD=1
        PLAYER_REBUILD_REASON="MeshSplatting indexed MeshRenderer path"
        PLAYER_REBUILD_SOURCES+=(
            "${UNITY_PROJECT_DIR}/Assets/TriBench/Scripts/StandardMeshTriAssetRenderer.cs"
            "${UNITY_PROJECT_DIR}/Assets/TriBench/Shaders/MeshSplatIndexedMesh.shader"
            "${UNITY_PROJECT_DIR}/Assets/TriBench/Resources/MeshSplatIndexedMesh.shader"
        )
    fi
    if [[ -n "${UNITY_PLAYER_BIN}" ]]; then
        UNITY_PLAYER_BIN="$(abs_path "${UNITY_PLAYER_BIN}")"
    else
        UNITY_PLAYER_BIN="$(abs_path "${UNITY_PLAYER_OUTPUT}")"
    fi
    if [[ "${SKIP_PLAYER_BUILD}" -eq 0 && "${REQUIRES_CURRENT_PLAYER_BUILD}" -eq 1 && "${FORCE_PLAYER_BUILD}" -eq 0 && -x "${UNITY_PLAYER_BIN}" ]]; then
        PLAYER_BUILD_STALE=0
        for source_path in "${PLAYER_REBUILD_SOURCES[@]}"; do
            if [[ -e "${source_path}" && "${source_path}" -nt "${UNITY_PLAYER_BIN}" ]]; then
                PLAYER_BUILD_STALE=1
                break
            fi
        done
        if [[ "${PLAYER_BUILD_STALE}" -eq 1 ]]; then
            echo "[unity-player-build] ${PLAYER_REBUILD_REASON} source is newer than the Player; forcing rebuild."
            FORCE_PLAYER_BUILD=1
        else
            echo "[unity-player-build] Existing standalone Player is current for ${PLAYER_REBUILD_REASON}; rebuild not forced."
        fi
    fi
    if [[ "${SKIP_PLAYER_BUILD}" -eq 0 ]]; then
        if [[ "${FORCE_PLAYER_BUILD}" -eq 1 || ! -x "${UNITY_PLAYER_BIN}" ]]; then
            BUILD_PLAYER_CMD=("${PYTHON_BIN}" "tools/build_unity_profile_player.py"
                --unity "${UNITY_BIN}"
                --unity-project "${UNITY_PROJECT_DIR}"
                --output "${UNITY_PLAYER_BIN}"
                --target "${UNITY_PLAYER_TARGET}")
            [[ -n "${UNITY_DISPLAY}" ]] && BUILD_PLAYER_CMD+=(--display "${UNITY_DISPLAY}")
            [[ "${FORCE_PLAYER_BUILD}" -eq 1 ]] && BUILD_PLAYER_CMD+=(--force)
            if ! run_command "unity-player-build" "${BUILD_PLAYER_CMD[@]}"; then
                echo "[unity-player-build] Standalone Player build failed" >&2
                exit 1
            fi
        else
            echo "[unity-player-build] Existing standalone Player found; skipping: ${UNITY_PLAYER_BIN}"
        fi
    fi
    if [[ "${DRY_RUN}" -eq 0 && ! -x "${UNITY_PLAYER_BIN}" ]]; then
        echo "Standalone Unity Player is required for GPU profile: ${UNITY_PLAYER_BIN}" >&2
        exit 1
    fi
fi

for target in "${EXPANDED_TARGETS[@]}"; do
    DATASET="$(dataset_label "${target%%/*}")"
    SCENE="${target#*/}"
    LABEL="${DATASET}/${SCENE}"
    CONFIG_FILE="${CONFIG_ROOT}/${METHOD_ID}/${DATASET}/${SCENE}.yaml"
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        if handle_failure "${LABEL}" "Config missing: ${CONFIG_FILE}"; then continue; fi
        break
    fi

    MODEL_PATH="$(config_value "${CONFIG_FILE}" "output.dir")"
    DATASET_PATH="$(config_value "${CONFIG_FILE}" "dataset.root")"
    IMAGE_DIR="$(config_value "${CONFIG_FILE}" "dataset.image_dir")"
    RESOLUTION="$(config_value "${CONFIG_FILE}" "dataset.resolution")"
    if [[ -z "${MODEL_PATH}" ]]; then
        if handle_failure "${LABEL}" "Config missing output.dir"; then continue; fi
        break
    fi
    if [[ -z "${DATASET_PATH}" ]]; then
        if handle_failure "${LABEL}" "Config missing dataset.root"; then continue; fi
        break
    fi
    [[ -n "${IMAGE_DIR}" ]] || IMAGE_DIR="images"
    [[ -n "${RESOLUTION}" ]] || RESOLUTION="1"

    MODEL_PATH="$(abs_path "${MODEL_PATH}")"
    if [[ -n "${DATASETS_ROOT_OVERRIDE}" ]]; then
        DATASET_PATH="$(abs_path "${DATASETS_ROOT_OVERRIDE}")/${SCENE}"
    else
        DATASET_PATH="$(abs_path "${DATASET_PATH}")"
    fi
    RESULT_DIR="${MODEL_PATH}/${OUTPUT_NAME}"
    TRI_ASSET="${MODEL_PATH}/${ASSET_SUBDIR}/${METHOD_ID}.triasset"
    REF_IMAGE_DIR="$(resolve_reference_image_dir "${DATASET}" "${SCENE}" "${DATASET_PATH}" "${IMAGE_DIR}" "${RESOLUTION}" "${REFERENCE_IMAGE_DIR_OVERRIDE}")"
    REF_PATH="${DATASET_PATH}/${REF_IMAGE_DIR}"
    if [[ "${REF_IMAGE_DIR}" != "${REFERENCE_IMAGE_DIR_OVERRIDE:-${IMAGE_DIR}}" ]]; then
        echo "[${LABEL}] Reference image dir resolved to ${REF_IMAGE_DIR} (config requested ${IMAGE_DIR})"
    fi

    if [[ "${DRY_RUN}" -eq 0 && ! -f "${TRI_ASSET}/manifest.json" ]]; then
        if handle_failure "${LABEL}" "Unity asset missing after export stage: ${TRI_ASSET}"; then continue; fi
        break
    fi
    if [[ ! -f "${DATASET_PATH}/sparse/0/cameras.bin" || ! -f "${DATASET_PATH}/sparse/0/images.bin" ]]; then
        if handle_failure "${LABEL}" "Unity camera replay requires COLMAP sparse/0/cameras.bin and images.bin: ${DATASET_PATH}"; then continue; fi
        break
    fi
    if ! has_images "${REF_PATH}"; then
        if handle_failure "${LABEL}" "Reference images missing: ${REF_PATH}"; then continue; fi
        break
    fi

    UNITY_OUTPUTS_ROOT="$(dirname "${MODEL_PATH}")"
    UNITY_DATASETS_ROOT="$(dirname "${DATASET_PATH}")"
    UNITY_SCENE_NAME="$(basename "${MODEL_PATH}")"
    CONDITION_ARG="--method-aware"
    [[ "${CONDITION}" == "general-purpose" ]] && CONDITION_ARG="--general-purpose"

    if [[ "${SKIP_CAPTURE}" -eq 0 ]]; then
        if [[ "${FORCE_CAPTURE}" -eq 1 ]] || ! capture_complete "${RESULT_DIR}"; then
            CAPTURE_CMD=("${PYTHON_BIN}" "tools/run_unity_triasset_eval.py"
                --unity "${UNITY_BIN}"
                --unity-project "${UNITY_PROJECT_DIR}"
                --datasets-root "${UNITY_DATASETS_ROOT}"
                --outputs-root "${UNITY_OUTPUTS_ROOT}"
                --method "${METHOD_ID}"
                --asset-subdir "${ASSET_SUBDIR}"
                --output-name "${OUTPUT_NAME}"
                --scenes "${UNITY_SCENE_NAME}"
                --reference-image-dir "${REF_IMAGE_DIR}"
                --test-only
                --fps-warmup "${FPS_WARMUP}"
                --fps-frames "${FPS_FRAMES}"
                --topology "${TOPOLOGY}"
                --log-name "unity_editor.log"
                "${CONDITION_ARG}")
            [[ "${INDEXED_MESH_METHOD_AWARE}" -eq 1 ]] && CAPTURE_CMD+=(--indexed-mesh-method-aware)
            if ! run_command "${LABEL}:capture" "${CAPTURE_CMD[@]}"; then
                if handle_failure "${LABEL}" "Unity capture failed"; then continue; fi
                break
            fi
        else
            echo "[${LABEL}] Capture complete; skipping: ${RESULT_DIR}"
        fi
    fi

    if [[ "${SKIP_PROFILE}" -eq 0 ]]; then
        for run in $(seq 1 "${PROFILE_RUNS}"); do
            PROFILE_JSON="${RESULT_DIR}/runtime_profile_run_$(printf '%02d' "${run}").json"
            if [[ "${FORCE_PROFILE}" -eq 1 ]] || ! profile_complete "${PROFILE_JSON}" "${REQUIRE_GPU_TIMING}"; then
                if [[ "${PROFILE_RUNTIME}" == "player" ]]; then
                    PROFILE_CMD=("${PYTHON_BIN}" "tools/run_unity_triasset_player_profile.py"
                        --player "${UNITY_PLAYER_BIN}"
                        --datasets-root "${UNITY_DATASETS_ROOT}"
                        --outputs-root "${UNITY_OUTPUTS_ROOT}"
                        --method "${METHOD_ID}"
                        --asset-subdir "${ASSET_SUBDIR}"
                        --output-name "${OUTPUT_NAME}"
                        --scenes "${UNITY_SCENE_NAME}"
                        --reference-image-dir "${REF_IMAGE_DIR}"
                        --profile-run "${run}"
                        --profile-views "${PROFILE_VIEWS}"
                        --profile-warmup "${PROFILE_WARMUP}"
                        --profile-frames "${PROFILE_FRAMES}"
                        --topology "${TOPOLOGY}"
                        --log-name "unity_player_profile_run_$(printf '%02d' "${run}").log"
                        --require-gpu-timing
                        --gpu-timing-min-fraction "${GPU_TIMING_MIN_FRACTION}"
                        "${CONDITION_ARG}")
                    [[ "${INDEXED_MESH_METHOD_AWARE}" -eq 1 ]] && PROFILE_CMD+=(--indexed-mesh-method-aware)
                    [[ -n "${UNITY_DISPLAY}" ]] && PROFILE_CMD+=(--display "${UNITY_DISPLAY}")
                    [[ "${PLAYER_BATCHMODE}" -eq 1 ]] && PROFILE_CMD+=(--batchmode)
                    [[ "${SHOW_PLAYER_OUTPUT}" -eq 1 ]] && PROFILE_CMD+=(--show-console-output)
                else
                    PROFILE_CMD=("${PYTHON_BIN}" "tools/run_unity_triasset_eval.py"
                        --unity "${UNITY_BIN}"
                        --unity-project "${UNITY_PROJECT_DIR}"
                        --datasets-root "${UNITY_DATASETS_ROOT}"
                        --outputs-root "${UNITY_OUTPUTS_ROOT}"
                        --method "${METHOD_ID}"
                        --asset-subdir "${ASSET_SUBDIR}"
                        --output-name "${OUTPUT_NAME}"
                        --scenes "${UNITY_SCENE_NAME}"
                        --reference-image-dir "${REF_IMAGE_DIR}"
                        --profile-only
                        --profile-run "${run}"
                        --profile-views "${PROFILE_VIEWS}"
                        --profile-warmup "${PROFILE_WARMUP}"
                        --profile-frames "${PROFILE_FRAMES}"
                        --topology "${TOPOLOGY}"
                        --log-name "unity_profile_run_$(printf '%02d' "${run}").log"
                        "${CONDITION_ARG}")
                    [[ "${INDEXED_MESH_METHOD_AWARE}" -eq 1 ]] && PROFILE_CMD+=(--indexed-mesh-method-aware)
                fi
                if ! run_command "${LABEL}:profile-${run}" "${PROFILE_CMD[@]}"; then
                    if handle_failure "${LABEL}" "Unity profile run ${run} failed"; then continue 2; fi
                    break 2
                fi
            else
                echo "[${LABEL}] Profile run ${run} complete; skipping: ${PROFILE_JSON}"
            fi
        done
    fi

    if [[ "${SKIP_METRICS}" -eq 0 ]]; then
        METRICS_JSON="${RESULT_DIR}/metrics_summary.json"
        if [[ "${FORCE_METRICS}" -eq 1 ]] || ! metrics_complete "${METRICS_JSON}"; then
            if [[ "${DRY_RUN}" -eq 0 ]] && ! capture_complete "${RESULT_DIR}"; then
                if handle_failure "${LABEL}" "Cannot score metrics before Unity capture completes: ${RESULT_DIR}"; then continue; fi
                break
            fi
            METRICS_CMD=("${PYTHON_BIN}" "tools/evaluate_deployment_images.py"
                --prediction "${RESULT_DIR}"
                --ground-truth "${REF_PATH}"
                --output "${METRICS_JSON}")
            NATIVE_DIR="${MODEL_PATH}/${NATIVE_RENDER_SUBDIR}"
            if has_images "${NATIVE_DIR}"; then
                METRICS_CMD+=(--native "${NATIVE_DIR}" --native-shape-policy "${NATIVE_SHAPE_POLICY}")
            else
                echo "[${LABEL}] Native fidelity unavailable; no images under ${NATIVE_DIR}"
            fi
            if [[ "${WITH_LPIPS}" -eq 1 ]]; then
                METRICS_CMD+=(--with-lpips --lpips-net vgg --lpips-device "${LPIPS_DEVICE}" --lpips-tile "${LPIPS_TILE}")
            fi
            printf '[%s:metrics] Command:' "${LABEL}"; printf ' %q' env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_VALUE}" "${METRICS_CMD[@]}"; printf '\n'
            if [[ "${DRY_RUN}" -eq 0 ]]; then
                mkdir -p "${RESULT_DIR}"
                if ! env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_VALUE}" "${METRICS_CMD[@]}" 2>&1 | tee "${RESULT_DIR}/metrics_vgg.log"; then
                    if handle_failure "${LABEL}" "Image metric evaluation failed"; then continue; fi
                    break
                fi
            fi
        else
            echo "[${LABEL}] Metrics complete; skipping: ${METRICS_JSON}"
        fi
    fi

    RESULT_ARGS+=(--result "${LABEL}=${RESULT_DIR}")
    SCENE_RESULT_COUNT=$((SCENE_RESULT_COUNT + 1))
done

if [[ "${#RESULT_ARGS[@]}" -eq 0 ]]; then
    echo "No scene results available for report." >&2
    exit 1
fi

if [[ "${FAILED}" -ne 0 && "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
    echo "Unity evaluation stopped before all requested scenes completed; not writing a partial report. Re-run with --continue-on-error if a partial report is intended." >&2
    exit 1
fi

FORMAT_CMD=("${PYTHON_BIN}" "tools/format_unity_metrics_report.py"
    --method "${METHOD_ID}"
    --condition "${CONDITION}"
    --runtime "${PROFILE_RUNTIME_LABEL}"
    --profile-runs "${PROFILE_RUNS}"
    --output-dir "${REPORT_DIR}"
    --name "${REPORT_NAME}")
FORMAT_CMD+=("${RESULT_ARGS[@]}")
if ! run_command "report" "${FORMAT_CMD[@]}"; then
    echo "[report] Unity metrics formatting failed" >&2
    exit 1
fi

[[ "${FAILED}" -eq 0 ]] || exit 1
echo "Unity evaluation complete: method=${METHOD_ID} condition=${CONDITION} scenes=${SCENE_RESULT_COUNT} report=${REPORT_DIR}/${REPORT_NAME}.csv"
