#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# MeshSplatting topology ablation:
#   mesh/indexed: preserve shared vertices from the exported checkpoint
#   shader-soup:  keep indexed buffers; procedural shader fetches per corner
#   materialized-soup: export duplicated per-corner buffers and sequential indices
#
# This script delegates export/capture/profile/metric calculation to
# single_unity_eval.sh and only controls topology/renderer conditions.

RENDERERS=(topology-native)
TOPOLOGIES=(mesh shader-soup materialized-soup)
OUTPUT_PREFIX="unity_mesh_topology"
REPORT_ROOT=""
REPORT_NAME="unity_metrics_report"
MATERIALIZED_ASSET_SUBDIR="unity_native_materialized_soup"
WRAPPER_CONTINUE=0
DRY_RUN=0
PASSTHROUGH=()
POSITIONAL=()

usage() {
    cat <<EOF
Usage:
  bash $0 <dataset/scene|dataset/all|all> [target ...] <gpu_id|cpu> [options]

Examples:
  bash $0 mipnerf360/all 0 --unity "\$UNITY" --unity-project "\$PROJECT"
  bash $0 mipnerf360/bicycle 0 --topologies mesh,shader-soup --profile-runs 5
  bash $0 tandt/all 0 --topologies mesh,shader-soup,materialized-soup --datasets-root /data/TanksAndTemples

Wrapper options:
  --topologies T     Comma-separated topology set: mesh,shader-soup,materialized-soup,all
  --topology T       Alias of --topologies
  --output-prefix P  Output folder prefix per scene (default: ${OUTPUT_PREFIX})
  --report-root DIR  Combined report root (default: outputs/unity_reports/mesh-splatting/${OUTPUT_PREFIX})
  --report-name NAME Report file prefix inside each condition directory (default: ${REPORT_NAME})
  --materialized-asset-subdir NAME
                     Asset directory for materialized-soup only (default: ${MATERIALIZED_ASSET_SUBDIR})

Common forwarded options:
  --unity PATH --unity-project PATH --datasets-root PATH
  --profile-runs N --profile-views N --profile-warmup N --profile-frames N
  --fps-warmup N --fps-frames N --force --report-only --continue-on-error --dry-run

All evaluation is run with method=mesh-splatting. mesh uses a true Unity
indexed MeshRenderer with method-aware SH appearance. shader-soup uses the
procedural method-aware renderer with corner-level access. materialized-soup
writes a separate duplicated .triasset and renders it through Unity MeshRenderer.
EOF
}

require_value() {
    [[ "$#" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
}

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

normalize_renderer() {
    local value
    value="$(lower "$1")"
    value="${value//_/-}"
    case "${value}" in
        method-aware|method-specific|native|aware) echo "method-aware" ;;
        general-purpose|standard-mesh|standard|mesh-renderer) echo "general-purpose" ;;
        *) echo ""; return 1 ;;
    esac
}

normalize_topology() {
    local value
    value="$(lower "$1")"
    value="${value//_/-}"
    case "${value}" in
        mesh|indexed|indexed-mesh) echo "mesh" ;;
        soup|shader-soup|shader-soup-indexed|shader-indexed-soup) echo "shader-soup" ;;
        materialized-soup|export-soup|true-soup|deindexed|deindexed-soup|triangle-soup) echo "materialized-soup" ;;
        *) echo ""; return 1 ;;
    esac
}

parse_renderer_list() {
    local raw="$1" item normalized
    RENDERERS=()
    raw="$(lower "${raw}")"
    raw="${raw// /,}"
    if [[ "${raw}" == "both" || "${raw}" == "all" ]]; then
        RENDERERS=(method-aware general-purpose)
        return
    fi
    IFS=',' read -r -a _items <<< "${raw}"
    for item in "${_items[@]}"; do
        [[ -n "${item}" ]] || continue
        normalized="$(normalize_renderer "${item}")" || { echo "Unknown renderer: ${item}" >&2; exit 1; }
        RENDERERS+=("${normalized}")
    done
    [[ "${#RENDERERS[@]}" -gt 0 ]] || { echo "No renderer selected" >&2; exit 1; }
}

parse_topology_list() {
    local raw="$1" item normalized
    TOPOLOGIES=()
    raw="$(lower "${raw}")"
    raw="${raw// /,}"
    if [[ "${raw}" == "both" || "${raw}" == "all" ]]; then
        TOPOLOGIES=(mesh shader-soup materialized-soup)
        return
    fi
    IFS=',' read -r -a _items <<< "${raw}"
    for item in "${_items[@]}"; do
        [[ -n "${item}" ]] || continue
        normalized="$(normalize_topology "${item}")" || { echo "Unknown topology: ${item}" >&2; exit 1; }
        TOPOLOGIES+=("${normalized}")
    done
    [[ "${#TOPOLOGIES[@]}" -gt 0 ]] || { echo "No topology selected" >&2; exit 1; }
}

if [[ "$#" -lt 1 || "${1}" == "-h" || "${1}" == "--help" ]]; then usage; exit 0; fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --renderer|--renderers) echo "single_mesh_topology_unity_eval.sh now selects renderer by topology; use single_unity_eval.sh for extra renderer sweeps." >&2; exit 1 ;;
        --topology|--topologies|--mesh-topology|--mesh_topology) require_value "$@"; parse_topology_list "$2"; shift 2 ;;
        --output-prefix|--output_prefix) require_value "$@"; OUTPUT_PREFIX="$2"; shift 2 ;;
        --report-root|--report_root) require_value "$@"; REPORT_ROOT="$2"; shift 2 ;;
        --report-name|--report_name) require_value "$@"; REPORT_NAME="$2"; shift 2 ;;
        --materialized-asset-subdir|--materialized_asset_subdir) require_value "$@"; MATERIALIZED_ASSET_SUBDIR="$2"; shift 2 ;;
        --method) echo "single_mesh_topology_unity_eval.sh fixes --method to mesh-splatting." >&2; exit 1 ;;
        --report-dir|--report_dir) echo "Use --report-root for topology ablation reports; per-condition --report-dir is generated automatically." >&2; exit 1 ;;
        --continue-on-error) WRAPPER_CONTINUE=1; PASSTHROUGH+=("$1"); shift ;;
        --dry-run|--dry_run) DRY_RUN=1; PASSTHROUGH+=("$1"); shift ;;
        --output-subdir|--output_subdir)
            require_value "$@"; PASSTHROUGH+=(--asset-subdir "$2"); shift 2 ;;
        --unity|--unity-project|--unity_project|--project|--datasets-root|--datasets_root|--config-root|--config_root|--asset-subdir|--asset_subdir|--profile-runs|--profile_runs|--profile-views|--profile_views|--profile-warmup|--profile_warmup|--profile-frames|--profile_frames|--fps-warmup|--fps_warmup|--fps-frames|--fps_frames|--profile-runtime|--profile_runtime|--unity-player|--unity_player|--unity-player-output|--unity_player_output|--unity-player-target|--unity_player_target|--display|--gpu-timing-min-fraction|--gpu_timing_min_fraction|--lpips-device|--lpips_device|--lpips-tile|--lpips_tile|--native-render-subdir|--native_render_subdir|--native-shape-policy|--native_shape_policy|--reference-image-dir|--reference_image_dir|--python)
            require_value "$@"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --method-aware|--method_aware|--method-specific|--method_specific|--general-purpose|--general_purpose|--standard-mesh|--standard_mesh)
            echo "Use --renderer to select renderer conditions in this wrapper." >&2; exit 1 ;;
        --force-player-build|--force_player_build|--skip-player-build|--skip_player_build|--player-batchmode|--player_batchmode|--show-player-output|--show_player_output|--no-lpips|--skip-lpips|--skip-export|--skip_export|--skip-capture|--skip_capture|--skip-profile|--skip_profile|--skip-metrics|--skip_metrics|--report-only|--report_only|--force|--force-export|--force_export|--force-capture|--force_capture|--force-profile|--force_profile|--force-metrics|--force_metrics|--skip-unity-patch|--skip_unity_patch)
            PASSTHROUGH+=("$1"); shift ;;
        --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ "${#POSITIONAL[@]}" -lt 2 ]]; then
    usage
    exit 1
fi

GPU_ID="${POSITIONAL[$((${#POSITIONAL[@]} - 1))]}"
TARGETS=("${POSITIONAL[@]:0:$((${#POSITIONAL[@]} - 1))}")

if [[ -z "${REPORT_ROOT}" ]]; then
    REPORT_ROOT="outputs/unity_reports/mesh-splatting/${OUTPUT_PREFIX}"
fi

FAILED=0
REPORT_PATHS=()
for renderer in "${RENDERERS[@]}"; do
    for topology in "${TOPOLOGIES[@]}"; do
        renderer_tag="${renderer//-/_}"
        topology_tag="${topology//-/_}"
        topology_arg="indexed"
        export_topology_arg="indexed"
        asset_subdir_args=()
        indexed_mesh_args=()
        condition_arg="--general-purpose"
        report_tag="${topology_tag}"
        if [[ "${topology}" == "shader-soup" ]]; then
            topology_arg="soup"
            condition_arg="--method-aware"
            report_tag="shader_soup"
        elif [[ "${topology}" == "mesh" ]]; then
            indexed_mesh_args=(--indexed-mesh-method-aware)
            report_tag="indexed_mesh"
        elif [[ "${topology}" == "materialized-soup" ]]; then
            topology_arg="indexed"
            export_topology_arg="soup"
            asset_subdir_args=(--asset-subdir "${MATERIALIZED_ASSET_SUBDIR}")
            indexed_mesh_args=(--indexed-mesh-method-aware)
            report_tag="materialized_soup"
        fi
        output_name="${OUTPUT_PREFIX}_${report_tag}"
        report_dir="${REPORT_ROOT}/${report_tag}"

        cmd=("bash" "${REPO_ROOT}/single_unity_eval.sh" "mesh-splatting")
        cmd+=("${TARGETS[@]}" "${GPU_ID}")
        cmd+=("${condition_arg}" --topology "${topology_arg}" --export-topology "${export_topology_arg}")
        cmd+=(--output-name "${output_name}" --report-dir "${report_dir}" --report-name "${REPORT_NAME}")
        cmd+=("${PASSTHROUGH[@]}")
        if [[ "${#indexed_mesh_args[@]}" -gt 0 ]]; then
            cmd+=("${indexed_mesh_args[@]}")
        fi
        if [[ "${#asset_subdir_args[@]}" -gt 0 ]]; then
            cmd+=("${asset_subdir_args[@]}")
        fi

        printf '[mesh-topology:%s/%s] Command:' "${renderer}" "${topology}"; printf ' %q' "${cmd[@]}"; printf '\n'
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            if ! "${cmd[@]}"; then
                FAILED=1
                if [[ "${WRAPPER_CONTINUE}" -eq 0 ]]; then
                    echo "[mesh-topology:${renderer}/${topology}] Failed; stopping." >&2
                    exit 1
                fi
            fi
        fi
        REPORT_PATHS+=("${topology}: ${report_dir}/${REPORT_NAME}.csv")
    done
done

echo "MeshSplatting topology ablation reports:"
for path in "${REPORT_PATHS[@]}"; do
    echo "  ${path}"
done

[[ "${FAILED}" -eq 0 ]] || exit 1
