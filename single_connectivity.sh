#!/usr/bin/env bash
set -euo pipefail

# single_connectivity.sh - batch MeshSplatting connectivity evaluation.
#
# Primary interface:
#   bash single_connectivity.sh <method> <dataset/scene|dataset/all|all> [options]
#
# Config lookup:
#   configs/<method>/<dataset>/<scene>.yaml
#
# Pipeline:
#   tools/calculate_connectivity.py per scene -> formatted paper-friendly table

CONFIG_ROOT="configs"
CONNECTIVITY_SCRIPT="${CONNECTIVITY_SCRIPT:-tools/calculate_connectivity.py}"
SUMMARY_FORMAT="markdown"
SUMMARY_OUTPUT=""
TOP_COMPONENTS=3
SKIP_VERTEX_CHECK=0
REUSE_EXISTING=1
CONTINUE_ON_ERROR=0
VERBOSE=0

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

usage() {
    cat <<EOF
Usage:
  bash $0 <method> <dataset/scene|dataset/all|all> [options]

Examples:
  bash $0 mesh-splatting mipnerf360/garden
  bash $0 meshsplatting nerf_synthetic/all
  bash $0 mesh-splatting dtu/all --format markdown --summary_output connectivity_dtu.md
  bash $0 mesh-splatting all --format csv --summary_output connectivity_all.csv

Options:
  --config_root PATH       Config root                              (default: ${CONFIG_ROOT})
  --connectivity_script P  Connectivity metric script               (default: ${CONNECTIVITY_SCRIPT})
  --format KIND            Summary format: markdown, plain, csv, latex
                                                               (default: ${SUMMARY_FORMAT})
  --summary_output PATH    Write formatted summary to this file
  --output PATH            Alias for --summary_output
  --top_components N       Number of largest components in JSON     (default: ${TOP_COMPONENTS})
  --skip_vertex_check      Skip non-manifold vertex one-ring check
  --reuse_existing         Reuse existing connectivity_metrics.json files (default)
  --force_recompute        Recalculate metrics even if output JSON exists
  --continue_on_error      Continue remaining scenes if one scene fails
  --python PATH            Python executable
  --verbose                Also print per-scene tool output
  -h, --help               Show this help

The summary reports:
  V/F Ratio, Vertex Valence, Boundary Edge Ratio, Non-Manifold Edge Ratio,
  Non-Manifold Vertex Ratio, Largest Connected Component face/area ratios,
  and connected component count.

Currently checkpoint-based connectivity evaluation is supported for
MeshSplatting only.
EOF
}

require_value() {
    if [[ "$#" -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
    fi
}

contains_scene() {
    local needle="$1"; shift
    for item in "$@"; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

canonical_method() {
    local name
    name="$(lower "$1")"
    name="${name//_/-}"
    case "${name}" in
        meshsplatting) echo "mesh-splatting" ;;
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
        nerfsynthetic|nerf-synthetic|nerf_synthetic|blender|synthetic) echo "nerf_synthetic" ;;
        dtu) echo "dtu" ;;
        *) echo "${name}" ;;
    esac
}

dataset_label() {
    case "$(normalize_dataset "$1")" in
        mipnerf360) echo "mipnerf360" ;;
        tandt) echo "tandt" ;;
        nerf_synthetic) echo "nerf_synthetic" ;;
        dtu) echo "dtu" ;;
        *) echo "$(normalize_dataset "$1")" ;;
    esac
}

resolve_python_bin() {
    if [[ "${PYTHON_BIN}" == */* ]]; then
        [[ ! -x "${PYTHON_BIN}" ]] && { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
        return 0
    fi
    local resolved
    resolved="$(command -v "${PYTHON_BIN}" || true)"
    [[ -z "${resolved}" ]] && { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
    PYTHON_BIN="${resolved}"
}

is_mipnerf360() { contains_scene "$1" "${MIPNERF360_SCENES[@]}"; }
is_tanks_and_temples() { contains_scene "$1" "${TANKS_AND_TEMPLES_SCENES[@]}"; }
is_dtu_scene() { contains_scene "$1" "${DTU_SCENES[@]}" || [[ "$1" =~ ^scan[0-9]+$ ]]; }
is_nerf_synthetic() { contains_scene "$1" "${NERF_SYNTHETIC_SCENES[@]}"; }

infer_dataset_for_scene() {
    local scene="$1"
    if is_mipnerf360 "${scene}"; then
        echo "mipnerf360"
    elif is_tanks_and_temples "${scene}"; then
        echo "tandt"
    elif is_dtu_scene "${scene}"; then
        echo "dtu"
    elif is_nerf_synthetic "${scene}"; then
        echo "nerf_synthetic"
    else
        echo "custom"
    fi
}

parse_target() {
    local target="$1"
    if [[ "${target}" == */* ]]; then
        PARSED_DATASET="$(dataset_label "${target%%/*}")"
        PARSED_SCENE="${target#*/}"
        return
    fi
    PARSED_DATASET="$(infer_dataset_for_scene "${target}")"
    PARSED_SCENE="${target}"
}

scene_config_path() {
    local method="$1" dataset="$2" scene="$3"
    echo "${CONFIG_ROOT}/${method}/$(dataset_label "${dataset}")/${scene}.yaml"
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

        if [[ "${target}" != */* ]]; then
            case "$(dataset_label "${target}")" in
                mipnerf360) for scene in "${MIPNERF360_SCENES[@]}"; do EXPANDED_TARGETS+=("mipnerf360/${scene}"); done; continue ;;
                tandt) for scene in "${TANKS_AND_TEMPLES_SCENES[@]}"; do EXPANDED_TARGETS+=("tandt/${scene}"); done; continue ;;
                nerf_synthetic) for scene in "${NERF_SYNTHETIC_SCENES[@]}"; do EXPANDED_TARGETS+=("nerf_synthetic/${scene}"); done; continue ;;
            esac
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

        EXPANDED_TARGETS+=("${target}")
    done
}

config_value() {
    local config_file="$1" dotted_key="$2"
    "${PYTHON_BIN}" - "${config_file}" "${dotted_key}" <<'PY'
import sys

from msbench.core.config import Config

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

connectivity_output_file() {
    local config_file="$1" method="$2" dataset="$3" scene="$4"
    local explicit out_dir
    explicit="$(config_value "${config_file}" "connectivity.output")"
    if [[ -n "${explicit}" ]]; then
        echo "${explicit}"
        return 0
    fi
    out_dir="$(config_value "${config_file}" "output.dir")"
    if [[ -n "${out_dir}" ]]; then
        echo "${out_dir}/connectivity_metrics.json"
    else
        echo "outputs/${method}/$(dataset_label "${dataset}")/${scene}/connectivity_metrics.json"
    fi
}

run_connectivity_scene() {
    local dataset="$1" scene="$2" config_file="$3" json_file="$4"
    local log_file
    log_file="$(dirname "${json_file}")/connectivity.log"
    mkdir -p "$(dirname "${json_file}")"

    if [[ "${REUSE_EXISTING}" -eq 1 && -s "${json_file}" ]]; then
        echo "[${dataset}/${scene}] Reading existing result: ${json_file}"
        return 0
    fi

    local cmd=("${PYTHON_BIN}" "${CONNECTIVITY_SCRIPT}" --config "${config_file}" --output "${json_file}" --top-components "${TOP_COMPONENTS}")
    if [[ "${SKIP_VERTEX_CHECK}" -eq 1 ]]; then
        cmd+=(--skip-vertex-check)
    fi

    echo "[${dataset}/${scene}] Connectivity started. Log: ${log_file}"
    if [[ "${VERBOSE}" -eq 1 ]]; then
        if ! "${cmd[@]}" 2>&1 | tee "${log_file}"; then
            echo "[${dataset}/${scene}] Connectivity FAILED. Last log:" >&2
            tail -n 40 "${log_file}" >&2 || true
            return 1
        fi
    else
        if ! "${cmd[@]}" > "${log_file}" 2>&1; then
            echo "[${dataset}/${scene}] Connectivity FAILED. Last log:" >&2
            tail -n 40 "${log_file}" >&2 || true
            return 1
        fi
    fi
    echo "[${dataset}/${scene}] Connectivity done. ${json_file}"
}

print_summary() {
    local format="$1" output="$2"; shift 2
    "${PYTHON_BIN}" - "${format}" "${output:-}" "$@" <<'PY'
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

fmt = sys.argv[1]
summary_output = sys.argv[2] or None
paths = [Path(item) for item in sys.argv[3:]]

FIELDS = [
    ("Dataset", "dataset", "text"),
    ("Scene", "scene", "text"),
    ("V/F Ratio", "vertex_face_ratio", "4"),
    ("Valence", "average_vertex_valence", "2"),
    ("Boundary Edge Ratio", "boundary_edge_ratio", "4"),
    ("Non-Manifold Edge Ratio", "non_manifold_edge_ratio", "4"),
    ("Non-Manifold Vertex Ratio", "non_manifold_vertex_ratio", "4"),
    ("LCC Face Ratio", "largest_component_ratio", "4"),
    ("LCC Area Ratio", "largest_component_area_ratio", "4"),
    ("Components", "components_count", "0"),
    ("JSON", "json", "text"),
]


def number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_value(value, style):
    if style == "text":
        return "" if value is None else str(value)
    value = number(value)
    if value is None:
        return "N/A"
    if style == "0":
        return str(int(round(value)))
    digits = int(style)
    return f"{value:.{digits}f}"


def read_row(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    components = data.get("components", {}) if isinstance(data, dict) else {}
    return {
        "dataset": data.get("dataset") or "unknown",
        "scene": data.get("scene") or path.parent.name,
        "vertex_face_ratio": metrics.get("vertex_face_ratio"),
        "average_vertex_valence": metrics.get("average_vertex_valence"),
        "boundary_edge_ratio": metrics.get("boundary_edge_ratio"),
        "non_manifold_edge_ratio": metrics.get("non_manifold_edge_ratio"),
        "non_manifold_vertex_ratio": metrics.get("non_manifold_vertex_ratio"),
        "largest_component_ratio": metrics.get("largest_component_ratio"),
        "largest_component_area_ratio": metrics.get("largest_component_area_ratio"),
        "components_count": components.get("count"),
        "json": str(path),
        "_is_mean": False,
    }


def mean_row(dataset, rows, scene):
    result = {"dataset": dataset, "scene": scene, "json": "", "_is_mean": True}
    for _, key, style in FIELDS:
        if style == "text" or key in result:
            continue
        values = [number(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def add_mean_rows(rows):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    result = []
    for dataset, group in grouped.items():
        result.extend(group)
        if len(group) > 1:
            result.append(mean_row(dataset, group, "MEAN"))
    if len(rows) > 1 and len(grouped) > 1:
        result.append(mean_row("ALL", rows, "MEAN"))
    return result


def table_cells(rows):
    return [[fmt_value(row.get(key), style) for _, key, style in FIELDS] for row in rows]


def column_widths(records):
    return [
        max(3, max(len(record[i]) for record in records))
        for i in range(len(records[0]))
    ]


def align_cell(text, width, style):
    if style == "text":
        return str(text).ljust(width)
    return str(text).rjust(width)


def markdown_separator(width, style):
    if style == "text":
        return "-" * width
    return "-" * (width - 1) + ":"


def render_plain(rows):
    headers = [name for name, _, _ in FIELDS]
    records = table_cells(rows)
    widths = column_widths([headers] + records)
    styles = [style for _, _, style in FIELDS]
    lines = [
        "  ".join(align_cell(text, width, style) for text, width, style in zip(headers, widths, styles)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(align_cell(text, width, style) for text, width, style in zip(record, widths, styles))
        for record in records
    )
    return "\n".join(lines)


def render_markdown(rows):
    headers = [name for name, _, _ in FIELDS]
    records = table_cells(rows)
    widths = column_widths([headers] + records)
    styles = [style for _, _, style in FIELDS]
    lines = [
        "| " + " | ".join(align_cell(text, width, style) for text, width, style in zip(headers, widths, styles)) + " |",
        "| " + " | ".join(markdown_separator(width, style) for width, style in zip(widths, styles)) + " |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(align_cell(text, width, style) for text, width, style in zip(record, widths, styles))
            + " |"
        )
    return "\n".join(lines)


def latex_escape(text):
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")


def render_latex(rows):
    headers = [name for name, _, _ in FIELDS[:-1]]
    lines = [
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        record = [fmt_value(row.get(key), style) for _, key, style in FIELDS[:-1]]
        lines.append(" & ".join(latex_escape(item) for item in record) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def render_csv(rows):
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([name for name, _, _ in FIELDS])
    writer.writerows(table_cells(rows))
    return output.getvalue().rstrip("\n")


rows = [read_row(path) for path in paths]
rows = add_mean_rows(rows)
if fmt == "plain":
    text = render_plain(rows)
elif fmt == "markdown":
    text = render_markdown(rows)
elif fmt == "csv":
    text = render_csv(rows)
elif fmt == "latex":
    text = render_latex(rows)
else:
    raise SystemExit(f"Unknown summary format: {fmt}")

if summary_output:
    out = Path(summary_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"[single_connectivity] summary saved: {out}")
print(text)
PY
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

METHOD_ID="$(canonical_method "$1")"
shift
if [[ "${METHOD_ID}" != "mesh-splatting" ]]; then
    echo "Currently only MeshSplatting connectivity is supported, got: ${METHOD_ID}" >&2
    exit 1
fi

TARGET_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --*) break ;;
        *)
            TARGET_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${#TARGET_ARGS[@]}" -eq 0 ]]; then
    echo "Missing target. Expected dataset/scene, dataset/all, or all." >&2
    usage
    exit 1
fi

if [[ "${#TARGET_ARGS[@]}" -gt 1 ]]; then
    last_index=$((${#TARGET_ARGS[@]} - 1))
    if [[ "${TARGET_ARGS[${last_index}]}" =~ ^[0-9]+$ ]]; then
        unset "TARGET_ARGS[${last_index}]"
    fi
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config_root|--config-root) require_value "$@"; CONFIG_ROOT="$2"; shift 2 ;;
        --connectivity_script|--connectivity-script) require_value "$@"; CONNECTIVITY_SCRIPT="$2"; shift 2 ;;
        --format) require_value "$@"; SUMMARY_FORMAT="$(lower "$2")"; shift 2 ;;
        --summary_output|--summary-output|--output) require_value "$@"; SUMMARY_OUTPUT="$2"; shift 2 ;;
        --top_components|--top-components) require_value "$@"; TOP_COMPONENTS="$2"; shift 2 ;;
        --skip_vertex_check|--skip-vertex-check) SKIP_VERTEX_CHECK=1; shift ;;
        --reuse_existing|--reuse-existing) REUSE_EXISTING=1; shift ;;
        --force_recompute|--force-recompute) REUSE_EXISTING=0; shift ;;
        --continue_on_error|--continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        --python) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --verbose) VERBOSE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

case "${SUMMARY_FORMAT}" in
    markdown|plain|csv|latex) ;;
    *) echo "Invalid --format '${SUMMARY_FORMAT}'. Expected markdown, plain, csv, or latex." >&2; exit 1 ;;
esac

resolve_python_bin
if [[ ! -f "${CONNECTIVITY_SCRIPT}" ]]; then
    echo "Connectivity script not found: ${CONNECTIVITY_SCRIPT}" >&2
    exit 1
fi

expand_targets "${TARGET_ARGS[@]}"
JSON_FILES=()
FAILED_TARGETS=()

echo "[single_connectivity] method=${METHOD_ID}"
echo "[single_connectivity] targets=${#EXPANDED_TARGETS[@]}"
echo "[single_connectivity] summary_format=${SUMMARY_FORMAT}"

for target in "${EXPANDED_TARGETS[@]}"; do
    parse_target "${target}"
    DATASET="${PARSED_DATASET}"
    SCENE="${PARSED_SCENE}"
    CONFIG_FILE="$(scene_config_path "${METHOD_ID}" "${DATASET}" "${SCENE}")"
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[${DATASET}/${SCENE}] Missing config: ${CONFIG_FILE}" >&2
        FAILED_TARGETS+=("${DATASET}/${SCENE}")
        if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
            continue
        fi
        exit 1
    fi

    JSON_FILE="$(connectivity_output_file "${CONFIG_FILE}" "${METHOD_ID}" "${DATASET}" "${SCENE}")"
    if run_connectivity_scene "${DATASET}" "${SCENE}" "${CONFIG_FILE}" "${JSON_FILE}"; then
        JSON_FILES+=("${JSON_FILE}")
    else
        FAILED_TARGETS+=("${DATASET}/${SCENE}")
        if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
            exit 1
        fi
    fi
done

if [[ "${#JSON_FILES[@]}" -eq 0 ]]; then
    echo "No connectivity metrics were produced." >&2
    exit 1
fi

echo
print_summary "${SUMMARY_FORMAT}" "${SUMMARY_OUTPUT}" "${JSON_FILES[@]}"

if [[ "${#FAILED_TARGETS[@]}" -gt 0 ]]; then
    echo
    echo "[single_connectivity] Failed targets: ${FAILED_TARGETS[*]}" >&2
    exit 1
fi
