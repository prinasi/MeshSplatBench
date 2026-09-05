# Unity-native TriAsset export

`msbench export-unity` exports a feature-preserving `.triasset` directory for
the MeshSplatBench Unity renderer.  It is the deployment asset for Unity quality,
FPS, and runtime-memory comparisons; it is intentionally **not** an OFF/COFF,
PLY, OBJ, or GLB replacement.

## Do I need Unity now?

No.  Exporting the package and its Python tests only require the MeshSplatBench
environment and a checkpoint; the exporter reads tensors on CPU and does not
import a CUDA renderer.
If the environment does not include the optional Typer CLI, the batch helper
automatically uses `tools/export_unity_triasset.py`, which only needs the
repository and PyTorch.

Install Unity when you need to:

- run the Unity-native renderer;
- compare Unity output against native renderer output;
- measure Player FPS, GPU/CPU frame time, load time, or runtime memory.

Use a pinned Unity LTS version through Unity Hub, commit `ProjectSettings` and
`Packages/manifest.json` for the benchmark project, and measure a standalone
Player rather than the Editor.  Unity itself may be installed on another
machine, but each result must record Unity version, graphics API, GPU/driver,
and Player build settings.

## Export

```bash
msbench export-unity \
  --method triangle-splatting \
  --checkpoint outputs/triangle-splatting/garden \
  --output assets/garden/triangle-splatting
```

For completed config-driven runs, use the batch helper with the same target
syntax as `single_train.sh`:

```bash
bash single_export_unity.sh triangle-splatting mipnerf360/all cpu
bash single_export_unity.sh 2dts dtu/scan105 cpu --force
```

It reads each scene YAML's `output.dir` and `adapter.checkpoint`, then writes
`<output.dir>/unity_native/<method>.triasset`. Passing `cpu` explicitly hides
CUDA devices; a numeric GPU id remains accepted for compatibility with the
training helper, although the current tensor package exporter runs on CPU.

The final directory is `assets/garden/triangle-splatting.triasset/`:

```text
manifest.json
buffers/
  positions.bin
  indices.bin
  opacity_logits.bin
  ... method-specific renderer buffers ...
```

The manifest is the contract between MeshSplatBench and Unity.  Every buffer is
little-endian, contiguous, and described by `dtype`, `shape`, and `semantic`.
It also records the exact checkpoint SHA-256 and coordinate/color-space
conventions.  A native Unity renderer must reject unknown schema versions and
missing required buffers instead of falling back to flat mesh colours.
The corrected opacity/gamma contract is revision `2` of schema `1.0`; loaders
and the CPU validator reject older exports so stale assets cannot silently enter
a new benchmark run.
Revision-2 manifests also freeze the evaluation background, call the custom
engine condition `method-aware`, explicitly set `cuda_equivalent: false`, and
declare whether an ordinary Unity Mesh has a comparable appearance input.

Validate an export on any CPU-only machine before copying it to Unity:

```bash
python3 tools/validate_unity_triasset_cpu.py \
  outputs/mesh-splatting/mipnerf360/garden/unity_native/mesh-splatting.triasset
```

The validator checks every declared byte size and, for MeshSplatting, recomputes
up to one million face opacities from the indexed vertex logits. Use
`--max-faces 0` for a complete check and `--verify-checkpoint-hash` when the
source checkpoint is locally available.

## Preserved renderer state

| Method | Preserved attributes |
|---|---|
| Triangle Splatting | triangle soup, opacity/sigma logits, complete SH |
| Mesh Splatting | indexed mesh, vertex-weight/sigma logits, native `min`-reduced triangle opacity, complete vertex SH |
| 2DTS | triangle soup, opacity logits, complete face/vertex SH, gamma and gamma-rescale factor |
| DiffSoup | indexed mesh, multi-resolution feature/alpha tensors, ColorMLP weights |

The initial exporter intentionally does not emit a generic GLB texture bake.
Producing one without a fixed calibration camera set and atlas budget would
make a seemingly comparable CG result non-reproducible.  Generic CG export is
a separate benchmark profile and must report its texture-baking budget.
Consequently DiffSoup currently has no valid general-purpose appearance
condition: a white topology preview is rejected rather than scored. The other
three methods may use the declared opaque SH-DC vertex-colour baseline.

For MeshSplatting, `triangle_opacity.bin` is an activated per-face value.  It is
computed as `min(floor + (1-floor)*sigmoid(w_i))` over the three indexed
vertices, matching the native CUDA forward pass.  A Unity shader must read this
value by primitive id with flat/no-interpolation semantics; interpolating the
three vertex weights is a different representation.  The original
`vertex_weight_logits.bin` remains in the package for auditability.

## Unity project bootstrap

After Unity is installed and a project exists, add the neutral loader contract:

```bash
python3 tools/create_unity_triasset_package.py --unity-project /path/to/MeshSplatBenchUnity
```

It creates `Assets/MeshSplatBench/Scripts/TriAssetLoader.cs`,
`TriAssetRenderer.cs`, `TriAssetSplatRenderer.cs`, and
`MeshSplatBenchFrameBenchmark.cs`, plus a ComputeShader and a procedural shader.
The loader parses the manifest and exposes every raw tensor as a `ComputeBuffer`;
it fails closed unless a matching method renderer is attached.

Attach the loader and exactly one matching component to the same GameObject:

| Asset method | Component |
|---|---|
| `triangle-splatting` | `TriangleSplattingTriAssetRenderer` |
| `mesh-splatting` | `MeshSplattingTriAssetRenderer` |
| `2dts` | `D2TSTriAssetRenderer` |
| `diffsoup` | `DiffSoupTriAssetRenderer` |

The triangle-family components use a ComputeShader append-buffer visibility
pass and `DrawProceduralIndirect` (Metal on Apple Silicon), bind the learned
SH/opacity/sigma or gamma buffers directly, and never create a Unity `Mesh`.
They are a portability baseline, **not** a claim of bitwise CUDA equivalence:
the native renderers use tile sorting and their own transmittance compositor.
The current DiffSoup component is explicitly an alpha-tested neural-feature
preview; complete multiresolution interpolation and ColorMLP evaluation remain
to be validated before it can participate in fidelity or quality scores.

The generated shader matches MeshSplatting's per-face opacity reduction,
screen-space affine colour interpolation, projected-incenter coverage, and
2DTS gamma vertex rescaling.  Its visible-primitive append list is not a native
tile-local depth sort, so transparent results must retain the portability
baseline label until a per-camera compositor is validated.  An opaque terminal
MeshSplatting deployment pass is a separate condition, justified only for
checkpoints whose opacity floor and sigma have converged to the solid regime.
The exporter records `terminal_solid_eligible` only when
`opacity_floor >= 0.999` and `exp(sigma_logits) <= 0.001`; the generated Unity
component then selects the versioned opaque Z-buffer shader automatically.

### Migrating the legacy Metal-only project to Linux/Vulkan

The older `MeshSplatBenchUnity` project uses premultiplied front-to-back blending and
Metal-only shader declarations. Before running that project on Linux, apply the
idempotent compatibility patch after syncing this repository:

```bash
python3 tools/patch_legacy_unity_vulkan.py \
  --unity-project /path/to/MeshSplatBenchUnity
```

The patch preserves `.msbench-vulkan.bak` copies, adds Vulkan shader variants,

clears capture targets to transparent black before splat accumulation, composites
the declared black/white evaluation background after RGBA readback, and disables
a renderer when its shader pass cannot be selected. It also installs each
procedural draw as a target-camera command buffer. This is required for Linux
Editor batch capture because the legacy `OnRenderObject` submission is not
reliably invoked by a disabled camera rendered through `Camera.Render()`. A
successful renderer startup logs `Installed explicit camera draw ... on Vulkan`.
The command is submitted at `AfterForwardAlpha`, while the offscreen Vulkan
camera render pass still has its color attachment bound. Later camera events
can execute after Unity has ended that render pass and trigger a native driver
crash with `Missing Vulkan framebuffer attachment image?`. Command-buffer
installation is deferred until `ColmapBatchCapture` has assigned that offscreen
target; installing it in the renderer's `Start()` can make a headless Linux
Editor draw into a missing screen framebuffer. The installer checks
`Shader.isSupported` rather than
calling `Material.SetPass` before camera rendering begins, since the latter
applies graphics state immediately and can make Vulkan request a nonexistent
framebuffer. For 2DTS, the patch also binds an unused placeholder
`_Sigma` descriptor because Vulkan requires every shader-declared buffer to be
bound even when the active method branch does not read it.
The Python evaluator chooses Metal on macOS and Vulkan on Linux automatically
and rejects runs whose new PNGs are all solid clears.

The benchmark component uses a fixed HDR `RenderTexture`, disables VSync,
warms up, and writes CPU/GPU mean, p50, and p95 frame times after it finishes.
Its optional topology preview is explicitly debug-only and must not be used for
quality or FPS numbers.

Use the public condition names `method-aware` and `general-purpose`. The legacy
Unity command-line switches `-method-specific` and `-standard-mesh` are only
implementation details and must not appear as claims such as `faithful` or
`native-equivalent` in tables.

## Unity benchmark requirements

For a valid performance run, use the same camera trajectory and linear colour
space for all methods; disable VSync and dynamic resolution; render to a fixed
offscreen `RenderTexture`; warm up before timing; and do not read back or save
frames while measuring FPS.  Run quality evaluation separately with readback.

Report these independently:

1. Native-to-Unity fidelity: Unity renderer output versus MeshSplatBench native
   renderer output on fixed cameras.
2. Quality: Unity output versus GT on a held-out camera trajectory.
3. Steady-state performance: CPU/GPU frame p50/p95 and FPS.
4. Memory/load: disk bytes, peak load memory, steady-state graphics buffers,
   textures, and resident memory.

Score saved Unity frames, dataset GT, and saved Native CUDA frames with the same
offline implementation. The default uses the native triangle-family Gaussian
SSIM and full-frame LPIPS-VGG; `--lpips-tile` is a labelled low-memory
approximation and must not be mixed into the main comparison:

```bash
python3 tools/evaluate_deployment_images.py \
  --prediction outputs/mesh-splatting/mipnerf360/garden/unity_method_aware_rev2 \
  --ground-truth /datasets/MipNeRF360/garden/images_4 \
  --native outputs/mesh-splatting/mipnerf360/garden/renders/test/renders \
  --with-lpips --lpips-net vgg --lpips-device cpu \
  --native-shape-policy strict
```

For completed config-driven runs, `single_unity_eval.sh` wires the full
Unity-side reporting path together:

```bash
bash single_unity_eval.sh 2dts mipnerf360/bicycle 0 \
  --unity "$UNITY" \
  --unity-project "$PROJECT" \
  --output-name unity_method_aware_vulkan_cb
```

The wrapper uses the same target syntax as `single_train.sh`, including
`mipnerf360/all` and the target-first `--method` form. It exports a TriAsset
only when `<output.dir>/unity_native/<method>.triasset` is missing or invalid,
captures held-out Unity PNGs plus `fps_per_test_view.csv`, builds or reuses a
Linux standalone Development Player for repeated no-I/O GPU runtime profiles,
scores PSNR/SSIM/LPIPS-VGG against the configured
reference image directory, and writes:

```text
outputs/unity_reports/<method>/<output-name>/unity_metrics_report.csv
outputs/unity_reports/<method>/<output-name>/unity_metrics_report.json
outputs/unity_reports/<method>/<output-name>/unity_metrics_report.md
```

The default profile runtime is `standalone-player`; pass
`--profile-runtime editor` only for local debugging. The current Unity camera
replay requires COLMAP `sparse/0/cameras.bin` and `sparse/0/images.bin`; DTU
and synthetic datasets without that layout are rejected early. The formatted FPS
column is GPU-only. If Unity's runtime profile writes `gpu_samples: 0`, the
Player profile stage fails rather than substituting CPU or engine frame time.
`--gpu-timing-min-fraction` controls the minimum valid GPU timing coverage
accepted from Unity `FrameTimingManager`; it defaults to `0.5` because Vulkan
Player runs can return GPU timings on most, but not every, frame.
Unity-vs-GT quality remains strict about image shape. Native-fidelity comparison
uses `--native-shape-policy crop` by default in the wrapper because existing
native render folders can be one pixel smaller after dataset downsampling; the
crop is recorded in `metrics_summary.json`. Use `--native-shape-policy strict`
for final native-fidelity tables when all native renders have been regenerated
at the exact Unity capture resolution, or `skip` to omit mismatched native views.

`tools/evaluate_unity_triasset_outputs.py` applies this evaluator to all nine
Mip-NeRF 360 scenes and refuses stale assets or an unsupported general-purpose
condition. Quality aggregation deliberately excludes FPS; use repeated
standalone Player profiles and `tools/aggregate_unity_deployment_profile.py`
for CPU/GPU P50/P95, load, asset, and graphics allocation results.
