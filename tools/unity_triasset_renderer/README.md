# TriBench Unity procedural renderer templates

`TriAssetSplatRenderer.cs` implements the method-aware condition using an
append-buffer ComputeShader cull pass and
`DrawProceduralIndirect`, which maps to Metal on Apple Silicon.  It has concrete
components for Triangle Splatting, Mesh Splatting, 2DTS, and DiffSoup.

The triangle-family shader preserves SH colour, opacity, and sigma/gamma inputs
in GPU buffers.  It is a Unity implementation baseline, not a claim of bitwise
native CUDA equivalence: the CUDA implementations use tile sorting and their
own transmittance compositor.  DiffSoup currently provides a deterministic
alpha-tested feature preview; the complete multires feature interpolator and
ColorMLP evaluation require a separately validated shader implementation.

`GeneralPurposeMeshTriAssetRenderer.cs` implements the ordinary opaque Mesh
condition from SH-DC vertex colours. It rejects methods without a comparable
colour field; DiffSoup needs a separately specified fixed-budget appearance
bake and is never rendered as a scored white mesh.

Do not report native-to-Unity quality or FPS equivalence until those renderer
differences have been evaluated on a fixed camera set.
