# Legacy primitive-budget configurations

Despite the historical directory name, these configurations cap the number of
primitives (typically at 15,000). They are a **primitive-budget ablation**, not
a texture or appearance ablation.

Do not label results from this directory as `texture ablation`. A valid texture
study must freeze the checkpoint/cameras/renderer and vary an explicit atlas or
appearance-bake budget, recording calibration views, atlas resolution, texel
count, format, and disk/runtime memory.
