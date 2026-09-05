using UnityEngine;

namespace TriBench.UnityNative
{
    public abstract class TriAssetRenderer : MonoBehaviour
    {
        [Tooltip("Absolute path to the .triasset directory.")]
        public string AssetDirectory;

        public abstract bool IsReady { get; }
        public abstract string Status { get; }

        /// <summary>Switches numerical-output mode when the renderer needs one.</summary>
        public virtual void SetOutputRawCodeValues(bool enabled) { }

        /// <summary>Synchronizes per-camera shader state before rendering.</summary>
        public virtual void PrepareCamera(Camera camera) { }
    }
}
