using UnityEngine;

namespace TriBench.Unity
{
    // Implementations must consume raw TriAsset buffers, never baked colours.
    public abstract class TriAssetRenderer : MonoBehaviour
    {
        public abstract string MethodName { get; }
        public virtual bool SupportsMethod(string method) { return MethodName == method; }
        public abstract void Configure(TriAssetLoader asset);
    }
}
