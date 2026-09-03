using System;

namespace TriBench.UnityNative
{
    /// <summary>Command-line options supplied by the batch Unity evaluator.</summary>
    internal static class TriAssetRuntimeOptions
    {
        internal static string Get(string name, string fallback = null)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
            return fallback;
        }

        internal static int GetInt(string name, int fallback)
        {
            return int.TryParse(Get(name), out int value) && value > 0 ? value : fallback;
        }

        internal static bool Has(string name)
        {
            foreach (string arg in Environment.GetCommandLineArgs())
                if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }
    }
}
