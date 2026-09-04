using System.IO;
using TriBench.UnityNative;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class TriBenchGardenSetup
{
    const string ScenePath = "Assets/Scenes/TriBenchGarden.unity";
    const string AssetPath = "/Volumes/GLOWAY/Unity/unity_native/triangle-splatting.triasset";

    static TriBenchGardenSetup()
    {
        EditorApplication.delayCall += EnsureScene;
    }

    [MenuItem("TriBench/Setup Garden Triangle Splatting Scene")]
    public static void EnsureScene()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) return;
        if (File.Exists(Path.GetFullPath(ScenePath)))
        {
            if (SceneManager.GetActiveScene().path != ScenePath)
                EditorSceneManager.OpenScene(ScenePath);
            ConfigureCapture();
            return;
        }

        PlayerSettings.colorSpace = ColorSpace.Linear;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneOSX, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneOSX, new[] { GraphicsDeviceType.Metal });
        QualitySettings.vSyncCount = 0;

        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        scene.name = "TriBenchGarden";

        GameObject cameraObject = new GameObject("Main Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        cameraObject.tag = "MainCamera";
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.015f, 0.02f, 0.025f, 0f);
        camera.fieldOfView = 58f;
        camera.nearClipPlane = 0.01f;
        camera.farClipPlane = 300f;
        cameraObject.transform.position = new Vector3(0.6f, 4.2f, -28f);
        cameraObject.transform.LookAt(new Vector3(0.6f, 1.5f, 1.6f));

        GameObject garden = new GameObject("Garden - Triangle Splatting");
        TriangleSplattingTriAssetRenderer renderer = garden.AddComponent<TriangleSplattingTriAssetRenderer>();
        renderer.AssetDirectory = AssetPath;
        renderer.TargetCamera = camera;
        renderer.PrimitiveCount = 4517295;
        renderer.SortFrontToBack = true;

        ConfigureCapture();
        Selection.activeGameObject = garden;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(ScenePath)));
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log("[TriBench] Garden scene created. Press Play to load and render the complete .triasset.");
    }

    static void ConfigureCapture()
    {
        GameObject garden = GameObject.Find("Garden - Triangle Splatting");
        if (garden == null) return;
        TriAssetRenderer renderer = garden.GetComponent<TriAssetRenderer>();
        ColmapBatchCapture capture = garden.GetComponent<ColmapBatchCapture>();
        if (capture == null) capture = garden.AddComponent<ColmapBatchCapture>();
        capture.Renderer = renderer;
        capture.CaptureCamera = Camera.main;
        capture.AutoStart = true;
        EditorUtility.SetDirty(garden);
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        EditorSceneManager.SaveOpenScenes();
    }
}
