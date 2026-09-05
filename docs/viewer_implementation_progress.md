# NerfBaselines-Compatible Viewer 实现进度

> 实时跟踪 `nerfbaselines_compatible_viewer_plan.md` 中各阶段的实现状态。
> 更新时间：2026-07-01

## 总体状态

| 阶段 | 名称 | 状态 | 说明 |
|------|------|------|------|
| 1 | 静态资源迁移 | ✅ 完成 | viewer_static/ 保留22个必要文件，3DGS本地渲染入口已禁用并移除静态文件 |
| 2 | 双点云导出 | ✅ 完成 | viewer_geometry.py + 独立 CLI 导出工具 |
| 3 | train/test dataset | ✅ 完成 | viewer.py 同时加载多个split |
| 4 | geometry endpoint | ✅ 完成 | /dataset/pointcloud.ply + /dataset/geometry/* 路由 |
| 5 | HTTP remote renderer | ✅ 完成 | POST /render 兼容nerfbaselines pose格式 |
| 6 | WebSocket renderer | ✅ 完成 | /render-websocket + 完整RFC 6455实现 |
| 7 | Keyframe/轨迹 | ✅ 基础完成 | 复用前端camera path UI，新增 viewer-path 离线渲染 |
| 8 | Mesh 适配 | ✅ 基础完成 | mesh PLY/GLB best-effort导出 + mesh preview参数 |
| 9 | 配置与 CLI | ✅ 完成 | viewer 读取显式 geometry 参数；几何导出拆分到 tools/export_viewer_geometry.py |
| 10 | 测试矩阵 | ✅ 完成 | 327项pytest：324 passed, 3 skipped |

## macOS 无 CUDA 适配

MeshSplatBench 原有 adapter 依赖 CUDA rasterizer。在 macOS 上通过以下策略实现无 CUDA 运行：

- 几何导出（Phase 2）纯 CPU 操作，不依赖 CUDA ✅
- viewer server（静态文件、dataset.json、PLY endpoint）纯 CPU ✅
- 实际渲染（POST /render）使用 DummyRenderer 返回占位图，保证 viewer UI 可用 ✅
- 当检测到 CUDA 不可用时自动回退到 DummyRenderer ✅
- `msbench/renderers/dummy.py` (125行)：CPU-only渲染器，返回梯度占位图 + 相机位姿叠加文字
- `DummyRenderer.render()` 返回 NCHW [1,C,H,W] 张量，`_to_hwc()` 转换为 [H,W,C] 供 PIL 编码

## 实施顺序（按计划推荐）

1. ✅ 阶段 2：双点云导出
2. ✅ 阶段 3 + 4：train/test dataset + pointcloud endpoint
3. ✅ 阶段 1 + 5：迁移 nerfbaselines static viewer + HTTP remote renderer
4. ✅ 阶段 6：WebSocket
5. ✅ 阶段 9：配置与 CLI 集成
6. ✅ 阶段 7/8：高级功能基础实现
7. ✅ 阶段 10：测试矩阵

## 冒烟测试结果（macOS, 无 CUDA）

10项HTTP端点测试全部通过：

```
Test 1:  GET  /                         → 200, 37,439 bytes (index.html)
Test 2:  GET  /dataset.json             → 200, 29 bytes
Test 3:  GET  /dataset/pointcloud.ply   → 200, 32,621 bytes
Test 4:  POST /render (color)           → 200, 8,077 bytes
Test 5:  POST /render (depth)           → 200, 1,830 bytes
Test 6:  POST /render (alpha)           → 200, 1,830 bytes
Test 7:  POST /render (normal)          → 200, 1,829 bytes
Test 8:  POST /render (split compare)   → 200, 7,062 bytes
Test 9:  GET  /viewer.js                → 200, 163,289 bytes
Test 10: GET  /styles.css               → 200, 15,352 bytes
```

## pytest 验证结果（Linux, triben-dev）

完整测试套件已通过：

```
324 passed, 3 skipped in 8.92s
```

覆盖范围包括：

- viewer HTTP/WebSocket request 构造、渲染输出编码、split compare、dataset manifest。
- 双点云 PLY schema、mesh PLY、geometry metadata/attributes。
- nerfbaselines-v1 trajectory 读写、CameraBatch 转换、离线帧渲染。
- 现有 cameras/config/renderers/metrics/training loop 等回归测试。

## 文件清单

### 新增文件

| 文件路径 | 行数 | 说明 |
|----------|------|------|
| `msbench/core/viewer.py` | 1,108 | 完全重写：nerfbaselines兼容的HTTP+WS viewer server |
| `msbench/core/viewer_geometry.py` | 核心模块 | 双点云导出与原始几何导出 |
| `msbench/core/trajectory.py` | 279 | nerfbaselines-v1 trajectory读写和离线渲染 |
| `msbench/core/_websocket.py` | 1,732 | RFC 6455 WebSocket实现（从nerfbaselines复制） |
| `msbench/core/palettes.json` | — | 色彩调色板定义（从nerfbaselines复制） |
| `msbench/renderers/dummy.py` | 125 | CPU-only DummyRenderer（macOS无CUDA回退） |
| `tools/export_viewer_geometry.py` | 独立入口 | 几何导出CLI工具 |
| `msbench/core/viewer_static/` | 22文件 | nerfbaselines前端必要静态资源 |

### 修改文件

| 文件路径 | 说明 |
|----------|------|
| `msbench/cli/render.py` | viewer命令新增geometry/split参数、CUDA自动回退、viewer-path离线轨迹渲染 |
| `msbench/core/rendering.py` | render_video兼容imageio.v2 writer和imageio.v3.imwrite |

### viewer_static/ 目录结构

```
viewer_static/
├── index.html              # nerfbaselines viewer入口（模板注入）
├── viewer.js               # 主viewer逻辑 (163KB)
├── controls.js             # 相机控制器
├── threejs_utils.js        # Three.js工具函数
├── interpolation.js        # 相机插值
├── mesh.js                 # Mesh显示
├── styles.css              # 样式表
├── favicon.ico
└── third-party/
    ├── three.module.js     # Three.js r128+
    ├── tabler-icons.min.css
    ├── client-zip.js       # ZIP打包
    ├── mp4-muxer.js        # MP4封装
    ├── webm-muxer.js       # WebM封装
    ├── es-module-shims.wasm.js
    ├── LICENSE
    ├── fonts/              # tabler-icons字体 (仅woff2)
    ├── lines/              # Three.js Line2系列 (5个文件)
    └── loaders/PLYLoader.js
```

## 详细日志

### Phase 1: 静态资源迁移 — ✅ 完成

- [x] 从 nerfbaselines 迁移前端文件到 `msbench/core/viewer_static/`，仅保留22个必要静态资源
- [x] 包含 Three.js、viewer.js、controls.js、PLYLoader 等核心组件
- [x] 保留 third-party/ 子目录结构（fonts/、lines/、loaders/）
- [x] 禁用 `GET /3dgs.js`，MeshSplatBench viewer 不暴露 3DGS local renderer。
- [x] 物理移除 `viewer_static/3dgs.js` 与 gaussian-splats 第三方文件；Tabler 字体仅保留 woff2。

### Phase 2: 双点云导出 — ✅ 完成

- [x] 创建 `msbench/core/viewer_geometry.py`
  - `primitive_to_point_cloud()`: 从任意 BasePrimitive 采样三角形表面点
  - `_sample_triangle_surface()`: 面积加权重心坐标采样
  - `write_point_cloud_ply()` / `write_mesh_ply()`: PLY写入
  - `export_viewer_geometry()`: 一站式导出接口
  - 导出: `geometry_viewer_points.ply` (500K), `geometry_cg_points.ply` (3M), `geometry_metadata.json`, `geometry_attributes.npz`, 可选 `geometry_mesh.ply`
  - `GeometryExportConfig` dataclass: viewer_num_points=500K, cg_num_points=3M
  - `export_viewer_point_cloud()`: 只导出 viewer 所需的 `geometry_viewer_points.ply`
  - `export_original_geometry()`: 不采样、不降采样，按训练后的 primitive 原始拓扑导出 `geometry_original.ply`
  - Color modes: dc (SH DC), opacity, white, rendered
  - Normal modes: none, primitive, estimated
  - 体素降采样用于viewer点云
- [x] 创建 `tools/export_viewer_geometry.py` 独立导出入口
  - CLI: `--config` 或 `--method --checkpoint` 输入
  - 默认导出原始 `geometry_original.ply`，不施加额外点数限制
  - 参数: `--filename`, `--export-glb`

### Phase 3: train/test dataset 加载 — ✅ 完成

- [x] `serve_viewer()` 接受 `train_dataset` 和 `test_dataset` 参数
- [x] `_ViewerBackend` 存储两个split的dataset字典
- [x] `_build_dataset_json()` 输出 `{"train": {"cameras": [...]}, "test": {"cameras": [...]}, "metadata": {...}}`
- [x] CLI `--viewer-splits` 参数控制加载哪些split（默认 "train,test"）

### Phase 4: geometry endpoint — ✅ 完成

- [x] `GET /dataset/pointcloud.ply` → 主viewer点云
- [x] `GET /dataset/geometry/metadata.json` → 几何元数据
- [x] `GET /dataset/geometry/attributes.npz` → 属性数组
- [x] `GET /dataset/geometry/viewer_points.ply` → 轻量viewer点云
- [x] `GET /dataset/geometry/cg_points.ply` → 高保真CG点云
- [x] `GET /dataset/geometry/mesh.ply` → 可选mesh
- [x] `GET /dataset/images/{split}/{idx}.jpg` → GT图像

### Phase 5: HTTP remote renderer — ✅ 完成

- [x] `POST /render` 接受nerfbaselines pose格式（3×4 cam-to-world矩阵扁平化为12个float）
- [x] `_intrinsics_matrix()` 同时支持 `[fx, fy, cx, cy]` 和 3×3 矩阵
- [x] `_pose_matrix()` 从请求中提取并构造4×4 cam-to-world矩阵
- [x] `_camera_from_request()` 构造 CameraBatch
- [x] `_render_viewer_frame()` 处理渲染请求，支持 color/depth/alpha/normal 输出
- [x] `_combine_outputs()` 实现nerfbaselines风格的倾斜分割对比
- [x] `_format_output()` 使用 `_to_hwc()` 转换NCHW→HWC供PIL编码
- [x] COOP/COEP headers 支持SharedArrayBuffer

### Phase 6: WebSocket renderer — ✅ 完成

- [x] 复制 nerfbaselines `_websocket.py` (1,732行) 完整RFC 6455实现
- [x] `GET /render-websocket` WebSocket升级路由
- [x] `handle_websocket_message()` 处理渲染请求
- [x] 消息格式: `uint32 header_length + JSON header + image payload`
- [x] PerMessageDeflate 扩展支持

### Phase 7: Keyframe、轨迹和视频导出 — ✅ 基础完成

- [x] 前端 keyframe/camera path UI 由 nerfbaselines static viewer 提供。
- [x] 新增 `msbench/core/trajectory.py`
  - `load_trajectory()` / `save_trajectory()` 支持 `nerfbaselines-v1` JSON。
  - `trajectory_cameras()` 将 trajectory frame 转成 `CameraBatch`。
  - `render_trajectory_frames()` 支持按 trajectory 离线渲染帧序列和MP4。
- [x] 新增 `msbench render viewer-path`
  - `--trajectory/-t` 读取 viewer 导出的 camera path。
  - `--output-types color,depth,alpha,normal` 支持多输出类型。
  - 适配 CUDA 不可用时的 DummyRenderer fallback。

### Phase 8: Mesh 相关适配 — ✅ 基础完成

- [x] `viewer_geometry.py` 支持 `geometry_mesh.ply` best-effort 导出。
- [x] 对暴露 `V/F` 的 primitive 优先导出 indexed mesh。
- [x] 对 triangle soup primitive 回退到独立顶点 face PLY。
- [x] `write_mesh_glb()` 支持可选 `geometry_mesh.glb`，便于 Unity/Blender 使用。
- [x] viewer server 保留 `/dataset/geometry/mesh.ply` 和 `/dataset/geometry/mesh.glb` 支持。
- [x] viewer CLI 不再接收 CG/mesh 几何参数，避免 CG 导出资产进入 viewer 流程。

### Phase 9: 配置与 CLI 集成 — ✅ 完成

- [x] `msbench render viewer` 命令新增参数:
  - `--viewer-splits` (默认 "train,test")
  - `--geometry` 指定 viewer 点云，默认自动生成/复用 `geometry_viewer_points.ply`
  - `--auto-export-geometry` / `--no-auto-export-geometry`
- [x] `tools/export_viewer_geometry.py` 只负责导出 CG 软件用的原始几何资产
- [x] viewer 启动时自动发现 `<output.dir>/viewer/geometry_viewer_points.ply`；缺失时只生成 viewer 点云
- [x] `_try_build_adapter()` / `_try_load_adapter()`: CUDA不可用时自动回退到DummyRenderer
- [x] 配置文件中 `render.viewer` section 支持 host/port/max_render_size/jpeg_quality/geometry 路径

### Phase 10: 测试矩阵 — ✅ 完成

- [x] 10项HTTP端点冒烟测试通过（GET/POST/WebSocket路由）
- [x] 几何导出验证：PLY文件可被plyfile读取，顶点数和属性正确
- [x] 单元测试（pytest）
- [x] viewer/geometry/trajectory 集成级测试
- [x] WebSocket message封包和错误路径测试
- [x] 完整项目回归测试：324 passed, 3 skipped

## 待办事项

1. 使用真实MeshSplatBench checkpoint测试完整viewer流程，尤其是四个方法的自由视角交互。
2. 用真实 `mesh-splatting` checkpoint 验证 indexed mesh PLY/GLB 与 Unity/Blender 导入效果。
3. 用浏览器端手动验收 keyframe preview、前端 mp4/webm/png zip 导出。
