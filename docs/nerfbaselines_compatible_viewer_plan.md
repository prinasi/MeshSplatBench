# NerfBaselines-Compatible MeshSplatBench Viewer 适配计划

## 目标与范围

目标是在 MeshSplatBench 中实现一个尽可能接近 `nerfbaselines/nerfbaselines/viewer/` 体验的 viewer，但只服务当前四个方法：

- `triangle-splatting`
- `mesh-splatting`
- `2dts`
- `diffsoup`

我们不适配 nerfbaselines 的 3DGS 本地渲染器。MeshSplatBench 的核心交互渲染仍通过后端 `RendererAdapter.render(CameraBatch)` 完成；前端主要复用 nerfbaselines 的 Three.js viewer 能力，包括自由相机、train/test camera frustum、点云/mesh 预览、keyframe、轨迹和远程渲染协议。

整体策略是“协议兼容优先，模型格式解耦”：先把四个方法的最终权重导出成两类标准点云资产，再让 viewer 只消费轻量 viewer 点云作为 3D 场景锚点。真实图像仍由当前 adapter 远程渲染；高保真 CG 点云和可选 mesh 资产则用于 Unity、Blender、MeshLab 等外部软件检查与复用。

## 兼容性边界

### 计划适配

- 自由视角远程渲染。
- train/test 相机 frustum 显示与点击跳转。
- dataset 图片和缩略图预览。
- 双点云 PLY 预览与导出。
- 可选 mesh PLY/GLB 导出。
- HTTP `POST /render` 协议。
- WebSocket `/render-websocket` 协议。
- `color`、`depth`、`alpha/accumulation`、`normal` 输出切换。
- 输出 colormap、range 控制和 split compare。
- keyframe、camera path、轨迹预览和前端视频导出。
- `viewer_initial_pose`、`viewer_transform` 风格的初始化参数。

### 不作为目标

- 不适配 nerfbaselines 的 3DGS local renderer。
- 不要求支持任意 nerfbaselines method spec。
- 不要求完全支持 appearance embedding；只有在 MeshSplatBench adapter 统一暴露后再做。
- 不默认启用 Cloudflare public URL；可作为后续可选功能。

## 当前主要差距

当前 MeshSplatBench viewer 仍以 `<img>` 展示远程渲染结果，缺少完整 Three.js 场景。它有 `POST /render` 的雏形，但没有 nerfbaselines 前端所依赖的完整环境：

- `/dataset.json` 只返回一个 split，默认通常是 `test`。
- 没有 train/test 同时加载。
- 没有 `/dataset/pointcloud.ply` 或统一 geometry endpoint。
- 没有服务 nerfbaselines static assets。
- 没有 WebSocket renderer。
- 没有相机 frustum、point cloud、raycast target、trajectory UI。
- intrinsics 协议当前是 flat 3x3，而 nerfbaselines 前端更常用 `[fx, fy, cx, cy]`。

因此后续重点不是继续增强 `<img>` 版 UI，而是迁移到 nerfbaselines-compatible static viewer。

## Geometry 资产设计原则

我们不建议把所有信息塞进一个 PLY。PLY 在不同软件中的兼容性差异很大，尤其是自定义属性、opacity、SH、primitive id 和 triangle soup face。更稳妥的做法是导出两种点云，并把研究属性放到 sidecar 文件中：

1. `geometry_viewer_points.ply`
   - 目标：服务 nerfbaselines-compatible viewer 的实时交互。
   - 特点：轻量、顶点-only、加载快，默认 30 万到 100 万点。
   - schema：只使用主流 PLY 字段 `x y z red green blue`，可选 `nx ny nz`。
   - 用途：默认映射为 `/dataset/pointcloud.ply`，用于相机 frustum、orbit target、场景范围估计和点云预览。
2. `geometry_cg_points.ply`
   - 目标：服务 Unity、Blender、MeshLab、CloudCompare、Open3D 等 CG/点云工具。
   - 特点：高保真、更密集、标准字段优先，可比 viewer 点云大很多。
   - schema：仍尽量只使用 `x y z red green blue` 和可选 normals，避免软件无法识别的自定义 vertex property。
   - 用途：外部软件检查几何质量、坐标系、scale、颜色和资产复用。
3. 可选 mesh 资产
   - `geometry_mesh.ply`：仅当方法能稳定导出 face 时生成，包含标准 `vertex` 和 `face` 元素。
   - `geometry_mesh.glb`：后续增强项，比 PLY 更适合 Unity/Blender 的 mesh、材质和坐标约定。
4. sidecar 文件
   - `geometry_metadata.json`：记录 method、checkpoint、坐标系、up axis、scale、资产路径、采样策略、schema version。
   - `geometry_attributes.npz`：记录不适合进入标准 PLY 的研究属性，例如 opacity、primitive_id、triangle_id、area、SH/DC feature。

这样 viewer 和 CG 工具都可以得到自己最容易消费的资产，同时不会牺牲方法内部调试所需的额外属性。

## 阶段 1：定义兼容协议与静态资源布局

### 任务

1. 新增 viewer static 目录，例如：
   - `msbench/core/viewer_static/`
   - 或 `msbench/viewer/static/`
2. 从 nerfbaselines viewer 迁移必要前端文件：
   - `index.html`
   - `viewer.js`
   - `controls.js`
   - `threejs_utils.js`
   - `interpolation.js`
   - `mesh.js`
   - `styles.css`
   - `third-party/three.module.js`
   - `third-party/tabler-icons.min.css`
   - `third-party/client-zip.js`
   - `third-party/mp4-muxer.js`
   - `third-party/webm-muxer.js`
3. 删除或禁用 3DGS 入口：
   - 不暴露 `set_3dgs_renderer`
   - 不服务 `3dgs.js`
   - 不在 MeshSplatBench params 中生成 3DGS renderer 配置
4. 定义 MeshSplatBench viewer params：
   - `renderer.type = "remote"`
   - `renderer.http_url = "./render"`
   - `renderer.websocket_url = "./render-websocket"`
   - `renderer.output_types = [...]`
   - `dataset.url = "./dataset.json"`
   - `dataset.pointcloud_url = "./dataset/pointcloud.ply"`，如果已导出
   - `state.render_resolution`
   - `state.prerender_resolution`
   - `state.outputs_configuration`
   - `viewer_initial_pose`
   - `viewer_transform`

### 预期效果

打开 MeshSplatBench viewer 时不再进入自制 `<img>` 页面，而是进入 nerfbaselines 风格的完整 Three.js UI。即使暂时还没有点云，也应能看到空 3D 场景、菜单和远程渲染输出。

### 验证

- `GET /` 能返回替换过 `{{ data|safe }}` 的 `index.html`。
- 浏览器不报 missing module。
- UI 中能看到 settings、camera path、dataset 面板。
- `renderer.type=remote` 能触发 HTTP renderer。

## 阶段 2：实现双点云 viewer geometry 导出

### 任务

1. 新增核心导出模块：
   - `msbench/core/viewer_geometry.py`
2. 新增 CLI/tool：
   - `tools/export_viewer_geometry.py`
3. 支持输入方式：
   - `--config configs/...yaml`
   - 或 `--method --checkpoint --output-dir`
4. 输出两类点云：
   - `<run_dir>/viewer/geometry_viewer_points.ply`
   - `<run_dir>/viewer/geometry_cg_points.ply`
   - `--viewer-num-points` 控制 viewer 点云规模，默认 `500000`
   - `--cg-num-points` 控制 CG 点云规模，默认可更大，例如 `3000000`
   - `--color-mode dc|opacity|white|rendered`
   - `--normal-mode none|primitive|estimated`
5. 可选输出 mesh：
   - `<run_dir>/viewer/geometry_mesh.ply`
   - 后续可选 `<run_dir>/viewer/geometry_mesh.glb`
   - `--export-mesh/--no-export-mesh`
   - `--export-glb/--no-export-glb`
6. 定义 PLY schema：
   - viewer/cg 点云必选 vertex 属性：`x y z red green blue`
   - viewer/cg 点云可选 vertex 属性：`nx ny nz`
   - 不把 `opacity`、`primitive_id`、SH、triangle id 放入标准点云 PLY，避免 Unity/Blender 等工具兼容性下降。
   - mesh PLY 使用标准 `vertex` + `face`，不依赖自定义面属性。
7. 导出 sidecar：
   - `geometry_metadata.json`
   - `geometry_attributes.npz`
   - metadata 记录 method、checkpoint、asset paths、num points、has mesh、color mode、normal mode、coordinate system、up axis、scale、schema version。
   - attributes 保存 opacity、primitive_id、triangle_id、area、SH/DC feature 等研究属性。
8. 每个方法的导出策略：
   - `triangle-splatting`
     - 从 `to_primitive()` 或内部 `_triangles_points` 获取三角形。
     - viewer 点云使用面积采样 + 下采样，保证网页加载速度。
     - CG 点云使用更高采样密度，尽量保留薄结构和边缘。
     - 如 triangle soup 拓扑稳定，可额外导出 `geometry_mesh.ply`，否则只导出点云。
     - 颜色优先使用 SH DC / diffuse 近似；不可用时用 opacity colormap。
   - `mesh-splatting`
     - 从 indexed mesh vertices/faces 导出。
     - 必须导出两类 sampled point cloud。
     - 优先导出 `geometry_mesh.ply`，后续可导出 `geometry_mesh.glb`。
   - `2dts`
     - 优先复用 native/exported PLY。
     - 如只有 checkpoint，则从 `to_primitive()` 获取 triangle vertices 和 opacity。
     - 采样三角形表面点并分别写入 viewer/cg 点云。
   - `diffsoup`
     - 从 checkpoint `V/F/alpha_acc/feat_acc` 或 `to_primitive()` 获取 mesh-like triangle soup。
     - 按面积采样三角形表面点，分别生成 viewer/cg 点云。
     - 如果 face 语义稳定，额外导出 `geometry_mesh.ply`。
     - 颜色初期使用 white/opacity，后续可通过若干训练视角渲染反投影估计颜色。

### 预期效果

四个方法都能产出 `geometry_viewer_points.ply` 和 `geometry_cg_points.ply`。viewer 不再需要理解四种模型权重，只要加载 viewer 点云即可获得可交互的 3D 空间锚点；Unity/Blender 等外部工具则使用 CG 点云或可选 mesh 资产。

### 验证

- 对四种 checkpoint 运行：
  - `python tools/export_viewer_geometry.py --config ... --output-dir /tmp/msbench-viewer-geometry`
- 用 `trimesh.load` 或 `plyfile` 能读出 vertex。
- viewer 点云数量符合 `--viewer-num-points`。
- CG 点云数量符合 `--cg-num-points`。
- `mesh-splatting` 可选导出标准 face mesh。
- viewer 点云在 nerfbaselines `PLYLoader` 中能显示颜色。
- CG 点云能被 Blender/MeshLab/Open3D 读取；Unity 侧至少可通过标准 PLY importer 或转换为 GLB 读取。

## 阶段 3：同时加载 train/test dataset

### 任务

1. 修改 `msbench render viewer`：
   - 不再只构建一个 split。
   - 默认同时构建 `train` 和 `test` dataset。
   - 保留 `--split`，但语义改为 initial/active split，而非唯一 split。
2. 新增参数：
   - `--viewer-splits train,test`
   - `--geometry /path/to/geometry_viewer_points.ply`
   - `--cg-geometry /path/to/geometry_cg_points.ply`
   - `--mesh-geometry /path/to/geometry_mesh.ply`
   - `--auto-export-geometry/--no-auto-export-geometry`
3. 修改 `get_viewer_dataset`：
   - 返回 `{ "train": ..., "test": ..., "metadata": ... }`
   - 每个 camera 的 intrinsics 返回 `[fx, fy, cx, cy]`
   - 保留兼容字段可选 `K`
4. 修改 image endpoint：
   - `/dataset/images/train/{idx}.jpg`
   - `/dataset/images/test/{idx}.jpg`
5. dataset metadata 中加入：
   - scene center/radius/up
   - viewer transform
   - initial pose
   - depth range，如可估计

### 预期效果

viewer 中可以显示 train/test 两组 camera frustum。用户可以选择显示训练相机、测试相机，点击任意相机跳转，而不是只能看测试集图像列表。

### 验证

- `/dataset.json` 同时包含 `train` 和 `test`。
- 前端 dataset 面板显示 train/test camera 选项。
- 点击 train/test frustum 可以跳转视角。
- 缩略图能分别加载两个 split。

## 阶段 4：实现 point cloud 与 geometry endpoint

### 任务

1. viewer 启动时确定 viewer 点云：
   - 如果用户传入 `--geometry`，直接使用。
   - 如果配置中已有 `<run_dir>/viewer/geometry_viewer_points.ply`，直接使用。
   - 如果启用 `--auto-export-geometry`，调用阶段 2 的导出逻辑。
2. 服务 nerfbaselines-compatible endpoint：
   - `GET /dataset/pointcloud.ply`
   - 默认返回 `geometry_viewer_points.ply`。
   - 这是前端 DatasetManager 的主入口，不返回高保真 CG 点云，避免网页加载过慢。
3. 服务扩展 geometry endpoint：
   - `GET /dataset/geometry/viewer_points.ply`
   - `GET /dataset/geometry/cg_points.ply`
   - `GET /dataset/geometry/mesh.ply`，如果存在
   - `GET /dataset/geometry/metadata.json`
4. 对大文件增加响应头：
   - `Content-Length`
   - `X-File-Size`
5. params 中设置：
   - `dataset.pointcloud_url = "./dataset/pointcloud.ply"`
   - `state.dataset_has_pointcloud = true`
   - `state.geometry_assets` 记录可下载的 CG 点云和 mesh URL。

### 预期效果

nerfbaselines DatasetManager 能直接加载 MeshSplatBench 导出的 viewer 点云，并在 Three.js 场景中显示点云预览。自由视角操作会围绕实际几何锚点，而不是空白或测试图像。CG 点云和 mesh 资产可作为下载/检查入口提供，但不会拖慢默认 viewer。

### 验证

- `curl /dataset/pointcloud.ply` 得到 viewer 点云 PLY。
- `curl /dataset/geometry/cg_points.ply` 能得到高保真 CG 点云。
- 如果存在 mesh，`curl /dataset/geometry/mesh.ply` 能得到标准 mesh PLY。
- 前端打开 `dataset_show_pointcloud` 后显示几何。
- 点云和相机 frustum 坐标系对齐。

## 阶段 5：兼容 HTTP remote renderer

### 任务

1. 保留并扩展 `POST /render`。
2. 接收 nerfbaselines request 字段：
   - `pose`
   - `image_size`
   - `intrinsics`
   - `output_type`
   - `palette`
   - `output_range`
   - `split_output_type`
   - `split_percentage`
   - `split_tilt`
   - `split_palette`
   - `split_range`
   - `lossless`
3. 将 nerfbaselines pose 转为 MeshSplatBench `CameraBatch`：
   - 明确坐标系转换：Three.js camera pose -> MeshSplatBench c2w。
   - 用单元测试覆盖 identity、dataset camera roundtrip、DTU/Blender/Colmap。
4. 输出格式：
   - `color`: JPEG/PNG
   - scalar output: palette PNG
   - `normal`: normalized RGB PNG
5. split compare：
   - 后端渲染两个 output_type。
   - 按 split mask 合成。

### 预期效果

nerfbaselines 前端在任何自由视角都能调用 MeshSplatBench adapter 得到正确图像。输出类型和 split compare 基本可用。

### 验证

- 从前端自由移动相机，图像随视角变化。
- `color/depth/alpha/normal` 切换正常。
- split compare 滑块正常。
- HTTP 请求错误能在前端显示通知。

## 阶段 6：实现 WebSocket remote renderer

### 任务

1. 从 nerfbaselines `_websocket.py` 移植轻量 WebSocket handler，或实现最小 RFC6455 handler。
2. 新增 endpoint：
   - `GET /render-websocket`
3. 消息格式兼容 nerfbaselines：
   - 前端发送 JSON string。
   - 后端返回：`uint32 header_length + JSON header + image payload`
4. 加并发控制：
   - 单 GPU adapter 默认串行渲染。
   - 丢弃或延迟过期 request。
5. params 中优先设置：
   - `renderer.websocket_url = "./render-websocket"`
   - `renderer.http_url = "./render"`

### 预期效果

viewer 交互更接近 nerfbaselines，连续拖动时延迟更低，不需要每帧创建完整 HTTP 请求。

### 验证

- 前端显示 `frame_renderer_url` 为 websocket。
- 拖动相机时请求稳定，无 WebSocket disconnect。
- HTTP fallback 仍可用。

## 阶段 7：Keyframe、Trajectory 和视频导出

### 任务

1. 保留 nerfbaselines 前端 keyframe UI。
2. 确保 `compute_camera_path` 运行所需状态都存在：
   - resolution
   - framerate
   - fov
   - interpolation mode
3. 后端支持 trajectory render 请求字段：
   - pose
   - fov/aspect -> intrinsics
   - image_size
   - output_type
   - split fields
4. 可选新增后端命令：
   - `msbench render viewer-path --camera-path path.json`
   - 用于离线高质量渲染前端导出的 camera path。

### 预期效果

用户能在 viewer 里添加 keyframe、预览 trajectory，并导出前端视频或帧序列。后续可以把前端 camera path 交给 MeshSplatBench CLI 做离线高质量渲染。

### 验证

- 添加多个 keyframe 后能看到轨迹曲线。
- preview 播放正常。
- 前端导出 mp4/webm/png zip 正常。
- 离线导出读取 camera path 后画面和 preview 一致。

## 阶段 8：Mesh 相关适配

### 任务

1. 对 `mesh-splatting` 优先导出标准 `geometry_mesh.ply`：
   - `vertex` 使用标准坐标、颜色、可选 normals。
   - `face` 使用标准 index list。
   - 不写入 MeshSplatBench 私有 face 属性；私有属性进入 `geometry_attributes.npz`。
2. 对 `triangle-splatting`、`2dts`、`diffsoup`：
   - 默认导出 viewer/cg 两类点云。
   - 只有在 face 语义稳定且不会误导外部软件时，才导出 `geometry_mesh.ply`。
3. 如果 nerfbaselines `mesh.js` local renderer 可以直接加载普通 mesh PLY，则为 `mesh-splatting` 可选启用：
   - `renderer.type = "mesh"` 仅作为预览模式。
   - 真实图像仍使用 remote renderer。
4. 明确坐标系和 scale：
   - geometry PLY、dataset cameras、remote renderer 必须同一坐标系。
   - `geometry_metadata.json` 必须记录 up axis、unit/scale 和 world transform。
5. Unity/Blender 兼容增强：
   - 优先保证 `geometry_cg_points.ply` 是纯标准 PLY。
   - 后续提供 `geometry_mesh.glb`，减少 Unity 对 PLY importer 的依赖。
   - 对 GLB 明确是否执行坐标系转换，例如 MeshSplatBench world -> glTF convention。

### 预期效果

`mesh-splatting` 可以获得更强的 mesh 预览；其他方法至少有 viewer 点云和 CG 点云。用户不需要理解各方法 checkpoint 格式，也可以把导出的 CG 点云或 mesh 放进外部软件检查。

### 验证

- `mesh-splatting` mesh PLY 在 Blender/MeshLab 中可见面结构。
- `triangle-splatting`、`2dts`、`diffsoup` 至少可见 viewer/cg 点云几何。
- 几何位置和远程渲染画面一致。

## 阶段 9：配置与 CLI 集成

### 任务

1. 扩展 base config：
   ```yaml
   render:
     viewer:
       split: test
       splits: [train, test]
       host: 127.0.0.1
       port: 7007
       max_render_size: 1280
       geometry: null              # viewer 点云，默认映射到 /dataset/pointcloud.ply
       cg_geometry: null           # 高保真 CG 点云
       mesh_geometry: null         # 可选 mesh PLY/GLB
       auto_export_geometry: true
       viewer_num_points: 500000
       cg_num_points: 3000000
       geometry_color_mode: dc
       geometry_normal_mode: none
       export_mesh: auto
       export_glb: false
       use_websocket: true
   ```
2. CLI 参数：
   - `--viewer-splits`
   - `--geometry`
   - `--cg-geometry`
   - `--mesh-geometry`
   - `--auto-export-geometry/--no-auto-export-geometry`
   - `--viewer-num-points`
   - `--cg-num-points`
   - `--geometry-color-mode`
   - `--geometry-normal-mode`
   - `--export-mesh/--no-export-mesh`
   - `--export-glb/--no-export-glb`
   - `--use-websocket/--no-websocket`
3. 输出目录：
   - `<run_dir>/viewer/geometry_viewer_points.ply`
   - `<run_dir>/viewer/geometry_cg_points.ply`
   - `<run_dir>/viewer/geometry_mesh.ply`
   - `<run_dir>/viewer/geometry_mesh.glb`
   - `<run_dir>/viewer/geometry_metadata.json`
   - `<run_dir>/viewer/geometry_attributes.npz`
   - `<run_dir>/viewer/viewer_config.json`

### 预期效果

所有 viewer 相关产物都有固定位置。用户可以直接运行：

```bash
msbench render viewer --config configs/triangle-splatting/dtu/scan24.yaml
```

并自动获得 train/test cameras、viewer 点云、CG 点云、可选 mesh、自由视角远程渲染。

### 验证

- 四个方法配置都能启动 viewer。
- 缺失 geometry 时按需自动导出。
- 已存在 geometry 时不重复导出。

## 阶段 10：测试矩阵

### 单元测试

- viewer 点云 PLY schema 写入与读取。
- CG 点云 PLY schema 写入与读取。
- mesh PLY schema 写入与读取，如方法支持 face。
- `geometry_metadata.json` 与 `geometry_attributes.npz` 一致性。
- 每个方法 `to_primitive()` 到 viewer geometry 的转换。
- `/dataset.json` train/test 格式。
- intrinsics `[fx, fy, cx, cy]` 与 `K` 转换。
- pose roundtrip。
- split output 合成。
- WebSocket 消息封包/解包。

### 集成测试

- 启动 viewer server 后请求：
  - `/`
  - `/dataset.json`
  - `/dataset/images/train/0.jpg`
  - `/dataset/images/test/0.jpg`
  - `/dataset/pointcloud.ply`
  - `/dataset/geometry/viewer_points.ply`
  - `/dataset/geometry/cg_points.ply`
  - `/dataset/geometry/metadata.json`
  - `/render`
  - `/render-websocket`
- 用 dummy adapter 验证自由 pose 会改变返回图像。
- 用真实小 checkpoint 做 smoke test。
- 用 `trimesh`、`plyfile` 或 Open3D 读取两类点云。
- 可选用 Blender/Unity 导入 `geometry_cg_points.ply` 或 `geometry_mesh.glb` 做手动兼容验收。

### 手动验收

每个方法至少检查：

- viewer 页面无 console error。
- train/test camera frustum 可显示。
- viewer point cloud 可显示。
- CG point cloud 可被外部工具读取。
- 支持 mesh 的方法可导出并打开 mesh。
- 点击 dataset camera 可跳转。
- 鼠标/键盘可自由移动视角。
- 图像确实随视角变化。
- output type 切换可用。
- keyframe 和 trajectory preview 可用。

## 推荐实施顺序

1. 先做阶段 2：双点云导出。
2. 再做阶段 3 和 4：train/test dataset + pointcloud endpoint。
3. 然后做阶段 1 和 5：迁移 nerfbaselines static viewer + HTTP remote renderer。
4. 确认自由视角、frustum、pointcloud 全部工作后，再做阶段 6 WebSocket。
5. 最后做阶段 7/8/9 的高级功能和 CLI 整理。

这样可以避免一开始就陷入大型前端迁移。双点云资产是最关键的基础设施：viewer 只面对轻量、标准的 `geometry_viewer_points.ply`，外部软件只面对高保真、标准的 `geometry_cg_points.ply`。可选 mesh 资产在此基础上逐步增强，不阻塞自由视角 remote rendering 主线。

## 风险与决策点

- **坐标系风险**：MeshSplatBench adapters、dataset loaders、nerfbaselines Three.js viewer 的坐标约定不完全一致。必须用 dataset camera roundtrip 测试锁死。
- **颜色风险**：triangle/mesh/diffsoup 的真实颜色可能是 view-dependent。viewer geometry 的颜色只用于空间预览，可以先用 DC/opacity/white，不要求与渲染完全一致。
- **性能风险**：大场景 PLY 过大会拖慢前端。viewer 点云需要 `--viewer-num-points`、voxel/downsample、按面积采样；CG 点云可以更大，但不默认加载到浏览器。
- **CG 兼容风险**：Unity/Blender 对 PLY 的自定义属性支持不稳定。标准点云 PLY 不写私有字段，复杂属性进入 sidecar；mesh 长期优先考虑 GLB。
- **WebSocket 并发风险**：四个 adapter 多数依赖 CUDA，全局串行渲染更安全。先保证正确，再优化交互帧率。
- **mesh 兼容风险**：双点云 PLY 是必选；face/mesh PLY/GLB 是增强。不要让 mesh local preview 阻塞 remote rendering 主线。
