"""Unity video trajectory rendering utilities for MeshSplatBench.

Handles PCA ellipse camera trajectory generation, Unity batch invocation,
direct frame streaming, and MP4 video encoding for both method-aware and
general-purpose Unity native renderer conditions. Frames are streamed directly
into the video encoder without saving intermediate PNG images to disk unless
explicitly requested via --write-frames.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
from PIL import Image


STREAM_MAGIC = 0x4D534256  # "MSBV" in little-endian uint32
SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR_SCENES = frozenset(("bonsai", "counter", "kitchen", "room"))


def find_unity_executable() -> Path | None:
    """Locate the Unity Editor executable on the local system."""
    env_unity = os.environ.get("UNITY")
    if env_unity and Path(env_unity).is_file():
        return Path(env_unity)

    # Search standard macOS locations
    if sys.platform == "darwin":
        candidates = sorted(
            Path("/Applications/Unity/Hub/Editor").glob("*/Unity.app/Contents/MacOS/Unity"),
            reverse=True,
        )
        if candidates:
            return candidates[0]

    # Search standard Linux locations
    if sys.platform.startswith("linux"):
        hub_dir = Path.home() / "Unity/Hub/Editor"
        if hub_dir.is_dir():
            candidates = sorted(hub_dir.glob("*/Editor/Unity"), reverse=True)
            if candidates:
                return candidates[0]

    unity_in_path = shutil.which("Unity") or shutil.which("unity")
    if unity_in_path:
        path = Path(unity_in_path).resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def find_unity_project() -> Path | None:
    """Locate the MeshSplatBench Unity project root."""
    env_proj = os.environ.get("PROJECT") or os.environ.get("UNITY_PROJECT")
    if env_proj and Path(env_proj).is_dir():
        return Path(env_proj).resolve()

    # Check relative to cwd
    cwd_candidate = Path.cwd() / "unity"
    if (cwd_candidate / "Assets").is_dir() and (cwd_candidate / "ProjectSettings").is_dir():
        return cwd_candidate.resolve()

    # Check relative to msbench package location
    pkg_candidate = Path(__file__).resolve().parents[1] / "unity"
    if (pkg_candidate / "Assets").is_dir() and (pkg_candidate / "ProjectSettings").is_dir():
        return pkg_candidate.resolve()
    return None


def reference_image_directory(scene: str) -> str:
    """Match the MipNeRF360 resolution convention used by native evaluation."""
    return "images_2" if scene in INDOOR_SCENES else "images_4"


def unity_graphics_arguments() -> list[str]:
    """Select the native Unity graphics API for the current host platform."""
    if sys.platform == "darwin":
        return ["-force-metal"]
    if sys.platform.startswith("linux"):
        return ["-force-vulkan"]
    return []


def unity_failure_summary(
    log_path: Path,
    *,
    max_lines: int = 16,
    stdout_lines: list[str] | None = None,
) -> str:
    """Return the useful tail of a Unity failure without dumping its crash trace."""
    matches = []
    needles = (
        "[msbench]", "[meshsplatbench]", "error", "exception", "fatal", "crash", "sigsegv",
        "missing vulkan framebuffer", "shader error", "compilation failed",
        "another unity instance", "multiple unity instances", "aborting batchmode",
    )

    if stdout_lines:
        for line in stdout_lines:
            stripped = line.strip()
            if stripped and any(needle in stripped.lower() for needle in needles):
                matches.append(stripped)

    if log_path.is_file():
        for line in log_path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped and any(needle in stripped.lower() for needle in needles):
                matches.append(stripped)

    if not matches:
        if stdout_lines:
            return "\n".join(stdout_lines[-max_lines:])
        if log_path.is_file():
            return f"No explicit error found; inspect {log_path}"
        return f"Unity log was not created: {log_path}"
    return "\n".join(matches[-max_lines:])


def find_running_unity_processes(unity_project: Path | None = None) -> list[tuple[int, str]]:
    """Find running Unity Editor/standalone processes associated with this project."""
    pids_found: dict[int, str] = {}

    # 1. Check PIDs directly from lockfiles in unity_project
    if unity_project is not None:
        unity_project = Path(unity_project).resolve()
        for rel in ("Temp/UnityLockfile", "Library/EditorInstance.json"):
            lf = unity_project / rel
            if lf.is_file():
                try:
                    text = lf.read_text(errors="replace")
                    target_pid = None
                    if lf.name == "EditorInstance.json":
                        target_pid = int(json.loads(text).get("process_id", 0))
                    else:
                        for tok in text.split():
                            if tok.isdigit():
                                target_pid = int(tok)
                                break
                    if target_pid and target_pid > 0 and target_pid != os.getpid():
                        try:
                            os.kill(target_pid, 0)
                            pids_found[target_pid] = f"<project lock in {rel}>"
                        except OSError:
                            pass
                except Exception:
                    pass

    # 2. Check running processes via ps
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            check=False,
        )
        proj_str = str(unity_project.resolve()) if unity_project is not None else None
        proj_name = unity_project.name if unity_project is not None else None
        for line in res.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid, cmd = int(parts[0]), parts[1]
            if pid == os.getpid():
                continue
            cmd_lower = cmd.lower()
            if any(ign in cmd_lower for ign in ("python", "pytest", "bash", "sh ", "git ")):
                continue
            is_unity = (
                "unity.app" in cmd_lower
                or "/unity" in cmd_lower
                or "editor/unity" in cmd_lower
                or cmd.startswith("Unity")
                or cmd.startswith("unity")
            )
            if is_unity:
                if proj_str is None or (proj_str in cmd) or (proj_name and f"-projectpath {proj_name}" in cmd_lower):
                    pids_found[pid] = cmd
    except Exception:
        pass

    return sorted(pids_found.items())


def clear_stale_unity_locks(unity_project: Path) -> list[Path]:
    """Remove Unity lockfiles if the process holding them is no longer running."""
    removed = []
    unity_project = Path(unity_project).resolve()
    for rel in ("Temp/UnityLockfile", "Library/EditorInstance.json"):
        lf = unity_project / rel
        if not lf.is_file():
            continue
        pid = None
        try:
            text = lf.read_text(errors="replace")
            if lf.name == "EditorInstance.json":
                pid = int(json.loads(text).get("process_id", 0))
            else:
                for tok in text.split():
                    if tok.isdigit():
                        pid = int(tok)
                        break
        except Exception:
            pass

        is_alive = False
        if pid and pid > 0:
            try:
                os.kill(pid, 0)
                is_alive = True
            except OSError:
                is_alive = False

        if not is_alive:
            try:
                lf.unlink(missing_ok=True)
                removed.append(lf)
            except Exception:
                pass
    return removed


def terminate_existing_unity_processes(unity_project: Path | None = None) -> list[int]:
    """Terminate existing Unity processes using the given project."""
    import signal

    running = find_running_unity_processes(unity_project)
    killed = []
    for pid, _ in running:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            pass
    if killed:
        time.sleep(1.0)
        for pid in killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if unity_project is not None:
        clear_stale_unity_locks(unity_project)
    return killed


def read_exact(stream: Any, n: int) -> bytes:
    """Read exactly n bytes from a socket, file, or stream.

    Raises EOFError if fewer bytes were returned before stream closure.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    use_socket = hasattr(stream, "recv_into")
    while pos < n:
        if use_socket:
            chunk = stream.recv_into(view[pos:])
        elif hasattr(stream, "readinto"):
            chunk = stream.readinto(view[pos:])
        else:
            raw = stream.read(n - pos)
            if not raw:
                break
            view[pos : pos + len(raw)] = raw
            chunk = len(raw)
        if not chunk:
            break
        pos += chunk
    if pos < n:
        raise EOFError(f"Unexpected end of stream: expected {n} bytes, got {pos}")
    return bytes(buf)


def decode_stream_to_video(
    stream: Any,
    output_video: Path,
    *,
    expected_frames: int | None = None,
    expected_fps: int | None = None,
    write_frames: bool = False,
    frames_dir: Path | None = None,
    log_path: Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Decode raw RGB frame stream directly into an MP4 video file.

    Reads 20-byte stream header:
        magic (uint32), width (uint32), height (uint32), total_frames (uint32), fps (uint32)
    Followed by total_frames of:
        frame_idx (uint32), width * height * 3 bytes (RGB24).

    Streams frames directly into the video encoder without saving intermediate PNGs.
    Uses a background encoder thread to decouple socket I/O from ffmpeg encoding,
    preventing TCP backpressure from stalling the Unity render loop.
    """
    import queue
    import threading

    output_video = Path(output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if write_frames and frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    header_bytes = read_exact(stream, 20)
    magic, width, height, total_frames, stream_fps = struct.unpack("<IIIII", header_bytes)
    if magic != STREAM_MAGIC:
        raise ValueError(
            f"Invalid stream magic header: 0x{magic:08X}, expected 0x{STREAM_MAGIC:08X} ('MSBV')"
        )

    if total_frames == 0:
        raise RuntimeError("Stream indicated 0 total frames.")

    fps = stream_fps if stream_fps > 0 else (expected_fps or 30)

    try:
        import imageio.v2 as iio
        writer_factory = iio.get_writer
        imwrite = None
    except ImportError:
        try:
            import imageio.v3 as iio_v3
            writer_factory = None
            imwrite = iio_v3.imwrite
        except ImportError as exc:
            raise ImportError("decode_stream_to_video requires imageio. Install imageio[ffmpeg].") from exc

    writer = (
        writer_factory(str(output_video), fps=fps, macro_block_size=None)
        if writer_factory is not None
        else None
    )
    frame_list = [] if writer is None else None

    sample_indices = {0, total_frames // 2, max(0, total_frames - 1)}
    solid_checks: list[bool] = []
    frame_bytes_len = width * height * 3

    # --- P2 optimisation: background encoder thread ---
    # Queue capacity limits memory to ~30 frames; blocks the receiver thread
    # if ffmpeg falls behind, providing natural backpressure without stalling Unity.
    _ENCODE_QUEUE_SIZE = 30
    encode_queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(
        maxsize=_ENCODE_QUEUE_SIZE
    )
    encoder_error: list[BaseException] = []  # mutable container for thread-safe error passing

    def _encoder_worker() -> None:
        """Background worker that drains the encode queue into the video writer."""
        try:
            while True:
                item = encode_queue.get()
                if item is None:  # sentinel → end of stream
                    break
                frame_idx, arr = item
                if writer is not None:
                    writer.append_data(arr)
                elif frame_list is not None:
                    frame_list.append(arr)
                if write_frames and frames_dir is not None:
                    Image.fromarray(arr).save(frames_dir / f"{frame_idx:05d}.png")
        except BaseException as exc:
            encoder_error.append(exc)

    encoder_thread = threading.Thread(target=_encoder_worker, daemon=True, name="video-encoder")
    encoder_thread.start()

    try:
        for idx in range(total_frames):
            frame_hdr = read_exact(stream, 4)
            actual_idx = struct.unpack("<I", frame_hdr)[0]
            body = read_exact(stream, frame_bytes_len)
            arr = np.frombuffer(body, dtype=np.uint8).reshape((height, width, 3))

            # H.264 / libx264 (yuv420p) requires width and height to be divisible by 2.
            # Crop the trailing odd pixel if necessary (e.g. 1237x822 -> 1236x822).
            if (arr.shape[1] % 2 != 0) or (arr.shape[0] % 2 != 0):
                arr = arr[: arr.shape[0] - (arr.shape[0] % 2), : arr.shape[1] - (arr.shape[1] % 2)]

            if idx in sample_indices:
                solid_checks.append(bool(arr.min() == arr.max()))

            # Enqueue for background encoding (blocks if queue full → backpressure)
            encode_queue.put((actual_idx, arr))

            if show_progress and ((idx + 1) % 15 == 0 or idx == total_frames - 1):
                pct = (idx + 1) / total_frames * 100.0
                print(
                    f"[MeshSplatBench] Synthesized video frame {idx + 1}/{total_frames} ({pct:.1f}%)",
                    flush=True,
                )
    finally:
        # Signal encoder to finish and wait for it
        encode_queue.put(None)
        encoder_thread.join(timeout=120)
        if writer is not None:
            writer.close()

    # Re-raise encoder thread errors
    if encoder_error:
        raise RuntimeError(f"Video encoder thread failed: {encoder_error[0]}") from encoder_error[0]

    if imwrite is not None and frame_list is not None:
        imwrite(str(output_video), np.asarray(frame_list), fps=fps)

    if len(solid_checks) >= 3 and all(solid_checks):
        log_info = f" Inspect {log_path}" if log_path and log_path.is_file() else ""
        raise RuntimeError(
            f"Unity rendered only solid-color frames; no geometry reached the camera.{log_info}"
        )

    return output_video


def decode_raw_stream_file(
    raw_path: Path,
    output_video: Path,
    *,
    expected_frames: int | None = None,
    expected_fps: int | None = None,
    write_frames: bool = False,
    frames_dir: Path | None = None,
    log_path: Path | None = None,
    delete_raw_on_success: bool = True,
) -> Path:
    """Decode a fallback .raw stream file into an MP4 video."""
    raw_path = Path(raw_path).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw stream file not found: {raw_path}")
    with open(raw_path, "rb") as f:
        out = decode_stream_to_video(
            f,
            output_video,
            expected_frames=expected_frames,
            expected_fps=expected_fps,
            write_frames=write_frames,
            frames_dir=frames_dir,
            log_path=log_path,
        )
    if delete_raw_on_success:
        raw_path.unlink(missing_ok=True)
    return out


def validate_captured_frames(frames_dir: Path, expected_count: int, *, newer_than: float | None = None) -> list[Path]:
    """Validate that rendered PNG frames exist and are non-empty."""
    images = sorted(frames_dir.glob("*.png"))
    if newer_than is not None:
        images = [p for p in images if p.stat().st_mtime >= newer_than - 1.0]
    if len(images) < expected_count:
        raise RuntimeError(
            f"Expected {expected_count} rendered frames, but found {len(images)} under {frames_dir}"
        )

    # Check a sample for all-solid false-successes
    sampled = [images[0], images[len(images) // 2], images[-1]]
    solid = []
    for path in sampled:
        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
        solid.append(all(low == high for low, high in extrema))
    if all(solid):
        raise RuntimeError(
            "Unity rendered only solid-color frames; no geometry reached the camera. "
            f"Inspect {frames_dir.parent / 'unity_editor.log'}"
        )
    return images


def encode_video_from_frames(
    frames_dir: Path,
    output_video: Path,
    fps: int = 30,
) -> Path:
    """Encode PNG frames in a directory into an MP4 video."""
    images = sorted(frames_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG frames to encode in {frames_dir}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio
        writer_factory = iio.get_writer
        imwrite = None
    except ImportError:
        try:
            import imageio.v3 as iio_v3
            writer_factory = None
            imwrite = iio_v3.imwrite
        except ImportError as exc:
            raise ImportError("encode_video_from_frames requires imageio. Install imageio[ffmpeg].") from exc

    writer = writer_factory(str(output_video), fps=fps, macro_block_size=None) if writer_factory is not None else None
    frame_list = [] if writer is None else None
    try:
        for p in images:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGB"))
            if (arr.shape[1] % 2 != 0) or (arr.shape[0] % 2 != 0):
                arr = arr[: arr.shape[0] - (arr.shape[0] % 2), : arr.shape[1] - (arr.shape[1] % 2)]
            if writer is not None:
                writer.append_data(arr)
            else:
                frame_list.append(arr)
    finally:
        if writer is not None:
            writer.close()
    if imwrite is not None and frame_list is not None:
        imwrite(str(output_video), np.asarray(frame_list), fps=fps)
    return output_video


def generate_trajectory_from_dataset(
    dataset_path: str | Path,
    *,
    dataset_type: str = "auto",
    split: str = "train",
    image_dir: str = "images",
    resolution: int = 1,
    eval_every: int = 8,
    frames: int = 240,
    fps: int = 30,
    zoom: float = 1.0,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> tuple[dict[str, Any], list[Any]]:
    """Build the exact PCA ellipse camera trajectory as msbench render video."""
    from msbench.core.datasets import load_dataset
    from msbench.core.rendering import generate_ellipse_cameras

    ds = load_dataset(
        dataset_path,
        dataset_type=dataset_type,
        split=split,
        eval_every=eval_every,
        image_dir=image_dir,
        resolution=resolution,
    )
    base_cameras = ds.get_all_cameras()
    cam_batches = generate_ellipse_cameras(
        base_cameras,
        n_frames=frames,
        zoom=zoom,
        z_variation=z_variation,
        z_phase=z_phase,
    )

    w = int(cam_batches[0].width)
    h = int(cam_batches[0].height)
    traj_frames = []
    for cam in cam_batches:
        c2w = cam.camtoworlds[0].detach().cpu().numpy()
        K = cam.Ks[0].detach().cpu().numpy()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        traj_frames.append({
            "pose": c2w.flatten().tolist(),
            "intrinsics": [fx, fy, cx, cy],
            "image_size": [w, h],
        })

    trajectory: dict[str, Any] = {
        "format": "nerfbaselines-v1",
        "camera_model": "pinhole",
        "image_size": [w, h],
        "fps": fps,
        "frames": traj_frames,
    }
    return trajectory, cam_batches


def run_unity_video_for_scene(
    *,
    unity_bin: Path,
    unity_project: Path,
    method: str,
    triasset_path: Path,
    dataset_path: Path | None,
    output_dir: Path,
    condition: str = "method-aware",
    topology: str = "indexed",
    indexed_mesh_method_aware: bool = False,
    trajectory_path: Path | None = None,
    dataset_type: str = "auto",
    image_dir: str = "images",
    resolution: int = 1,
    eval_every: int = 8,
    frames: int = 240,
    fps: int = 30,
    zoom: float = 1.0,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    split: str = "train",
    write_frames: bool = False,
    background_color: str | None = None,
    log_name: str = "unity_editor.log",
    force_unity: bool = False,
) -> Path:
    """Run Unity in headless batchmode to render a video trajectory.

    Streams rendered frames directly into the video encoder in memory without saving
    intermediate PNG files to disk.
    """
    unity_bin = Path(unity_bin).resolve()
    unity_project = Path(unity_project).resolve()
    triasset_path = Path(triasset_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    log_path = output_dir / log_name
    video_path = output_dir / "render_traj.mp4"
    raw_stream_path = output_dir / "render_traj.raw"

    # Pre-execution lock and process management
    clear_stale_unity_locks(unity_project)
    if force_unity:
        killed = terminate_existing_unity_processes(unity_project)
        if killed:
            print(f"[MeshSplatBench] Terminated existing Unity process(es): {killed}", flush=True)
            time.sleep(1.0)
            clear_stale_unity_locks(unity_project)
    else:
        conflicting = find_running_unity_processes(unity_project)
        if conflicting:
            pids_desc = "\n".join(f"  - PID {p}: {c[:90]}..." for p, c in conflicting)
            raise RuntimeError(
                f"Another Unity instance is currently running with project '{unity_project}' open:\n"
                f"{pids_desc}\n\n"
                f"Multiple Unity instances cannot open the same project.\n"
                f"To automatically terminate conflicting Unity processes, re-run with --force-unity.\n"
                f"Or stop manually: kill -9 {' '.join(str(p) for p, _ in conflicting)}"
            )

    log_path.unlink(missing_ok=True)

    manifest_path = triasset_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing triasset manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    rendering = manifest.get("rendering", {})
    general = rendering.get("general_purpose", {})
    if condition == "general-purpose" and not general.get("supported", False):
        raise RuntimeError(
            f"{method} cannot enter general-purpose video rendering: "
            f"{general.get('reason', 'manifest does not declare a comparable appearance field')}"
        )

    if background_color is None:
        background_color = rendering.get("background_color", "black")

    # 1. Resolve / generate trajectory JSON
    target_traj_json = output_dir / "trajectory.json"
    if trajectory_path is not None and Path(trajectory_path).is_file():
        trajectory_data = json.loads(Path(trajectory_path).read_text())
        if target_traj_json != Path(trajectory_path).resolve():
            shutil.copyfile(trajectory_path, target_traj_json)
        total_frames = len(trajectory_data.get("frames", []))
        fps = int(trajectory_data.get("fps", fps))
    else:
        if dataset_path is None or not Path(dataset_path).is_dir():
            raise ValueError("Either --trajectory or a valid --dataset path must be supplied.")
        trajectory_data, _ = generate_trajectory_from_dataset(
            dataset_path,
            dataset_type=dataset_type,
            split=split,
            image_dir=image_dir,
            resolution=resolution,
            eval_every=eval_every,
            frames=frames,
            fps=fps,
            zoom=zoom,
            z_variation=z_variation,
            z_phase=z_phase,
        )
        target_traj_json.write_text(json.dumps(trajectory_data, indent=2) + "\n")
        total_frames = len(trajectory_data["frames"])

    # 2. Prepare markers
    marker = output_dir / ".unity_video_complete"
    marker.unlink(missing_ok=True)
    alt_marker = output_dir / ".unity_capture_complete"
    alt_marker.unlink(missing_ok=True)

    # 3. Create TCP stream listener
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    stream_port = server_sock.getsockname()[1]

    # 4. Assemble Unity invocation
    command = [
        str(unity_bin),
        "-batchmode",
        *unity_graphics_arguments(),
        "-projectPath", str(unity_project),
        "-executeMethod", "MeshSplatBench.UnityNative.Editor.TriAssetBatchRunner.RunVideo",
        "-method", method,
        "-triasset", str(triasset_path),
        "-video-trajectory", str(target_traj_json),
        "-output", str(output_dir),
        "-topology", topology,
        "-background-color", str(background_color),
        "-video-stream-port", str(stream_port),
        "-logFile", str(log_path),
    ]
    if condition == "general-purpose":
        command.extend(("-standard-mesh", "1"))
    else:
        command.extend(("-method-specific", "1"))
    if indexed_mesh_method_aware:
        command.extend(("-indexed-mesh-method-aware", "1"))
    if write_frames:
        command.append("-write-frames")

    stream_result: dict[str, Any] = {
        "success": False,
        "connected": False,
        "error": None,
    }
    stop_event = threading.Event()

    process: subprocess.Popen | None = None
    stdout_lines: list[str] = []

    def _receiver_worker() -> None:
        conn = None
        try:
            while not stop_event.is_set():
                if process is not None and process.poll() is not None:
                    # Unity process terminated before connecting
                    break
                try:
                    conn, _ = server_sock.accept()
                    stream_result["connected"] = True
                    break
                except socket.timeout:
                    continue
                except OSError:
                    break

            if conn is None:
                return

            conn.settimeout(180.0)
            decode_stream_to_video(
                conn,
                video_path,
                expected_frames=total_frames,
                expected_fps=fps,
                write_frames=write_frames,
                frames_dir=frames_dir,
                log_path=log_path,
            )
            stream_result["success"] = True
        except Exception as exc:
            stream_result["error"] = exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    receiver_thread = threading.Thread(target=_receiver_worker, daemon=True)
    receiver_thread.start()

    print(
        f"[MeshSplatBench] Launching Unity video render: method={method}, "
        f"condition={condition}, frames={total_frames}, fps={fps}, stream_port={stream_port}, output={output_dir}",
        flush=True,
    )
    started = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _stdout_reader() -> None:
        try:
            if process.stdout is not None:
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    line_str = line.rstrip()
                    stdout_lines.append(line_str)
                    if any(k in line_str.lower() for k in ("aborting", "fatal error", "multiple unity instances")):
                        print(f"[Unity Batchmode] {line_str}", flush=True)
        except Exception:
            pass

    stdout_thread = threading.Thread(target=_stdout_reader, daemon=True)
    stdout_thread.start()

    try:
        while process.poll() is None:
            if stream_result["error"] is not None:
                # An error occurred in stream decoding
                break
            if marker.is_file() or alt_marker.is_file():
                # Unity finished rendering
                break
            time.sleep(0.5)

        if (marker.is_file() or alt_marker.is_file()) and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()

        exit_code = process.poll()
        if exit_code is None:
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10)
    except KeyboardInterrupt:
        print("\n[MeshSplatBench] Interrupted by user. Terminating Unity process...", flush=True)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        stop_event.set()
        try:
            server_sock.close()
        except Exception:
            pass
        receiver_thread.join(timeout=15)
        stdout_thread.join(timeout=5)

    # 5. Handle fallbacks if direct TCP streaming did not complete
    if not stream_result["success"] and raw_stream_path.is_file() and raw_stream_path.stat().st_size > 20:
        print(f"[MeshSplatBench] Synthesizing video from raw stream fallback: {raw_stream_path}...", flush=True)
        decode_raw_stream_file(
            raw_stream_path,
            video_path,
            expected_frames=total_frames,
            expected_fps=fps,
            write_frames=write_frames,
            frames_dir=frames_dir,
            log_path=log_path,
        )
        stream_result["success"] = True

    if not stream_result["success"] and frames_dir.is_dir():
        png_count = len(list(frames_dir.glob("*.png")))
        if png_count >= total_frames:
            print(f"[MeshSplatBench] Synthesizing video from PNG frames in {frames_dir}...", flush=True)
            validate_captured_frames(frames_dir, total_frames, newer_than=started)
            encode_video_from_frames(frames_dir, video_path, fps=fps)
            stream_result["success"] = True

    # 6. Cleanup frames directory if not requested
    if not write_frames and frames_dir.is_dir():
        shutil.rmtree(frames_dir, ignore_errors=True)

    if not marker.is_file() and not alt_marker.is_file() and not stream_result["success"]:
        if any("multiple unity instances cannot open the same project" in s.lower() for s in stdout_lines):
            raise RuntimeError(
                f"Unity aborted because another Unity instance is already running with project '{unity_project}' open.\n"
                f"Multiple Unity instances cannot open the same project.\n\n"
                f"To fix this:\n"
                f"  1. Run: pkill -9 -f Unity\n"
                f"  2. Remove any stale lock: rm -f '{unity_project}/Temp/UnityLockfile'\n"
                f"  3. Or retry with '--force-unity' to terminate conflicting processes automatically."
            )

        err_info = f"\nStream error: {stream_result['error']}" if stream_result["error"] else ""
        combined_tail = unity_failure_summary(log_path, stdout_lines=stdout_lines)
        raise RuntimeError(
            f"Unity video rendering failed without completion marker; exit={exit_code}{err_info}\n"
            f"Relevant log / stdout lines:\n{combined_tail}"
        )

    if stream_result["error"] is not None and not stream_result["success"]:
        raise stream_result["error"]

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video file was not created or is empty: {video_path}")

    print(f"[MeshSplatBench] Unity video saved: {video_path}", flush=True)
    return video_path
