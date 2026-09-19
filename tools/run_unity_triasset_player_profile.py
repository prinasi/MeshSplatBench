#!/usr/bin/env python3
"""Run standalone Unity Player profiles for MeshSplatBench TriAssets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


SCENES = ("bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill")
INDOOR_SCENES = frozenset(("bonsai", "counter", "kitchen", "room"))


def reference_image_directory(scene: str, method: str = "triangle-splatting") -> str:
    if method.replace("_", "-").lower() == "diffsoup":
        return "images_4"
    return "images_2" if scene in INDOOR_SCENES else "images_4"


def unity_graphics_arguments() -> list[str]:
    if sys.platform.startswith("linux"):
        return ["-force-vulkan"]
    if sys.platform == "darwin":
        return ["-force-metal"]
    return []


def valid_gpu_ms(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and 0.01 < float(value) < 1000.0 else None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def player_failure_summary(log_paths: list[Path], *, max_lines: int = 16) -> str:
    needles = (
        "[msbench]", "[meshsplatbench]", "error", "exception", "fatal", "crash", "gpu frametiming",
        "vulkan", "failed", "display", "x11", "wayland",
    )

    matches = []
    existing = [path for path in log_paths if path.is_file()]
    if not existing:
        return "Unity Player logs were not created: " + ", ".join(str(path) for path in log_paths)
    for path in existing:
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped and any(needle in stripped.lower() for needle in needles):
                matches.append(stripped)
    return "\n".join(matches[-max_lines:]) if matches else "No explicit error found; inspect " + ", ".join(str(path) for path in existing)


def derive_total_timeout(
    views: int,
    warmup_frames: int,
    timed_frames: int,
    *,
    frame_budget: float = 0.5,
    fixed: float = 600.0,
) -> float:
    """Hard cap for one Player profile run: workload estimate plus fixed startup/asset-load slack."""
    return views * (warmup_frames + timed_frames) * frame_budget + fixed


def derive_stall_timeout(
    views: int,
    warmup_frames: int,
    timed_frames: int,
    *,
    frame_budget: float = 2.0,
    minimum: float = 180.0,
    slack: float = 60.0,
) -> float:
    """Seconds without any progress signal before the Player is declared stalled.

    Between the per-view milestone logs the longest legitimate silence is one
    view's warmup + timed frames, so the budget is derived from that interval.
    """
    return max(minimum, views * (warmup_frames + timed_frames) * frame_budget + slack)


def profile_failure_message(
    *,
    exit_code: int | None,
    marker: Path,
    profile_path: Path,
    log_path: Path,
    console_log_path: Path,
) -> str:
    note = ""
    if not marker.is_file() and profile_path.is_file():
        note = (
            f"\n  The runtime profile JSON exists but the completion marker is missing: "
            f"the Player wrote {profile_path} and then stalled or failed on the quit path."
        )
    return (
        f"Unity Player profile failed: marker={marker.is_file()} exit={exit_code}{note}\n"
        f"Relevant Player log lines ({log_path}, {console_log_path}):\n"
        f"{player_failure_summary([log_path, console_log_path])}"
    )


class PlayerProfileSupervisor:
    """Bounds a standalone Unity Player profile run and localizes stalls.

    Progress signals are: growth of the Player log or console log, a new
    [MeshSplatBench] milestone line in the Player log, or appearance/growth of the
    runtime profile JSON.  Three independent deadlines are enforced:

    - startup timeout: time until the first progress signal.  Failure means the
      Player never reached the profile scene (engine/scene-load stall).
    - stall timeout: maximum consecutive time without any progress signal.
      Failure means the Player froze mid-load or mid-profile.
    - total timeout: hard cap on the whole run derived from the configured
      views/warmup/timed-frame workload.

    On timeout the Player is terminated/killed and a RuntimeError with
    diagnostics (elapsed time, liveness, exit code, marker/JSON state, recent
    milestones, and relevant log lines) is raised.
    """

    def __init__(
        self,
        *,
        process: subprocess.Popen,
        marker: Path,
        profile_path: Path,
        log_path: Path,
        console_log_path: Path,
        startup_timeout: float,
        stall_timeout: float,
        total_timeout: float,
        poll_interval: float = 1.0,
        heartbeat_interval: float = 60.0,
        verbose: bool = False,
        scene: str = "",
    ) -> None:
        if min(startup_timeout, stall_timeout, total_timeout) <= 0:
            raise ValueError("player profile timeouts must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.process = process
        self.marker = marker
        self.profile_path = profile_path
        self.log_path = log_path
        self.console_log_path = console_log_path
        self.startup_timeout = startup_timeout
        self.stall_timeout = stall_timeout
        self.total_timeout = total_timeout
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.verbose = verbose
        self.scene = scene
        self._log_offset = 0
        self._milestones: list[tuple[float, str]] = []

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size if path.is_file() else 0
        except OSError:
            return 0

    def _read_milestones(self) -> list[str]:
        """Return new [MeshSplatBench] lines appended to the Player log since last read."""
        size = self._file_size(self.log_path)
        if size < self._log_offset:
            self._log_offset = 0
        if size <= self._log_offset:
            return []
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._log_offset)
                chunk = handle.read(size - self._log_offset)
            self._log_offset = size
        except OSError:
            return []
        return [
            line.strip()
            for line in chunk.splitlines()
            if line.strip() and any(tag in line.strip().lower() for tag in ("[msbench]", "[meshsplatbench]"))
        ]




    def _kill(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        self.process.kill()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _timeout_message(self, stage: str, started: float, now: float) -> str:
        elapsed = now - started
        profile_detail = "absent"
        if self.profile_path.is_file():
            try:
                age = max(0.0, now - self.profile_path.stat().st_mtime)
                profile_detail = f"present ({self.profile_path.stat().st_size} bytes, written {age:.0f}s ago)"
            except OSError:
                profile_detail = "present (unreadable)"
        lines = [
            f"Unity Player profile timed out: stage={stage} scene={self.scene}",
            f"  elapsed={elapsed:.1f}s process_alive={self.process.poll() is None} exit_code={self.process.poll()}",
            f"  completion_marker={self.marker.is_file()}",
            f"  profile_json={profile_detail}",
        ]
        if stage == "startup":
            lines.append(
                "  No MeshSplatBench milestone or profile JSON appeared; the Player likely never "
                "reached the profile scene (engine startup/scene load stall)."
            )
        if self._milestones:
            lines.append("  Recent MeshSplatBench milestones:")
            for when, text in self._milestones[-6:]:
                lines.append(f"    +{when - started:6.1f}s {text}")
        else:
            lines.append("  No MeshSplatBench milestones were logged.")
        lines.append(f"  Relevant Player log lines ({self.log_path}, {self.console_log_path}):")
        lines.append(player_failure_summary([self.log_path, self.console_log_path]))
        return "\n".join(lines)

    def _status_line(self, started: float, now: float) -> str:
        last_milestone = self._milestones[-1][1] if self._milestones else "<none>"
        return (
            f"[MeshSplatBench] Player profile in progress scene={self.scene} elapsed={now - started:.0f}s "
            f"log_bytes={self._file_size(self.log_path)} json={self.profile_path.is_file()} "
            f"last_milestone={last_milestone[-90:]}"
        )

    _PRIMITIVE_COUNT_RE = re.compile(r"primitiveCount=(\d+)")
    _BUFFER_LOAD_SECONDS_PER_MILLION = 120.0
    _BUFFER_LOAD_MIN_EXTRA = 60.0

    def _effective_stall_timeout(self) -> float:
        """Return the stall timeout, extended if a large buffer load is in progress."""
        for _, text in reversed(self._milestones):
            if "Loading learned render buffers" in text:
                m = self._PRIMITIVE_COUNT_RE.search(text)
                if m:
                    millions = int(m.group(1)) / 1_000_000.0
                    extra = max(
                        self._BUFFER_LOAD_MIN_EXTRA,
                        millions * self._BUFFER_LOAD_SECONDS_PER_MILLION,
                    )
                    return self.stall_timeout + extra
                return self.stall_timeout + self._BUFFER_LOAD_MIN_EXTRA
            if "Renderer ready" not in text and "CaptureAll" not in text:
                break
        return self.stall_timeout

    def wait(self) -> int:
        """Wait until the completion marker appears or the Player exits.

        Returns the Player exit code if it has exited (0 if it is still alive
        when the marker appears).  Raises RuntimeError with diagnostics on any
        timeout; the Player has been killed in that case.
        """
        started = time.time()
        deadline_total = started + self.total_timeout
        deadline_startup = started + self.startup_timeout
        last_progress = started
        last_heartbeat = started
        first_progress = False
        log_size = self._file_size(self.log_path)
        console_size = self._file_size(self.console_log_path)
        profile_size = self._file_size(self.profile_path)
        while self.process.poll() is None and not self.marker.is_file():
            time.sleep(self.poll_interval)
            now = time.time()
            exit_code = self.process.poll()
            if exit_code is not None:
                return exit_code
            new_milestones = self._read_milestones()
            if new_milestones:
                first_progress = True
                for line in new_milestones:
                    self._milestones.append((now, line))
                self._milestones = self._milestones[-20:]
                if self.verbose:
                    for line in new_milestones:
                        print(f"[MeshSplatBench] player milestone: {line}", flush=True)
            new_log_size = self._file_size(self.log_path)
            new_console_size = self._file_size(self.console_log_path)
            new_profile_size = self._file_size(self.profile_path)
            if self.profile_path.is_file():
                first_progress = True
            grew = new_milestones or new_log_size > log_size or new_console_size > console_size
            profile_grew = new_profile_size > profile_size
            if grew or profile_grew:
                last_progress = now
            log_size, console_size, profile_size = new_log_size, new_console_size, new_profile_size
            exit_code = self.process.poll()
            if exit_code is not None:
                return exit_code
            if not first_progress and now >= deadline_startup:
                self._kill()
                raise RuntimeError(self._timeout_message("startup", started, now))
            effective_stall = self._effective_stall_timeout()
            if now - last_progress >= effective_stall:
                self._kill()
                raise RuntimeError(self._timeout_message("stall", started, now))
            if now >= deadline_total:
                self._kill()
                raise RuntimeError(self._timeout_message("total", started, now))
            if not self.verbose and now - last_heartbeat >= self.heartbeat_interval:
                print(self._status_line(started, now), flush=True)
                last_heartbeat = now
        return self.process.poll() if self.process.poll() is not None else 0


def profile_summary(path: Path) -> dict[str, float | int]:
    payload = json.loads(path.read_text())
    gpu = []
    for view in payload.get("views", []):
        for raw in view.get("gpu_frame_samples_ms", []):
            value = valid_gpu_ms(raw)
            if value is not None:
                gpu.append(value)
    p50 = quantile(gpu, 0.50)
    p95 = quantile(gpu, 0.95)
    return {
        "gpu_samples": len(gpu),
        "gpu_p50_ms": p50,
        "gpu_p95_ms": p95,
        "gpu_fps": 1000.0 / p50 if p50 > 0.0 else 0.0,
        "rendering_memory_mib": float(payload.get("graphics_driver_allocated_bytes", 0)) / (1024.0 * 1024.0),
    }


def validate_profile(
    path: Path,
    *,
    expected_views: int,
    expected_frames: int,
    newer_than: float,
    require_gpu_timing: bool,
    gpu_timing_min_fraction: float,
) -> None:
    if not path.is_file() or path.stat().st_mtime < newer_than - 1.0:
        raise RuntimeError(f"Unity Player profile is missing or stale: {path}")
    payload = json.loads(path.read_text())
    views = payload.get("views", [])
    if len(views) != expected_views:
        raise RuntimeError(f"Unity Player profile has {len(views)} views, expected {expected_views}: {path}")
    cpu_samples = sum(len(view.get("cpu_frame_samples_ms", [])) for view in views)
    expected_samples = expected_views * expected_frames
    if cpu_samples < expected_samples:
        raise RuntimeError(
            f"Unity Player profile has incomplete CPU/engine timing: {cpu_samples}/{expected_samples} samples"
        )
    per_view_gpu = [
        sum(
            1
            for value in view.get("gpu_frame_samples_ms", [])
            if valid_gpu_ms(value) is not None
        )
        for view in views
    ]
    gpu_samples = sum(per_view_gpu)
    min_per_view = max(1, int(expected_frames * min(gpu_timing_min_fraction, 0.25)))
    if require_gpu_timing and (
        gpu_samples < expected_samples * gpu_timing_min_fraction
        or any(count < min_per_view for count in per_view_gpu)
    ):
        raise RuntimeError(
            "Unity Player did not produce enough GPU frame timings: "
            f"{gpu_samples}/{expected_samples} valid samples, per_view={per_view_gpu}. "
            "This usually means the Player was launched without a real graphics device/display, "
            "or the graphics API/backend does not expose FrameTimingManager.gpuFrameTime for this run."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/triangle-splatting/mipnerf360"))
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-name", default="unity_method_aware")
    parser.add_argument("--log-name", default="unity_player_profile.log")
    parser.add_argument("--method", default="triangle-splatting")
    parser.add_argument("--asset-subdir", default="unity_native")
    parser.add_argument(
        "--topology",
        choices=("indexed", "mesh", "soup"),
        default="indexed",
        help="Unity mesh layout intervention. 'mesh' is an alias for indexed; 'soup' de-indexes shared vertices at load time.",
    )
    parser.add_argument(
        "--indexed-mesh-method-aware",
        action="store_true",
        help="For mesh-splatting + --general-purpose, use a true Unity indexed MeshRenderer with method-aware SH appearance.",
    )
    parser.add_argument("--reference-image-dir")
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--profile-run", type=int, default=1)
    parser.add_argument("--profile-views", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=60)
    parser.add_argument("--profile-frames", type=int, default=180)
    parser.add_argument("--batchmode", action="store_true", help="Launch the Player with -batchmode; do not use -nographics.")
    parser.add_argument("--display", help="Linux X display used by the Player; defaults to DISPLAY or :0.")
    parser.add_argument("--show-console-output", action="store_true", help="Do not redirect Player stdout/stderr into a per-scene console log.")
    parser.add_argument("--require-gpu-timing", action="store_true")
    parser.add_argument("--gpu-timing-min-fraction", type=float, default=0.8)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=300.0,
        help="Seconds until the first MeshSplatBench progress signal before aborting the run (default: 300).",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=0.0,
        help="Maximum consecutive seconds without progress before aborting; 0 derives it from the workload (default: 0).",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=0.0,
        help="Hard cap in seconds for the whole profile run; 0 derives it from the workload (default: 0).",
    )
    parser.add_argument(
        "--frame-budget",
        type=float,
        default=0.5,
        help="Assumed worst-case seconds per profiled frame when deriving --total-timeout (default: 0.5).",
    )
    parser.add_argument(
        "--stall-frame-budget",
        type=float,
        default=2.0,
        help="Assumed worst-case seconds per profiled frame when deriving --stall-timeout (default: 2.0).",
    )
    parser.add_argument(
        "--verbose-wait",
        action="store_true",
        help="Print every MeshSplatBench milestone line as it appears in the Player log.",
    )
    condition = parser.add_mutually_exclusive_group(required=True)
    condition.add_argument("--general-purpose", "--standard-mesh", dest="general_purpose", action="store_true")
    condition.add_argument("--method-aware", "--method-specific", dest="method_aware", action="store_true")
    args = parser.parse_args()

    if args.profile_run <= 0 or args.profile_views <= 0 or args.profile_warmup <= 0 or args.profile_frames <= 0:
        parser.error("profile run/views/warmup/frames must be positive")
    if not 0.0 < args.gpu_timing_min_fraction <= 1.0:
        parser.error("--gpu-timing-min-fraction must be in (0, 1]")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.stall_timeout < 0 or args.total_timeout < 0:
        parser.error("--stall-timeout and --total-timeout must be non-negative (0 derives a default)")
    if args.frame_budget <= 0 or args.stall_frame_budget <= 0:
        parser.error("--frame-budget and --stall-frame-budget must be positive")

    player = args.player.expanduser().resolve()
    if not player.is_file():
        raise FileNotFoundError(f"missing Unity Player executable: {player}")
    args.outputs_root = args.outputs_root.resolve()
    args.datasets_root = args.datasets_root.resolve()
    if args.topology == "mesh":
        args.topology = "indexed"

    try:
        from tools.validate_unity_triasset_cpu import validate_triasset
    except ModuleNotFoundError:
        from validate_unity_triasset_cpu import validate_triasset

    for scene in args.scenes:
        scene_root = args.outputs_root / scene
        triasset = (scene_root / args.asset_subdir / f"{args.method}.triasset").resolve()
        dataset = (args.datasets_root / scene).resolve()
        output = (scene_root / args.output_name).resolve()
        image_dir = dataset / (
            args.reference_image_dir or reference_image_directory(scene, args.method)
        )
        if not (triasset / "manifest.json").is_file():
            raise FileNotFoundError(f"missing triasset: {triasset}")
        validate_triasset(triasset, max_faces=10000)
        manifest_path = triasset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rendering = manifest["rendering"]
        topology_label = str(rendering.get("export_topology") or args.topology)
        general = rendering.get("general_purpose", {})
        if args.general_purpose and not general.get("supported", False):
            raise RuntimeError(
                f"{args.method}/{scene} cannot enter the general-purpose benchmark: "
                f"{general.get('reason', 'manifest does not declare a comparable appearance field')}"
            )
        if not (dataset / "sparse/0/cameras.bin").is_file() or not (dataset / "sparse/0/images.bin").is_file():
            raise FileNotFoundError(f"missing COLMAP sparse binaries: {dataset}")
        reference_image = next((p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if reference_image is None:
            raise FileNotFoundError(f"{image_dir.name} is empty: {dataset}")
        with Image.open(reference_image) as image:
            width, height = image.size

        output.mkdir(parents=True, exist_ok=True)
        log_path = output / args.log_name
        console_log_path = output / f"{args.log_name}.console.log"
        profile_path = output / f"runtime_profile_run_{args.profile_run:02d}.json"
        marker = output / ".unity_capture_complete"
        profile_path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        run_protocol = {
            "asset": str(triasset),
            "asset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": manifest.get("source", {}).get("checkpoint_sha256"),
            "schema_version": manifest.get("schema_version"),
            "export_contract_revision": manifest.get("export_contract_revision"),
            "method": args.method,
            "renderer_condition": "general-purpose" if args.general_purpose else "method-aware",
            "indexed_mesh_method_aware": args.indexed_mesh_method_aware,
            "mesh_topology_layout": topology_label,
            "runtime_topology_layout": args.topology,
            "method_aware_status": rendering.get("unity_method_aware_status"),
            "cuda_equivalent": False,
            "requires_per_camera_depth_sort": rendering.get("unity_method_aware_requires_per_camera_depth_sort", False),
            "background_color": rendering.get("background_color"),
            "dataset": str(dataset),
            "resolution": [width, height],
            "execution_mode": "profile",
            "profile": {
                "run": args.profile_run,
                "views": args.profile_views,
                "warmup_frames": args.profile_warmup,
                "timed_frames": args.profile_frames,
                "runtime": "standalone-player",
                "gpu_timing_required": args.require_gpu_timing,
                "gpu_timing_min_fraction": args.gpu_timing_min_fraction,
            },
        }
        (output / "msbench_run_protocol.json").write_text(json.dumps(run_protocol, indent=2) + "\n")
        command = [
            str(player),
            *unity_graphics_arguments(),
            "-screen-fullscreen", "0",
            "-screen-width", str(width),
            "-screen-height", str(height),
            "-method", args.method,
            "-triasset", str(triasset),
            "-dataset", str(dataset),
            "-output", str(output),
            "-triasset-width", str(width),
            "-triasset-height", str(height),
            "-profile-only", "1",
            "-profile-run", str(args.profile_run),
            "-profile-views", str(args.profile_views),
            "-profile-warmup", str(args.profile_warmup),
            "-profile-frames", str(args.profile_frames),
            "-topology", args.topology,
            "-background-color", str(rendering.get("background_color", "black")),
            "-logFile", str(log_path),
        ]
        if args.batchmode:
            command.insert(1, "-batchmode")
        if args.general_purpose:
            command.extend(("-standard-mesh", "1"))
        if args.indexed_mesh_method_aware:
            command.extend(("-indexed-mesh-method-aware", "1"))
        if args.method_aware:
            command.extend(("-method-specific", "1"))

        print(f"[MeshSplatBench] Player profile scene={scene}, reference={image_dir.name}, resolution={width}x{height}", flush=True)
        started = time.time()
        env = os.environ.copy()
        if sys.platform.startswith("linux") and not env.get("DISPLAY"):
            env["DISPLAY"] = args.display or ":0"
        elif args.display:
            env["DISPLAY"] = args.display
        console_handle = None
        if args.show_console_output:
            process = subprocess.Popen(command, env=env)
        else:
            console_handle = console_log_path.open("w", encoding="utf-8", errors="replace")
            process = subprocess.Popen(command, env=env, stdout=console_handle, stderr=subprocess.STDOUT)
        total_timeout = args.total_timeout or derive_total_timeout(
            args.profile_views, args.profile_warmup, args.profile_frames, frame_budget=args.frame_budget
        )
        stall_timeout = args.stall_timeout or derive_stall_timeout(
            args.profile_views, args.profile_warmup, args.profile_frames, frame_budget=args.stall_frame_budget
        )
        print(
            f"[MeshSplatBench] Player profile supervision scene={scene}: "
            f"startup_timeout={args.startup_timeout:g}s stall_timeout={stall_timeout:g}s "
            f"total_timeout={total_timeout:g}s",
            flush=True,
        )
        supervisor = PlayerProfileSupervisor(
            process=process,
            marker=marker,
            profile_path=profile_path,
            log_path=log_path,
            console_log_path=console_log_path,
            startup_timeout=args.startup_timeout,
            stall_timeout=stall_timeout,
            total_timeout=total_timeout,
            verbose=args.verbose_wait,
            scene=scene,
        )
        try:
            supervisor.wait()
        except RuntimeError:
            if console_handle is not None:
                console_handle.close()
            raise
        try:
            try:
                exit_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    exit_code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=10)
        finally:
            if console_handle is not None:
                console_handle.close()
        if not marker.is_file() or exit_code != 0:
            raise RuntimeError(
                profile_failure_message(
                    exit_code=exit_code,
                    marker=marker,
                    profile_path=profile_path,
                    log_path=log_path,
                    console_log_path=console_log_path,
                )
            )
        validate_profile(
            profile_path,
            expected_views=args.profile_views,
            expected_frames=args.profile_frames,
            newer_than=started,
            require_gpu_timing=args.require_gpu_timing,
            gpu_timing_min_fraction=args.gpu_timing_min_fraction,
        )
        summary = profile_summary(profile_path)
        print(
            f"[MeshSplatBench] Player profile complete scene={scene} run={args.profile_run} "
            f"gpu_fps={summary['gpu_fps']:.2f} gpu_p50_ms={summary['gpu_p50_ms']:.3f} "
            f"gpu_p95_ms={summary['gpu_p95_ms']:.3f} gpu_samples={summary['gpu_samples']} "
            f"render_mib={summary['rendering_memory_mib']:.1f} log={log_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
