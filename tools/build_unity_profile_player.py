#!/usr/bin/env python3
"""Build the TriBench standalone Unity profile Player."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def unity_graphics_arguments() -> list[str]:
    return ["-force-vulkan"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--unity-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="linux64", choices=("linux64", "macos", "windows64"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--display", help="Linux X display used by Unity Editor; defaults to DISPLAY or :0.")
    parser.add_argument("--log-name", default="unity_profile_player_build.log")
    args = parser.parse_args()

    unity = args.unity.expanduser().resolve()
    project = args.unity_project.expanduser().resolve()
    output = args.output.expanduser().resolve()
    log_path = output.parent / args.log_name
    if output.is_file() and not args.force:
        print(output)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(unity), "-batchmode", *unity_graphics_arguments(), "-projectPath", str(project),
        "-executeMethod", "TriBench.UnityNative.Editor.TriBenchProfilePlayerBuild.Build",
        "-profile-player-output", str(output),
        "-profile-player-target", args.target,
        "-logFile", str(log_path),
        "-quit",
    ]
    print("[TriBench] Building Unity profile Player:", " ".join(command), flush=True)
    env = os.environ.copy()
    if sys.platform.startswith("linux") and not env.get("DISPLAY"):
        env["DISPLAY"] = args.display or ":0"
    elif args.display:
        env["DISPLAY"] = args.display
    subprocess.run(command, check=True, env=env)
    if not output.is_file():
        raise FileNotFoundError(f"Unity Player build did not create executable: {output}")
    output.chmod(output.stat().st_mode | 0o111)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
