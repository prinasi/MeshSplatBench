"""CPU-only tests for canonical Unity deployment evaluation and aggregation."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from tools.aggregate_unity_deployment_profile import aggregate_condition
from tools.aggregate_unity_triasset_eval import main as aggregate_quality_main
from tools.evaluate_deployment_images import evaluate_directories, native_gaussian_ssim
from tools.format_unity_metrics_report import main as format_metrics_report_main
from msbench.vendor.triangle_splatting.utils.loss_utils import ssim as native_ssim


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8), mode="RGB").save(path)


class CanonicalImageEvaluationTests(unittest.TestCase):
    def test_ssim_matches_native_triangle_implementation(self) -> None:
        generator = np.random.default_rng(7)
        prediction = generator.random((16, 19, 3), dtype=np.float32)
        target = generator.random((16, 19, 3), dtype=np.float32)
        expected = native_ssim(
            torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0),
            torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0),
        ).item()
        self.assertAlmostEqual(native_gaussian_ssim(prediction, target), expected, places=7)

    def test_matches_native_prefix_and_scores_both_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rgb(root / "unity/test/frame.png", 128)
            _write_rgb(root / "gt/frame.JPG", 0)
            _write_rgb(root / "native/00000_frame.png", 64)

            report = evaluate_directories(root / "unity", root / "gt", native_dir=root / "native")

            self.assertEqual(report["protocol"]["evaluator_revision"], 2)
            self.assertEqual(report["test"]["count"], 1)
            self.assertEqual(report["native_fidelity"]["count"], 1)
            self.assertGreater(report["native_fidelity"]["psnr"], report["test"]["psnr"])

    def test_strict_pairing_rejects_missing_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rgb(root / "prediction/a.png", 1)
            _write_rgb(root / "prediction/b.png", 2)
            _write_rgb(root / "gt/a.png", 1)
            with self.assertRaisesRegex(ValueError, "missing_gt"):
                evaluate_directories(root / "prediction", root / "gt")

    def test_native_shape_policy_can_crop_native_fidelity_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rgb(root / "unity/frame.png", 128)
            _write_rgb(root / "gt/frame.png", 0)
            (root / "native").mkdir()
            Image.fromarray(np.full((15, 15, 3), 64, dtype=np.uint8), mode="RGB").save(root / "native/frame.png")

            with self.assertRaisesRegex(ValueError, "image shape mismatch"):
                evaluate_directories(root / "unity", root / "gt", native_dir=root / "native")

            report = evaluate_directories(
                root / "unity",
                root / "gt",
                native_dir=root / "native",
                native_shape_policy="crop",
            )

            self.assertEqual(report["test"]["count"], 1)
            self.assertEqual(report["native_fidelity"]["count"], 1)
            adjustments = report["protocol"]["native_shape_adjustments"]
            self.assertEqual(adjustments[0]["policy"], "crop")
            self.assertEqual(adjustments[0]["scored_shape"], [15, 15, 3])


class DeploymentAggregationTests(unittest.TestCase):
    def test_quality_cli_writes_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_root = root / "garden/unity_method_aware_rev2"
            report_root.mkdir(parents=True)
            report_root.joinpath("metrics_summary.json").write_text(json.dumps({
                "protocol": {"evaluator_revision": 1},
                "test": {"count": 2, "psnr": 20.0, "ssim": 0.8},
                "native_fidelity": {"count": 2, "psnr": 30.0, "ssim": 0.9},
            }))
            with patch.object(sys, "argv", [
                "aggregate_unity_triasset_eval.py",
                "--outputs-root", str(root),
                "--output-name", "unity_method_aware_rev2",
                "--scenes", "garden",
            ]):
                self.assertEqual(aggregate_quality_main(), 0)
            self.assertTrue((root / "unity_method_aware_rev2_summary.csv").is_file())
            aggregate = json.loads(
                (root / "unity_method_aware_rev2_summary.json").read_text()
            )
            self.assertEqual(aggregate["test"]["views_total"], 2)
            self.assertEqual(aggregate["test"]["native_psnr_macro"], 30.0)

    def test_requires_exact_runs_views_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "mesh-splatting/method-aware/garden"
            payload = {
                "views": [{
                    "cpu_frame_samples_ms": [1.0, 2.0, 3.0],
                    "gpu_frame_samples_ms": [0.5, 0.6, 0.7],
                }],
                "load_to_ready_ms": 100,
                "graphics_driver_allocated_bytes": 1024,
                "asset_bytes": 2048,
            }
            for run in range(2):
                path = scene / f"run_{run:02d}/runtime_profile_run_{run:02d}.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))

            report = aggregate_condition(
                root, "mesh-splatting", "method-aware", "garden",
                expected_runs=2, expected_views=1, expected_frames=3, gpu_fraction=0.8,
            )
            self.assertEqual(report["runs"], 2)
            self.assertEqual(report["gpu_complete_runs"], 2)
            self.assertIsNotNone(report["gpu_p95_ms"])

            with self.assertRaisesRegex(ValueError, "expected 3 independent runs"):
                aggregate_condition(
                    root, "mesh-splatting", "method-aware", "garden",
                    expected_runs=3, expected_views=1, expected_frames=3, gpu_fraction=0.8,
                )

    def test_unity_metrics_formatter_writes_paper_ready_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "bicycle/unity_method_aware"
            asset = root / "bicycle/unity_native/2dts.triasset"
            report_dir = root / "reports"
            asset.mkdir(parents=True)
            (asset / "manifest.json").write_text("{}")
            (asset / "buffers.bin").write_bytes(b"x" * 1024)
            result.mkdir(parents=True)
            (result / "msbench_run_protocol.json").write_text(json.dumps({"asset": str(asset)}))
            (result / "metrics_summary.json").write_text(json.dumps({
                "test": {"count": 2, "psnr": 21.0, "ssim": 0.81, "lpips_vgg": 0.12},
                "native_fidelity": {"psnr": 31.0, "ssim": 0.91, "lpips_vgg": 0.02},
            }))
            (result / "fps_per_test_view.csv").write_text(
                "view,mean_frame_ms,fps\n"
                "a,10.0,100.0\n"
                "b,20.0,50.0\n"
            )
            (result / "runtime_profile_run_01.json").write_text(json.dumps({
                "views": [{
                    "cpu_frame_samples_ms": [10.0, 20.0],
                    "gpu_frame_samples_ms": [5.0, 7.0],
                }],
                "load_to_ready_ms": 123.0,
                "graphics_driver_allocated_bytes": 2 * 1024 * 1024,
                "unity_total_allocated_bytes": 3 * 1024 * 1024,
                "unity_total_reserved_bytes": 4 * 1024 * 1024,
            }))

            with patch.object(sys, "argv", [
                "format_unity_metrics_report.py",
                "--result", f"mipnerf360/bicycle={result}",
                "--method", "2dts",
                "--condition", "method-aware",
                "--profile-runs", "1",
                "--output-dir", str(report_dir),
            ]):
                self.assertEqual(format_metrics_report_main(), 0)

            self.assertTrue((report_dir / "unity_metrics_report.csv").is_file())
            self.assertTrue((report_dir / "unity_metrics_report.json").is_file())
            payload = json.loads((report_dir / "unity_metrics_report.json").read_text())
            self.assertEqual(payload["scenes"][0]["views"], 2)
            self.assertNotIn("fps_profile_cpu_p50", payload["scenes"][0])
            self.assertNotIn("capture_fps", payload["scenes"][0])
            self.assertAlmostEqual(payload["scenes"][0]["fps_profile_gpu_p50"], 1000.0 / 6.0)
            self.assertEqual(payload["scenes"][0]["profile_gpu_samples"], 2)
            self.assertAlmostEqual(payload["scenes"][0]["rendering_memory_mib"], 2.0)
            self.assertGreater(payload["scenes"][0]["asset_memory_mib"], 0.0)


if __name__ == "__main__":
    unittest.main()
