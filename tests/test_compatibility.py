from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.compatibility import recommend_runtime_profile


class CompatibilityTests(unittest.TestCase):
    def test_cpu_profile_without_gpu(self) -> None:
        recommendation = recommend_runtime_profile(
            os_name="Linux",
            architecture="x86_64",
            has_nvidia_gpu=False,
        )
        self.assertEqual(recommendation.profile.name, "cpu")

    def test_cuda_121_profile_with_modern_driver(self) -> None:
        recommendation = recommend_runtime_profile(
            os_name="Linux",
            architecture="x86_64",
            has_nvidia_gpu=True,
            driver_version="550.54.14",
            driver_cuda_version="12.4",
        )
        self.assertEqual(recommendation.profile.name, "cu121")

    def test_cpu_fallback_for_unknown_architecture(self) -> None:
        recommendation = recommend_runtime_profile(
            os_name="Linux",
            architecture="armv7l",
            has_nvidia_gpu=True,
            driver_version="550.54.14",
            driver_cuda_version="12.4",
        )
        self.assertEqual(recommendation.profile.name, "cpu")


if __name__ == "__main__":
    unittest.main()
