from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Sequence

from .compatibility import (
    CPU_PROFILE,
    CUDA_11_8_PROFILE,
    CUDA_12_1_PROFILE,
    RuntimeProfile,
    RuntimeRecommendation,
    recommend_runtime_profile,
)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    os_name: str
    os_release: str
    architecture: str
    python_version: str
    has_nvidia_gpu: bool
    gpu_name: str | None
    nvidia_driver_version: str | None
    driver_cuda_version: str | None
    nvcc_version: str | None
    cudnn_version: str | None


@dataclass(frozen=True)
class InstallStep:
    description: str
    command: tuple[str, ...] | None
    manual: bool = False


def _run_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _detect_nvidia_driver() -> tuple[bool, str | None, str | None, str | None]:
    query = _run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not query:
        return False, None, None, None

    first_line = query.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    gpu_name = parts[0] if parts else None
    driver_version = parts[1] if len(parts) > 1 else None

    full_output = _run_command(["nvidia-smi"])
    driver_cuda_version = None
    if full_output:
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", full_output)
        if match:
            driver_cuda_version = match.group(1)

    return True, gpu_name, driver_version, driver_cuda_version


def _detect_nvcc_version() -> str | None:
    output = _run_command(["nvcc", "--version"])
    if not output:
        return None
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return match.group(1) if match else None


def _detect_cudnn_version() -> str | None:
    header_candidates = [
        Path("/usr/include/cudnn_version.h"),
        Path("/usr/include/x86_64-linux-gnu/cudnn_version.h"),
        Path("/usr/local/cuda/include/cudnn_version.h"),
    ]
    for header in header_candidates:
        if not header.exists():
            continue
        content = header.read_text(encoding="utf-8", errors="ignore")
        major = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", content)
        minor = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", content)
        patch = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", content)
        if major and minor and patch:
            return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    return None


def detect_environment() -> EnvironmentSnapshot:
    has_gpu, gpu_name, driver_version, driver_cuda_version = _detect_nvidia_driver()
    return EnvironmentSnapshot(
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine().lower(),
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        has_nvidia_gpu=has_gpu,
        gpu_name=gpu_name,
        nvidia_driver_version=driver_version,
        driver_cuda_version=driver_cuda_version,
        nvcc_version=_detect_nvcc_version(),
        cudnn_version=_detect_cudnn_version(),
    )


def select_profile(snapshot: EnvironmentSnapshot, requested_profile: str) -> RuntimeRecommendation:
    if requested_profile == "auto":
        return recommend_runtime_profile(
            os_name=snapshot.os_name,
            architecture=snapshot.architecture,
            has_nvidia_gpu=snapshot.has_nvidia_gpu,
            driver_version=snapshot.nvidia_driver_version,
            driver_cuda_version=snapshot.driver_cuda_version,
            nvcc_version=snapshot.nvcc_version,
        )

    forced_profiles = {
        "cpu": CPU_PROFILE,
        "cu118": CUDA_11_8_PROFILE,
        "cu121": CUDA_12_1_PROFILE,
    }
    profile = forced_profiles[requested_profile]
    return RuntimeRecommendation(
        profile=profile,
        reasons=(f"Profile was forced explicitly: {requested_profile}.",),
        warnings=(),
    )


def build_install_plan(snapshot: EnvironmentSnapshot, recommendation: RuntimeRecommendation) -> list[InstallStep]:
    profile = recommendation.profile
    steps = [
        InstallStep(
            description=f"Install the recommended Python version {profile.python_version} through uv.",
            command=("uv", "python", "install", profile.python_version),
        )
    ]

    if profile.name == "cpu":
        steps.append(
            InstallStep(
                description="Sync the project with the recommended CPU-compatible Python dependencies and dev tools.",
                command=("uv", "sync", "--extra", "cpu", "--extra", "dev"),
            )
        )
        steps.append(
            InstallStep(
                description=f"Install the compiled {profile.mmcv_package} wheel with OpenMIM.",
                command=("uv", "run", "mim", "install", f"mmcv=={profile.mmcv_version}"),
            )
        )
        return steps

    steps.append(
        InstallStep(
            description="Sync the project base environment and development tooling.",
            command=("uv", "sync", "--extra", "dev"),
        )
    )
    steps.append(
        InstallStep(
            description=f"Install PyTorch and TorchVision wheels for {profile.name}.",
            command=(
                "uv",
                "pip",
                "install",
                "--index-url",
                profile.pytorch_index_url or "",
                f"torch=={profile.torch_version}",
                f"torchvision=={profile.torchvision_version}",
            ),
        )
    )
    steps.append(
        InstallStep(
            description="Install the MMEngine, MMDetection and OpenMIM Python packages.",
            command=(
                "uv",
                "pip",
                "install",
                f"mmengine>={profile.mmengine_version},<1.0.0",
                f"mmdet=={profile.mmdet_version}",
                "openmim>=0.3.9",
            ),
        )
    )
    steps.append(
        InstallStep(
            description=f"Install the compiled {profile.mmcv_package} wheel with OpenMIM.",
            command=("uv", "run", "mim", "install", f"mmcv=={profile.mmcv_version}"),
        )
    )

    if not snapshot.has_nvidia_gpu:
        steps.append(
            InstallStep(
                description="No NVIDIA GPU was detected. Driver installation is a manual prerequisite for GPU profiles.",
                command=None,
                manual=True,
            )
        )
    if snapshot.cudnn_version is None:
        steps.append(
            InstallStep(
                description=(
                    "cuDNN was not detected. PyTorch wheels already bundle the runtime for most cases, "
                    "but system-wide cuDNN may still be needed for custom CUDA builds."
                ),
                command=None,
                manual=True,
            )
        )

    return steps


def _format_snapshot(snapshot: EnvironmentSnapshot) -> str:
    lines = [
        "Environment snapshot:",
        f"  OS: {snapshot.os_name} {snapshot.os_release}",
        f"  Architecture: {snapshot.architecture}",
        f"  Python: {snapshot.python_version}",
        f"  NVIDIA GPU: {'yes' if snapshot.has_nvidia_gpu else 'no'}",
        f"  GPU name: {snapshot.gpu_name or 'not detected'}",
        f"  NVIDIA driver: {snapshot.nvidia_driver_version or 'not detected'}",
        f"  CUDA from driver: {snapshot.driver_cuda_version or 'not detected'}",
        f"  CUDA toolkit (nvcc): {snapshot.nvcc_version or 'not detected'}",
        f"  cuDNN: {snapshot.cudnn_version or 'not detected'}",
    ]
    return "\n".join(lines)


def _format_recommendation(recommendation: RuntimeRecommendation) -> str:
    profile = recommendation.profile
    lines = [
        "Recommended runtime profile:",
        f"  Name: {profile.name}",
        f"  Python: {profile.python_version}",
        f"  PyTorch: {profile.torch_version}",
        f"  TorchVision: {profile.torchvision_version}",
        f"  MMEngine: {profile.mmengine_version}",
        f"  {profile.mmcv_package}: {profile.mmcv_version}",
        f"  MMDetection: {profile.mmdet_version}",
        f"  TensorBoard: {profile.tensorboard_version}",
    ]
    if profile.cuda_version:
        lines.append(f"  CUDA: {profile.cuda_version}")
    if profile.cudnn_version:
        lines.append(f"  cuDNN: {profile.cudnn_version}")
    if recommendation.reasons:
        lines.append("Reasons:")
        lines.extend(f"  - {reason}" for reason in recommendation.reasons)
    if recommendation.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in recommendation.warnings)
    return "\n".join(lines)


def _format_plan(steps: list[InstallStep]) -> str:
    lines = ["Suggested installation plan:"]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. {step.description}")
        if step.command:
            lines.append(f"   {' '.join(step.command)}")
        else:
            lines.append("   manual step")
    return "\n".join(lines)


def _execute_plan(steps: list[InstallStep]) -> int:
    for step in steps:
        if step.command is None:
            continue
        print(f"Running: {' '.join(step.command)}")
        completed = subprocess.run(step.command, check=False)
        if completed.returncode != 0:
            print(f"Command failed with exit code {completed.returncode}: {' '.join(step.command)}")
            return completed.returncode
    return 0


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "mode",
        nargs="?",
        default="check",
        choices=("check", "install"),
        help="`check` prints the detected environment and installation plan. `install` can execute the safe steps.",
    )
    parser.add_argument(
        "--profile",
        default="auto",
        choices=("auto", "cpu", "cu118", "cu121"),
        help="Override the automatically selected compatibility profile.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the non-manual steps of the generated installation plan.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    snapshot = detect_environment()
    recommendation = select_profile(snapshot, args.profile)
    plan = build_install_plan(snapshot, recommendation)

    print(_format_snapshot(snapshot))
    print()
    print(_format_recommendation(recommendation))
    print()
    print(_format_plan(plan))

    if args.mode == "install" and args.execute:
        print()
        return _execute_plan(plan)

    if args.mode == "install" and not args.execute:
        print()
        print("Dry-run only. Re-run with `--execute`, if you want to perform the safe install steps.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet-check-env",
        description="Inspect the local machine and build a safe MMDetection installation plan.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    return run_from_args(args)
