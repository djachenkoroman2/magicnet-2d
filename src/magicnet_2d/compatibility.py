from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    python_version: str
    torch_version: str
    torchvision_version: str
    mmengine_version: str
    mmcv_package: str
    mmcv_version: str
    mmdet_version: str
    tensorboard_version: str
    cuda_version: str | None = None
    cudnn_version: str | None = None
    pytorch_index_url: str | None = None


@dataclass(frozen=True)
class RuntimeRecommendation:
    profile: RuntimeProfile
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


CPU_PROFILE = RuntimeProfile(
    name="cpu",
    python_version="3.10",
    torch_version="2.1.0",
    torchvision_version="0.16.0",
    mmengine_version="0.10.5",
    mmcv_package="mmcv",
    mmcv_version="2.1.0",
    mmdet_version="3.3.0",
    tensorboard_version="2.16.2",
)

CUDA_11_8_PROFILE = RuntimeProfile(
    name="cu118",
    python_version="3.10",
    torch_version="2.1.0",
    torchvision_version="0.16.0",
    mmengine_version="0.10.5",
    mmcv_package="mmcv",
    mmcv_version="2.1.0",
    mmdet_version="3.3.0",
    tensorboard_version="2.16.2",
    cuda_version="11.8",
    cudnn_version="8.9+",
    pytorch_index_url="https://download.pytorch.org/whl/cu118",
)

CUDA_12_1_PROFILE = RuntimeProfile(
    name="cu121",
    python_version="3.10",
    torch_version="2.1.0",
    torchvision_version="0.16.0",
    mmengine_version="0.10.5",
    mmcv_package="mmcv",
    mmcv_version="2.1.0",
    mmdet_version="3.3.0",
    tensorboard_version="2.16.2",
    cuda_version="12.1",
    cudnn_version="8.9+",
    pytorch_index_url="https://download.pytorch.org/whl/cu121",
)


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    tokens = re.findall(r"\d+", value)
    if not tokens:
        return ()
    return tuple(int(token) for token in tokens[:3])


def _version_at_least(current: str | None, minimum: tuple[int, ...]) -> bool:
    current_tuple = _version_tuple(current)
    if not current_tuple:
        return False
    padded_current = current_tuple + (0,) * (len(minimum) - len(current_tuple))
    return padded_current >= minimum


def recommend_runtime_profile(
    *,
    os_name: str,
    architecture: str,
    has_nvidia_gpu: bool,
    driver_version: str | None = None,
    driver_cuda_version: str | None = None,
    nvcc_version: str | None = None,
) -> RuntimeRecommendation:
    reasons: list[str] = []
    warnings: list[str] = []

    if architecture not in {"x86_64", "amd64"}:
        warnings.append(
            "Non-standard architecture detected. Falling back to the CPU profile because GPU wheel coverage is less predictable."
        )
        return RuntimeRecommendation(CPU_PROFILE, tuple(reasons), tuple(warnings))

    if os_name not in {"Linux", "Windows"}:
        warnings.append(
            "This template is primarily tuned for Linux and Windows. Falling back to the CPU profile."
        )
        return RuntimeRecommendation(CPU_PROFILE, tuple(reasons), tuple(warnings))

    if not has_nvidia_gpu:
        reasons.append("No NVIDIA GPU detected, so the CPU profile is the safest option.")
        return RuntimeRecommendation(CPU_PROFILE, tuple(reasons), tuple(warnings))

    visible_cuda = driver_cuda_version or nvcc_version

    if _version_at_least(driver_version, (530, 30)) and (
        _version_at_least(visible_cuda, (12, 1)) or not visible_cuda
    ):
        reasons.append("Driver and CUDA capability are compatible with the CUDA 12.1 profile.")
        return RuntimeRecommendation(CUDA_12_1_PROFILE, tuple(reasons), tuple(warnings))

    if _version_at_least(driver_version, (520, 61)) and (
        _version_at_least(visible_cuda, (11, 8)) or not visible_cuda
    ):
        reasons.append("Driver and CUDA capability are compatible with the CUDA 11.8 profile.")
        return RuntimeRecommendation(CUDA_11_8_PROFILE, tuple(reasons), tuple(warnings))

    warnings.append(
        "NVIDIA GPU detected, but the available driver/CUDA information does not cleanly match the maintained GPU profiles."
    )
    warnings.append("Falling back to the CPU profile until the driver or toolkit is updated.")
    return RuntimeRecommendation(CPU_PROFILE, tuple(reasons), tuple(warnings))
