from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(frozen=True)
class DeviceChoice:
    torch_device: str
    label: str
    weight: float
    info: dict[str, Any]


def collect_host_info() -> dict[str, Any]:
    devices = [_cpu_info()]
    devices.extend(_gpu_info())
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "devices": devices,
        "selected": _select_device_info("auto", devices).info,
    }


def resolve_torch_device(requested: str, torch: Any) -> DeviceChoice:
    text = str(requested or "auto").strip().lower()
    if text == "metal":
        text = "mps"

    devices = collect_host_info().get("devices", [])
    if text in {"", "auto"}:
        if torch.cuda.is_available():
            cuda_choice = _select_device_info("cuda", devices)
            return DeviceChoice("cuda", cuda_choice.label, cuda_choice.weight, cuda_choice.info)
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            mps_choice = _select_device_info("mps", devices)
            return DeviceChoice("mps", mps_choice.label, mps_choice.weight, mps_choice.info)
        cpu_choice = _select_device_info("cpu", devices)
        return DeviceChoice("cpu", cpu_choice.label, cpu_choice.weight, cpu_choice.info)

    choice = _select_device_info(text, devices)
    return DeviceChoice(text, choice.label, choice.weight, choice.info)


def host_weight(host_info: dict[str, Any] | None, requested_device: str = "auto") -> float:
    if not isinstance(host_info, dict):
        return 1.0
    devices = host_info.get("devices", [])
    if not isinstance(devices, list):
        return 1.0
    return _select_device_info(requested_device, devices).weight


def _select_device_info(requested: str, devices: list[Any]) -> DeviceChoice:
    normalized = str(requested or "auto").strip().lower()
    usable = [device for device in devices if isinstance(device, dict)]

    if normalized.startswith("cuda"):
        cuda_devices = [
            device
            for device in usable
            if str(device.get("device", "")).lower() == "cuda"
            or str(device.get("kind", "")).lower() == "gpu"
            and "nvidia" in str(device.get("vendor", "")).lower()
        ]
        return _best(cuda_devices or usable, fallback_label=normalized or "cuda")

    if normalized in {"mps", "metal"}:
        metal_devices = [
            device
            for device in usable
            if str(device.get("device", "")).lower() in {"mps", "metal"}
            or "apple" in str(device.get("vendor", "")).lower()
        ]
        return _best(metal_devices or usable, fallback_label="mps")

    if normalized == "cpu":
        cpu_devices = [device for device in usable if str(device.get("kind", "")).lower() == "cpu"]
        return _best(cpu_devices or usable, fallback_label="cpu")

    gpu_devices = [device for device in usable if str(device.get("kind", "")).lower() == "gpu"]
    if gpu_devices:
        return _best(gpu_devices, fallback_label="gpu")
    return _best(usable, fallback_label="cpu")


def _best(devices: list[dict[str, Any]], *, fallback_label: str) -> DeviceChoice:
    if not devices:
        return DeviceChoice(fallback_label, fallback_label, 1.0, {})
    scored = sorted(devices, key=_device_weight, reverse=True)
    device = scored[0]
    label = str(device.get("device") or device.get("kind") or fallback_label).lower()
    return DeviceChoice(label, label, _device_weight(device), dict(device))


def _device_weight(device: dict[str, Any]) -> float:
    kind = str(device.get("kind", "")).lower()
    if kind == "gpu":
        memory = _float(
            device.get("available_vram_gb")
            or device.get("vram_gb")
            or device.get("available_ram_gb")
            or device.get("ram_gb"),
            0.0,
        )
        flops = _float(device.get("flops"), 0.0)
        compute_boost = max(flops, 1.0) ** 0.25 if flops > 0 else 1.0
        return max(memory, 1.0) * compute_boost

    memory = _float(device.get("available_ram_gb") or device.get("ram_gb"), 0.0)
    cores = _float(device.get("cores"), 1.0)
    return max(memory, 1.0) * max(cores, 1.0) ** 0.1 * 0.25


def _cpu_info() -> dict[str, Any]:
    return {
        "kind": "CPU",
        "device": "CPU",
        "name": _cpu_name(),
        "cores": _cpu_count(),
        "ram_gb": _ram_gb(),
        "available_ram_gb": _available_ram_gb(),
    }


def _gpu_info() -> list[dict[str, Any]]:
    system = platform.system()
    devices: list[dict[str, Any]] = []
    if system in {"Linux", "Windows"}:
        devices.extend(_nvidia_gpus())
        if not devices:
            devices.extend(_rocm_gpus())
    if system == "Darwin":
        devices.extend(_apple_gpus())
    return devices


def _nvidia_gpus() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=2, stderr=subprocess.DEVNULL)
    except Exception:
        return []

    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        total = _mib_to_gib(parts[1])
        free = _mib_to_gib(parts[2]) if len(parts) > 2 else total
        devices.append(
            {
                "kind": "GPU",
                "device": "CUDA",
                "vendor": "NVIDIA",
                "name": parts[0],
                "vram_gb": total,
                "available_vram_gb": free,
                "ram_gb": total,
                "available_ram_gb": free,
            }
        )
    return devices


def _rocm_gpus() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    devices: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw in output.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line or line.startswith(("#", "=")):
            continue
        if lower.startswith("gpu"):
            if current:
                devices.append(_finish_rocm_gpu(current))
            current = {}
        elif "product name" in lower and ":" in line:
            current["name"] = line.split(":", 1)[1].strip()
        elif "vram total memory" in lower and ":" in line:
            number = _first_number(line.split(":", 1)[1])
            current["vram_gb"] = round(number / (1024**3), 2) if number > 1024**2 else number
    if current:
        devices.append(_finish_rocm_gpu(current))
    return devices


def _finish_rocm_gpu(gpu: dict[str, Any]) -> dict[str, Any]:
    total = _float(gpu.get("vram_gb"), 0.0)
    return {
        "kind": "GPU",
        "device": "ROCM",
        "vendor": "AMD",
        "name": gpu.get("name", "AMD GPU"),
        "vram_gb": total,
        "available_vram_gb": total,
        "ram_gb": total,
        "available_ram_gb": total,
    }


def _apple_gpus() -> list[dict[str, Any]]:
    name = "Apple GPU"
    try:
        output = subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        for raw in output.splitlines():
            line = raw.strip()
            if "chipset model" in line.lower() and ":" in line:
                name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    total = _ram_gb()
    available = _available_ram_gb()
    return [
        {
            "kind": "GPU",
            "device": "METAL",
            "vendor": "Apple",
            "name": name,
            "unified_memory": True,
            "vram_gb": total,
            "available_vram_gb": available,
            "ram_gb": total,
            "available_ram_gb": available,
        }
    ]


def _cpu_count() -> int:
    if psutil is not None:
        try:
            return int(psutil.cpu_count(logical=True) or 0)
        except Exception:
            pass
    return int(os.cpu_count() or 0)


def _ram_gb() -> float:
    if psutil is not None:
        try:
            return round(float(psutil.virtual_memory().total) / (1024**3), 2)
        except Exception:
            pass
    if hasattr(os, "sysconf") and sys.platform != "win32":
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(float(pages * page_size) / (1024**3), 2)
        except Exception:
            pass
    return 0.0


def _available_ram_gb() -> float:
    if psutil is not None:
        try:
            return round(float(psutil.virtual_memory().available) / (1024**3), 2)
        except Exception:
            pass
    return 0.0


def _cpu_name() -> str:
    system = platform.system()
    commands = []
    if system == "Darwin":
        commands.append(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name") and ":" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        commands.append(["lscpu"])
    elif system == "Windows":
        commands.append(["wmic", "cpu", "get", "Name"])

    for cmd in commands:
        try:
            output = subprocess.check_output(cmd, text=True, timeout=2, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for line in output.splitlines():
            clean = line.strip()
            if clean and clean.lower() not in {"name"} and not clean.lower().startswith("model name:"):
                return clean
            if clean.lower().startswith("model name:"):
                return clean.split(":", 1)[1].strip()
    return platform.processor() or "CPU"


def _mib_to_gib(value: Any) -> float:
    number = _first_number(value)
    if number > 64:
        return round(number / 1024.0, 2)
    return round(number, 2)


def _first_number(value: Any) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else 0.0


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
