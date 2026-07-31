"""
System Info Utility
Collects general system and hardware information that is useful when setting
up or debugging other scripts in this project.

Usage (standalone):
    python system_info.py

Usage (as a module):
    from utilities.system_info import get_system_info, SystemInfo
"""

import platform
import sys
import os
import psutil
import cv2
from dataclasses import dataclass, field


@dataclass
class SystemInfo:
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    opencv_version: str
    python_executable: str
    env_packages: list[str] = field(default_factory=list)


def get_system_info() -> SystemInfo:
    """Gather and return system information as a SystemInfo dataclass."""
    uname = platform.uname()
    mem   = psutil.virtual_memory()

    # Collect installed packages visible to this Python interpreter
    try:
        import importlib.metadata as meta
        packages = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in meta.distributions()
        )
    except Exception:
        packages = []

    return SystemInfo(
        os_name=uname.system,
        os_version=uname.version,
        architecture=uname.machine,
        python_version=sys.version,
        cpu_name=uname.processor or platform.processor(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 0,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=round(mem.total / (1024 ** 3), 2),
        ram_available_gb=round(mem.available / (1024 ** 3), 2),
        opencv_version=cv2.__version__,
        python_executable=sys.executable,
        env_packages=packages,
    )


def print_system_report(info: SystemInfo, show_packages: bool = False):
    """Pretty-print the system info report."""
    print(f"\n{'='*55}")
    print("  System Information")
    print(f"{'='*55}")
    print(f"  OS              : {info.os_name} {info.os_version}")
    print(f"  Architecture    : {info.architecture}")
    print(f"  CPU             : {info.cpu_name}")
    print(f"  CPU Cores       : {info.cpu_cores_physical} physical / {info.cpu_cores_logical} logical")
    print(f"  RAM             : {info.ram_total_gb} GB total / {info.ram_available_gb} GB available")
    print(f"  Python          : {info.python_version}")
    print(f"  Python Path     : {info.python_executable}")
    print(f"  OpenCV Version  : {info.opencv_version}")

    if show_packages and info.env_packages:
        print(f"\n  {'─'*45}")
        print("  Installed packages in this environment:")
        for pkg in info.env_packages:
            print(f"    {pkg}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display system information.")
    parser.add_argument("--packages", action="store_true",
                        help="Also list installed Python packages.")
    args = parser.parse_args()

    info = get_system_info()
    print_system_report(info, show_packages=args.packages)
