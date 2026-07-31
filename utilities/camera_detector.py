"""
Camera Detector Utility
Lists all available camera devices on the system and shows their capabilities.

Usage (standalone):
    python camera_detector.py

Usage (as a module):
    from utilities.camera_detector import list_cameras, get_camera_info
"""

import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraInfo:
    index: int
    backend: str
    width: int
    height: int
    fps: float
    is_available: bool


def get_camera_info(index: int) -> Optional[CameraInfo]:
    """
    Try to open a camera at the given index and return its properties.
    Returns None if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW is faster on Windows
    if not cap.isOpened():
        # Fall back to default backend
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None

    info = CameraInfo(
        index=index,
        backend=cap.getBackendName(),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        is_available=True,
    )
    cap.release()
    return info


def list_cameras(max_test: int = 10) -> list[CameraInfo]:
    """
    Probe camera indices 0 .. max_test-1 and return info for all available ones.

    Args:
        max_test: How many indices to probe (default 10).

    Returns:
        List of CameraInfo for every camera that could be opened.
    """
    available = []
    for i in range(max_test):
        info = get_camera_info(i)
        if info is not None:
            available.append(info)
    return available


def print_camera_report(cameras: list[CameraInfo]):
    """Pretty-print a camera report to the console."""
    if not cameras:
        print("[INFO] No cameras detected on this system.")
        return

    print(f"\n{'='*55}")
    print(f"  {len(cameras)} camera(s) detected")
    print(f"{'='*55}")
    for cam in cameras:
        print(f"  Index   : {cam.index}")
        print(f"  Backend : {cam.backend}")
        print(f"  Resolution: {cam.width} x {cam.height}")
        print(f"  FPS     : {cam.fps:.1f}")
        print(f"  {'─'*45}")


if __name__ == "__main__":
    print("[INFO] Probing cameras, please wait...")
    cams = list_cameras()
    print_camera_report(cams)
