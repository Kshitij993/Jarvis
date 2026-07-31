"""
Microphone Detector Utility  (Enhanced)
Enumerates audio input devices with smart filtering and categorisation:
  - Tests each device before listing it (skips broken/ghost entries)
  - Labels devices as: microphone | system_audio | virtual | unknown
  - Detects WASAPI loopback devices (system audio capture) when pyaudiowpatch
    is installed; falls back to keyword-matching for "Stereo Mix" etc.

Usage (standalone):
    python utilities/microphone_detector.py

Usage (as a module):
    from utilities.microphone_detector import list_microphones, list_system_audio_devices
"""

from dataclasses import dataclass
from typing import Optional

# pyaudiowpatch adds WASAPI loopback support on Windows.
# Fall back to plain pyaudio if it is not installed.
try:
    import pyaudiowpatch as _pa_lib
    _WPATCH = True
except ImportError:
    import pyaudio as _pa_lib          # type: ignore[no-redef]
    _WPATCH = False

import pyaudio                          # always needed for the paInt16 constant


# ── Keyword classifiers ───────────────────────────────────────────────────────

_MIC_HINTS = {
    "mic", "microphone", "headset", "headphone mic", "realtek mic",
    "conexant", "line in", "capture", "input", "recording",
}
_SYS_HINTS = {
    "stereo mix", "what u hear", "what you hear", "wave out mix",
    "loopback", "monitor of", "speakers (loopback)", "output (loopback)",
}
_VIRTUAL_HINTS = {
    "virtual", "vb-audio", "vb audio", "cable output", "cable input",
    "voicemeeter", "voice meeter", "obs virtual", "soundflower",
    "blackhole", "dante virtual", "asio",
}


@dataclass
class MicrophoneInfo:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    host_api_name: str
    category: str       # 'microphone' | 'system_audio' | 'virtual' | 'unknown'
    is_loopback: bool   # True for WASAPI loopback / system-audio devices


def _categorise(name: str, is_loopback: bool) -> str:
    if is_loopback:
        return "system_audio"
    n = name.lower()
    if any(h in n for h in _VIRTUAL_HINTS):
        return "virtual"
    if any(h in n for h in _SYS_HINTS):
        return "system_audio"
    if any(h in n for h in _MIC_HINTS):
        return "microphone"
    return "unknown"


def _test_open(pa, index: int, channels: int, rate: float) -> bool:
    """Return True only if the device can actually be opened for recording."""
    try:
        s = pa.open(
            format=pyaudio.paInt16,
            channels=min(channels, 2),
            rate=int(rate),
            input=True,
            input_device_index=index,
            frames_per_buffer=512,
        )
        s.close()
        return True
    except Exception:
        return False


def list_microphones(
    test_devices: bool = True,
    include_virtual: bool = False,
    include_system_audio: bool = False,
) -> list[MicrophoneInfo]:
    """
    Return verified audio input devices.

    Parameters
    ----------
    test_devices
        Try to open each device; skip devices that fail (default True).
        Eliminates ghost devices and broken entries.
    include_virtual
        Include virtual audio cables (VB-Audio, etc.).  Default False.
    include_system_audio
        Include loopback / "what you hear" devices.  Default False.
        Use list_system_audio_devices() to get only those.
    """
    pa = _pa_lib.PyAudio()
    results: list[MicrophoneInfo] = []

    try:
        host_names: dict[int, str] = {
            h: pa.get_host_api_info_by_index(h).get("name", f"API {h}")
            for h in range(pa.get_host_api_count())
        }

        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if int(dev.get("maxInputChannels", 0)) <= 0:
                continue

            name      = dev.get("name", f"Device {i}")
            host_api  = host_names.get(int(dev.get("hostApi", -1)), "Unknown")
            channels  = int(dev.get("maxInputChannels", 1))
            rate      = float(dev.get("defaultSampleRate", 44100))
            is_loop   = bool(dev.get("isLoopbackDevice", False))  # pyaudiowpatch only
            category  = _categorise(name, is_loop)

            # Apply category filters
            if category == "virtual"      and not include_virtual:
                continue
            if category == "system_audio" and not include_system_audio:
                continue

            # Skip devices that cannot actually be opened
            if test_devices and not _test_open(pa, i, channels, rate):
                continue

            results.append(MicrophoneInfo(
                index=i, name=name,
                max_input_channels=channels,
                default_sample_rate=rate,
                host_api_name=host_api,
                category=category,
                is_loopback=is_loop,
            ))
    finally:
        pa.terminate()

    # Sort: verified mics first, unknown second, system audio last
    _order = {"microphone": 0, "unknown": 1, "system_audio": 2, "virtual": 3}
    results.sort(key=lambda m: _order.get(m.category, 1))
    return results


def list_system_audio_devices() -> list[MicrophoneInfo]:
    """
    Return loopback / system-audio devices suitable for capturing speaker output.

    On Windows with pyaudiowpatch installed:  returns proper WASAPI loopback devices.
    Without pyaudiowpatch:                    returns devices whose names match
                                              "Stereo Mix", "What U Hear", etc.
    On Linux/Mac:                             PulseAudio monitor sources are already
                                              listed as normal input devices — enable
                                              them and they appear here automatically.
    """
    pa = _pa_lib.PyAudio()
    results: list[MicrophoneInfo] = []

    try:
        host_names: dict[int, str] = {
            h: pa.get_host_api_info_by_index(h).get("name", f"API {h}")
            for h in range(pa.get_host_api_count())
        }

        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if int(dev.get("maxInputChannels", 0)) <= 0:
                continue

            name     = dev.get("name", f"Device {i}")
            host_api = host_names.get(int(dev.get("hostApi", -1)), "Unknown")
            channels = int(dev.get("maxInputChannels", 1))
            rate     = float(dev.get("defaultSampleRate", 44100))
            is_loop  = bool(dev.get("isLoopbackDevice", False))
            category = _categorise(name, is_loop)

            if category != "system_audio":
                continue
            if not _test_open(pa, i, channels, rate):
                continue

            results.append(MicrophoneInfo(
                index=i, name=name,
                max_input_channels=channels,
                default_sample_rate=rate,
                host_api_name=host_api,
                category=category,
                is_loopback=is_loop,
            ))
    finally:
        pa.terminate()

    return results


def get_default_microphone() -> Optional[MicrophoneInfo]:
    """Return the first verified microphone, or None."""
    mics = list_microphones(test_devices=True)
    return mics[0] if mics else None


def get_best_system_audio_device() -> Optional[MicrophoneInfo]:
    """
    Return the loopback device for the **current default audio output**.

    When headphones / a headset are connected, Windows switches the default
    output to that device.  A naive list_system_audio_devices()[0] would
    return the old speaker loopback (now silent).  This function instead
    queries the WASAPI default output device and returns its matching
    loopback — so it always follows wherever the sound is actually going.

    Requires pyaudiowpatch for reliable results.  Falls back to the first
    available system-audio device if pyaudiowpatch is not installed or if
    matching fails.
    """
    if _WPATCH:
        try:
            pa = _pa_lib.PyAudio()
            try:
                wasapi    = pa.get_host_api_info_by_type(_pa_lib.paWASAPI)
                out_idx   = wasapi["defaultOutputDevice"]
                out_name  = pa.get_device_info_by_index(out_idx).get("name", "")
                host_names = {
                    h: pa.get_host_api_info_by_index(h).get("name", f"API {h}")
                    for h in range(pa.get_host_api_count())
                }
                for i in range(pa.get_device_count()):
                    dev = pa.get_device_info_by_index(i)
                    if (bool(dev.get("isLoopbackDevice", False))
                            and int(dev.get("maxInputChannels", 0)) > 0
                            and out_name in dev.get("name", "")):
                        m = MicrophoneInfo(
                            index=i,
                            name=dev.get("name", f"Device {i}"),
                            max_input_channels=int(dev["maxInputChannels"]),
                            default_sample_rate=float(dev["defaultSampleRate"]),
                            host_api_name=host_names.get(
                                int(dev.get("hostApi", -1)), "WASAPI"),
                            category="system_audio",
                            is_loopback=True,
                        )
                        if _test_open(pa, i,
                                      m.max_input_channels,
                                      m.default_sample_rate):
                            return m
            finally:
                pa.terminate()
        except Exception:
            pass

    # Fall back: first available system-audio device (keyword-matched)
    devs = list_system_audio_devices()
    return devs[0] if devs else None


def print_microphone_report(mics: list[MicrophoneInfo]) -> None:
    if not mics:
        print("[INFO] No working audio-input devices found.")
        return
    print(f"\n{'='*55}")
    print(f"  {len(mics)} device(s) detected")
    print(f"{'='*55}")
    for m in mics:
        tag = f"[{m.category}{'  loopback' if m.is_loopback else ''}]"
        print(f"  [{m.index}]  {m.name}  {tag}")
        print(f"       {m.host_api_name}  |  "
              f"{m.max_input_channels} ch  |  {m.default_sample_rate:.0f} Hz")
        print()


if __name__ == "__main__":
    print("\n-- Microphones --")
    print_microphone_report(list_microphones())
    print("\n-- System Audio (loopback) --")
    sys_devs = list_system_audio_devices()
    if sys_devs:
        print_microphone_report(sys_devs)
    else:
        print("  None found.\n"
              "  Windows: enable 'Stereo Mix' in Sound settings, or\n"
              "           pip install pyaudiowpatch  for WASAPI loopback.")
