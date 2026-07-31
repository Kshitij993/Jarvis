"""
Audio Input Selector
Helps you find and confirm the right microphone on your system.

Lists every audio input device, lets you pick one, records a short
test clip, and plays it back so you can confirm it's picking up sound.

Run:
    python audio/audio_input_selector.py
"""

import sys
import wave
import pyaudio

CHUNK      = 1024
FORMAT     = pyaudio.paInt16
CHANNELS   = 1
RATE       = 44100
RECORD_SEC = 3


def list_input_devices(pa: pyaudio.PyAudio) -> list[dict]:
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append(info)
    return devices


def record(pa: pyaudio.PyAudio, device_index: int, seconds: int) -> bytes:
    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
    print(f"\n  Recording {seconds}s — speak now!")
    frames = []
    total = int(RATE / CHUNK * seconds)
    for i in range(total):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
        pct = (i + 1) / total
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        print(f"\r  [{bar}] {int(pct * 100):3d}%", end="", flush=True)
    print()
    stream.stop_stream()
    stream.close()
    return b"".join(frames)


def play(pa: pyaudio.PyAudio, audio_bytes: bytes) -> None:
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    print("  Playing back...")
    for i in range(0, len(audio_bytes), CHUNK * 2):
        stream.write(audio_bytes[i : i + CHUNK * 2])
    stream.stop_stream()
    stream.close()


def main() -> None:
    pa = pyaudio.PyAudio()
    print("\nAudio Input Selector\n" + "=" * 40)

    devices = list_input_devices(pa)
    if not devices:
        print("[ERROR] No input devices found.")
        pa.terminate()
        sys.exit(1)

    default_idx = pa.get_default_input_device_info()["index"]
    print("\nAvailable microphones:\n")
    for i, dev in enumerate(devices):
        tag = "  ← default" if dev["index"] == default_idx else ""
        print(f"  [{i}]  {dev['name']}{tag}")
        print(f"        {int(dev['defaultSampleRate'])} Hz  |  {dev['maxInputChannels']} ch\n")

    choice = input("Pick a number to test (Enter = default): ").strip()
    if choice == "":
        dev_index = default_idx
        dev_name  = pa.get_device_info_by_index(default_idx)["name"]
    else:
        sel       = devices[int(choice)]
        dev_index = sel["index"]
        dev_name  = sel["name"]

    print(f"\nUsing: {dev_name}  (index {dev_index})")

    audio = record(pa, dev_index, RECORD_SEC)
    play(pa, audio)

    if input("\nSave to test.wav? [y/N]: ").strip().lower() == "y":
        with wave.open("test.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio)
        print("  Saved → test.wav")

    pa.terminate()
    print(f"\nUse device index {dev_index} in other audio scripts.")


if __name__ == "__main__":
    main()
