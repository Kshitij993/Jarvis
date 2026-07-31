"""
Record and Play
Records audio from the default microphone and plays it back.

Usage:
    python audio/record_and_play.py                   # 5 second clip
    python audio/record_and_play.py --duration 10     # 10 seconds
    python audio/record_and_play.py --output my.wav   # custom file name
"""

import argparse
import wave
import pyaudio

CHUNK    = 1024
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 44100


def record(pa: pyaudio.PyAudio, seconds: int, path: str) -> None:
    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK,
    )
    total  = int(RATE / CHUNK * seconds)
    frames = []
    print(f"Recording {seconds}s — speak now!")
    for i in range(total):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
        pct = (i + 1) / total
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r  [{bar}] {int(pct * 100):3d}%", end="", flush=True)
    print()
    stream.stop_stream()
    stream.close()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    print(f"Saved → {path}")


def play(pa: pyaudio.PyAudio, path: str) -> None:
    print(f"Playing back {path} ...")
    with wave.open(path, "rb") as wf:
        stream = pa.open(
            format=pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)
        stream.stop_stream()
        stream.close()
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record and play audio.")
    ap.add_argument("--duration", type=int, default=5,  help="Seconds to record (default 5)")
    ap.add_argument("--output",   type=str, default="recording.wav", help="Output file")
    args = ap.parse_args()

    pa = pyaudio.PyAudio()
    try:
        record(pa, args.duration, args.output)
        play(pa, args.output)
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
