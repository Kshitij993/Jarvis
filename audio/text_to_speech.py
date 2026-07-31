"""
Text to Speech
Converts text you type into spoken audio using pyttsx3.
Fully offline — no internet or API key required.

Features:
  - Lists all voices installed on your system
  - Choose a voice, speech rate, and volume
  - Type anything and hear it spoken immediately

Run:
    python audio/text_to_speech.py
"""

import sys
import pyttsx3


def pick_voice(engine: pyttsx3.Engine) -> None:
    voices = engine.getProperty("voices")
    if not voices:
        print("[WARN] No voices found on this system.")
        return

    print("\nAvailable voices:\n")
    for i, v in enumerate(voices):
        lang = v.languages[0] if v.languages else "?"
        print(f"  [{i}]  {v.name}  ({lang})")

    choice = input("\nPick a voice number (Enter = default): ").strip()
    if choice.isdigit() and int(choice) < len(voices):
        engine.setProperty("voice", voices[int(choice)].id)


def main() -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate",   160)   # words per minute
    engine.setProperty("volume", 1.0)   # 0.0 – 1.0

    print("Text to Speech\n" + "=" * 40)

    pick_voice(engine)

    rate_in = input("Speech rate in words/min (Enter = 160): ").strip()
    if rate_in.isdigit():
        engine.setProperty("rate", int(rate_in))

    print("\nType text and press Enter to speak it.")
    print("Empty line to quit.\n")

    while True:
        try:
            text = input("> ").strip()
            if not text:
                print("Goodbye.")
                break
            engine.say(text)
            engine.runAndWait()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

    engine.stop()


if __name__ == "__main__":
    main()
