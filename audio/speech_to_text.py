"""
Speech to Text
Records a phrase from your microphone and transcribes it.

Uses Google Web Speech API via the SpeechRecognition library.
Requires an internet connection. No API key needed for basic use.

How it works:
  - Listens until you stop speaking (silence detection is automatic)
  - Sends the audio clip to Google's free endpoint
  - Prints the transcription

Run:
    python audio/speech_to_text.py
    Ctrl+C to quit
"""

import sys
import speech_recognition as sr


def transcribe_once(r: sr.Recognizer, mic: sr.Microphone) -> str | None:
    with mic as source:
        print("Adjusting for noise... ", end="", flush=True)
        r.adjust_for_ambient_noise(source, duration=0.8)
        print("ready.  Speak now (pausing stops the recording)...")
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            print("[INFO] No speech detected.")
            return None

    print("Transcribing...")
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("[WARN] Couldn't understand that. Try again.")
        return None
    except sr.RequestError as exc:
        print(f"[ERROR] Google API unavailable: {exc}")
        return None


def main() -> None:
    r   = sr.Recognizer()
    mic = sr.Microphone()

    print("Speech to Text  —  Ctrl+C to quit")
    print("Note: requires internet (Google Web Speech API)\n")

    while True:
        try:
            text = transcribe_once(r, mic)
            if text:
                print(f"\n  → {text}\n")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            sys.exit(0)


if __name__ == "__main__":
    main()
