# audio/

Scripts for working with microphones, recording, and speech. They build on each other — start with `audio_input_selector.py` to confirm your mic is working, then move to the others.

---

### `audio_input_selector.py`
Lists every audio input device on your system, records a short test clip from whichever one you pick, and plays it back. Use this first to figure out which device index to use in the other scripts.

```bash
python audio/audio_input_selector.py
```

---

### `record_and_play.py`
Records from the default mic and immediately plays back what it captured. Accepts `--duration` and `--output` flags.

```bash
python audio/record_and_play.py
python audio/record_and_play.py --duration 10 --output my_clip.wav
```

---

### `speech_to_text.py`
Records a phrase (stops automatically when you pause) and transcribes it via the Google Web Speech API. Keeps looping so you can transcribe multiple phrases one after another.

> Requires an internet connection. No API key needed.

```bash
python audio/speech_to_text.py
```

---

### `text_to_speech.py`
Reads back whatever text you type using `pyttsx3` — fully offline, no internet or API key. Lists the voices installed on your system and lets you pick one and adjust the speech rate.

```bash
python audio/text_to_speech.py
```

---

### `live_speech_to_text_ui.py`
A tkinter window that listens continuously and appends each transcribed phrase to a scrollable transcript with timestamps. Has a microphone selector, Start/Stop button, Copy to Clipboard, and Clear.

> Requires an internet connection (Google Web Speech API).

```bash
python audio/live_speech_to_text_ui.py
```

---

### `live_speech_to_text_offline_ui.py`
Same idea as the online version but runs entirely offline using **Vosk**. The big difference is it shows **live partial results** — words appear on screen as you speak, before you've finished the sentence.

On first run click **Download Model** — it fetches the small English model (~50 MB) once and saves it to `audio/models/`. All future runs load it from disk instantly.

Settings work the same as the online version: the **Pause to finalise** slider controls how long silence must last before the phrase is committed (raise it to avoid mid-sentence cuts). The script uses both Vosk's internal VAD and its own RMS-based silence detector so the slider gives real control.

```bash
python audio/live_speech_to_text_offline_ui.py
```

---

**Dependencies:** `pyaudio` (recording), `SpeechRecognition` (online STT), `pyttsx3` (TTS), `vosk` (offline STT) — all installed by the root `install` scripts.
