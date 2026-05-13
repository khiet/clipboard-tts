# Clipboard TTS

Speak the macOS clipboard out loud using the [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS model. Streams audio directly to `ffplay` so playback starts as soon as the first chunk is synthesized. No temp file, no waiting for the whole text to render.

Runs fully offline after the model is cached on first run.

## Prerequisites

System packages (Homebrew):

```sh
brew install espeak-ng ffmpeg
```

- `espeak-ng`: phonemizer backend Kokoro uses for English text.
- `ffmpeg`: provides `ffplay`, used to stream raw PCM audio to the speakers.

Python toolchain via [mise](https://mise.jdx.dev/) (auto-creates a `.venv`):

```sh
mise install
```

## Install Python dependencies

```sh
pip install kokoro
```

That pulls in `torch`, `numpy`, `soundfile`, `huggingface_hub`, and their transitive dependencies.

## First run

The first invocation downloads the Kokoro model (a few hundred MB) into `~/.cache/huggingface/`. Make sure you're online for this:

```sh
HF_HUB_OFFLINE=0 python speak_clipboard.py
```

After that, `mise.toml` sets `HF_HUB_OFFLINE=1` so subsequent runs skip the network entirely.

## Usage

Copy any text, then:

```sh
python speak_clipboard.py
```

Pair it with a system shortcut (Raycast, Hammerspoon, Karabiner, Shortcuts.app, etc.) to make it one keypress away.

## Configuration

Edit `speak_clipboard.py` to change:

- `lang_code="b"`: Kokoro language (`a` = American English, `b` = British English, etc.)
- `voice="bf_emma"`: voice ID; see the [Kokoro voices list](https://huggingface.co/hexgrad/Kokoro-82M)
- `speed=1.0`: playback speed multiplier
- `split_pattern=r"\n+"`: how the input is chunked for streaming synthesis

## How it works

1. `pbpaste` reads the clipboard.
2. `KPipeline` synthesizes audio in chunks (split on newlines).
3. Each chunk is converted to raw float32 PCM (`24kHz`, mono) and piped to `ffplay` via stdin.
4. `ffplay` plays the stream live and exits when stdin closes (`-autoexit`).

## Refreshing the model

To pick up an upstream model update:

```sh
HF_HUB_OFFLINE=0 python speak_clipboard.py
```

The cache lives at `~/.cache/huggingface/hub/` and persists across reboots.
