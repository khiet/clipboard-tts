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

After that, `mise.toml` sets `HF_HUB_OFFLINE=1` so subsequent runs skip the network entirely. When you later want to try a new voice, the script's `-d/--download` flag (see below) handles the offline override for you — no need to remember the env var.

## Usage

Copy any text, then:

```sh
python speak_clipboard.py                      # default voice (bf_emma), 1.0x
python speak_clipboard.py --speed 1.2          # 1.2x playback speed
python speak_clipboard.py -v af_bella          # American female 'Bella'
python speak_clipboard.py -v bm_george -s 1.1  # British male, 1.1x
python speak_clipboard.py -d -v jf_alpha       # download & try a new voice
```

Flags:

- `-s, --speed FLOAT`: playback speed multiplier (must be > 0). Default `1.0`.
- `-v, --voice ID`: Kokoro voice ID. Default `bf_emma`. See [Voices](#voices).
- `-d, --download`: allow Hugging Face downloads for this run by unsetting `HF_HUB_OFFLINE`. Use this the first time you try a new voice; cached voices then work offline.

Pair it with a system shortcut (Raycast, Hammerspoon, Karabiner, Shortcuts.app, etc.) to make it one keypress away. The companion `~/speak_clipboard.sh` wrapper forwards all flags through to the Python script.

## Voices

Voice IDs follow the format `<lang><gender>_<name>`:

- `lang`: `a` American, `b` British, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese, `p` Brazilian Portuguese, `z` Mandarin
- `gender`: `f` female, `m` male

Examples: `af_heart`, `af_bella`, `am_adam`, `bf_emma`, `bf_isabella`, `bm_george`, `jf_alpha`.

The script auto-derives Kokoro's `lang_code` from the voice's first character, so cross-language voices work without extra config.

Find available voices:

- Authoritative list: <https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices>
- Locally cached only: `ls ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/voices/`

The first time you use a voice that isn't cached, run with `-d` to fetch it. If you forget, the script catches the offline-mode error and prints a hint pointing at the right command.

## Configuration

Most knobs are CLI flags now (see [Usage](#usage)). To change defaults or chunking behaviour, edit `speak_clipboard.py`:

- `DEFAULT_VOICE`: voice used when `--voice` isn't passed.
- `split_pattern=r"\n+"` in `stream_audio()`: how the input is chunked for streaming synthesis.

## How it works

1. `pbpaste` reads the clipboard.
2. `KPipeline` synthesizes audio in chunks (split on newlines).
3. Each chunk is converted to raw float32 PCM (`24kHz`, mono) and piped to `ffplay` via stdin.
4. `ffplay` plays the stream live and exits when stdin closes (`-autoexit`).

## Refreshing the model

To pick up an upstream model update, just run with `-d` once:

```sh
./speak_clipboard.sh -d
```

Or set the env var manually:

```sh
HF_HUB_OFFLINE=0 python speak_clipboard.py
```

The cache lives at `~/.cache/huggingface/hub/` and persists across reboots.
