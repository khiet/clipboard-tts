# Clipboard TTS

Speak the macOS clipboard out loud using the [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS model. Renders the text to audio, then plays it through `mpv` driven over its JSON IPC socket — giving you interactive pause, seek, and live speed controls when run in a terminal.

Runs fully offline after the model is cached on first run.

## Prerequisites

System packages (Homebrew):

```sh
brew install espeak-ng mpv
```

- `espeak-ng`: phonemizer backend Kokoro uses for English text.
- `mpv`: media player used for playback and transport controls; driven over its JSON IPC socket.

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

- `-s, --speed FLOAT`: initial playback speed multiplier (must be > 0). Default `1.0`. Adjustable live with the up/down arrows.
- `-v, --voice ID`: Kokoro voice ID. Default `bf_emma`. See [Voices](#voices).
- `-d, --download`: allow Hugging Face downloads for this run by unsetting `HF_HUB_OFFLINE`. Use this the first time you try a new voice; cached voices then work offline.

### Playback controls

While playing **in a terminal**, these keys are live:

| Key       | Action                                  |
| --------- | --------------------------------------- |
| `space`   | pause / resume                          |
| `←` / `→` | seek backward / forward 5s              |
| `↑` / `↓` | playback speed −/+ 0.1x (pitch-corrected) |
| `q`       | quit                                    |

Controls require a focused terminal. When launched without a TTY (e.g. from a global shortcut), the script just plays the audio through with no interactive controls.

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
2. `KPipeline` synthesizes the full text to a float32 PCM buffer (`24kHz`, mono) at 1.0x.
3. The buffer is written to a temp WAV and played by `mpv`, launched with `--input-ipc-server` so the script can send commands over a Unix socket.
4. Keypresses in the terminal are translated to mpv IPC commands (`set pause`, `seek`, `set_property speed`). Playback speed is applied by mpv with pitch correction, so it can change live without re-rendering.
5. The temp WAV and socket are cleaned up when playback ends or you quit.

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
