# Clipboard TTS

Speak the macOS clipboard out loud using the [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS model. Renders the text to audio, then plays it through `mpv` driven over its JSON IPC socket — giving you interactive pause, seek, and live speed controls when run in a terminal.

Clips are saved under `audios/` so you can list and replay them without re-synthesizing. Runs fully offline after the model is cached on first run.

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
- `-l, --list`: list saved clips and exit. See [Saved clips](#saved-clips).
- `-p, --play [N]`: replay a saved clip instead of reading the clipboard.
- `--force`: re-synthesize even when a matching clip is already saved.
- `--no-save`: play from a temp file; leave `audios/` untouched.
- `--keep N`: retain only the N most recently played clips. Default `50`.

### Playback controls

While playing **in a terminal**, these keys are live:

| Key       | Action                                  |
| --------- | --------------------------------------- |
| `space`   | pause / resume                          |
| `←` / `→` | seek backward / forward 5s              |
| `↑` / `↓` | playback speed −/+ 0.1x (pitch-corrected) |
| `q`       | quit                                    |

Controls require a focused terminal. When launched without a TTY (e.g. from a global shortcut), the script just plays the audio through with no interactive controls.

Pair it with a system shortcut (Raycast, Hammerspoon, Karabiner, Shortcuts.app, etc.) to make it one keypress away. See [Raycast setup](#raycast-setup).

## Raycast setup

Two script commands live in `~/.raycast-scripts/`:

- `speak-clipboard.sh` — reads the clipboard aloud. Forwards any flags through, so you can copy it per-voice (`-v af_bella`) or add `--no-save` to keep shortcut runs out of `audios/`.
- `stop-speaking.sh` — cuts playback short.

Register them once: **Raycast → Settings → Extensions → Script Commands → Add Directories**, then pick `~/.raycast-scripts`. The two commands appear in Raycast search as "Speak Clipboard" and "Stop Speaking"; assign hotkeys from the same settings pane (⌘K on a selected command → Configure Hotkey).

Raycast runs scripts with a bare environment — mise never activates and Homebrew isn't on `PATH` — so the wrapper hard-codes the venv interpreter, `PATH=/opt/homebrew/bin:...` (for `mpv` and `espeak-ng`), and `HF_HUB_OFFLINE=1`. If you move the repo, update `REPO` at the top of `speak-clipboard.sh`.

A Raycast launch has no TTY, so the transport keys are dead and there is no way to pause or stop from the keyboard — that's what "Stop Speaking" is for. It matches on mpv's IPC socket name, so it won't touch unrelated mpv windows.

## Saved clips

Every run saves its audio to `audios/` (gitignored) as `<timestamp>_<voice>_<hash>.wav`, alongside an `audios/index.json` catalog. List and replay them:

```sh
python speak_clipboard.py -l          # list saved clips
python speak_clipboard.py -p          # pick one interactively
python speak_clipboard.py -p 3        # replay clip 3 straight away
python speak_clipboard.py -p 3 -s 1.2 # replay clip 3 at 1.2x
```

```
  1. 2026-07-16T11-05-51  bf_emma       A third clip to trigger pruning.
  2. 2026-07-16T11-05-23  bf_emma       Testing the audio library and replay...
```

Numbering is by recency of play, so `1` is always the clip you last listened to. Replaying skips synthesis and the torch import entirely, so `-l` and `-p` are near-instant.

**Reuse.** The `<hash>` is derived from the text and voice together, so running on unchanged clipboard text just replays the saved clip instead of re-synthesizing. Pass `--force` to re-render (it replaces the file in place), or `--no-save` for a throwaway run that leaves `audios/` alone.

**Retention.** WAV at 24kHz mono runs about 2.8 MB per minute of speech, so the 50 most recently played clips are kept and older ones are pruned after each synthesis. Change the bound with `--keep N`, or `--keep 0` to keep everything. Deleting files from `audios/` by hand is safe — missing files drop out of the listing automatically.

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
- `DEFAULT_KEEP`: how many clips `audios/` retains without an explicit `--keep`.
- `AUDIO_DIR`: where clips are stored.
- `split_pattern=r"\n+"` in `synthesize()`: how the input is chunked for synthesis.

## How it works

1. `pbpaste` reads the clipboard.
2. The text and voice are hashed. On a hit in `audios/index.json`, synthesis is skipped and the saved WAV is played (step 5).
3. Otherwise `KPipeline` synthesizes the full text to a float32 PCM buffer (`24kHz`, mono) at 1.0x.
4. The buffer is written to `audios/<timestamp>_<voice>_<hash>.wav` and recorded in the index; clips beyond `--keep` are pruned.
5. `mpv` plays the WAV, launched with `--input-ipc-server` so the script can send commands over a Unix socket.
6. Keypresses in the terminal are translated to mpv IPC commands (`set pause`, `seek`, `set_property speed`). Playback speed is applied by mpv with pitch correction, so it can change live without re-rendering.
7. The socket is cleaned up when playback ends or you quit. The WAV stays for replay unless `--no-save` was passed.

## Refreshing the model

To pick up an upstream model update, just run with `-d` once:

```sh
python speak_clipboard.py -d
```

Or set the env var manually:

```sh
HF_HUB_OFFLINE=0 python speak_clipboard.py
```

The cache lives at `~/.cache/huggingface/hub/` and persists across reboots.
