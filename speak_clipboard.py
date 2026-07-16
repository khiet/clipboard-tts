"""Speak the macOS clipboard out loud using the Kokoro TTS model.

Audio is synthesized in full, saved under ``audios/``, then played through
``mpv`` driven over its JSON IPC socket. That gives interactive playback
controls when run in a terminal:

    [space]  pause / resume
    [<-/->]  seek backward / forward 5s
    [up/dn]  playback speed -/+ 0.1x (pitch-corrected)
    [q]      quit

Saved clips are content-addressed on (text, voice), so re-running on unchanged
clipboard text replays the existing file instead of re-synthesizing.

Usage:
    python speak_clipboard.py [-s SPEED] [-v VOICE] [-d] [--force] [--no-save]
    python speak_clipboard.py -l
    python speak_clipboard.py -p [N]

Options:
    -s, --speed SPEED   Initial playback speed multiplier (must be > 0).
                        Default: 1.0. Adjustable live with up/down arrows.
                        Examples: 1.2 = 1.2x faster, 0.8 = 0.8x slower.
    -v, --voice VOICE   Kokoro voice ID. Default: bf_emma
                        Format: <lang><gender>_<name> where:
                          lang:   a=American, b=British, e=Spanish, f=French,
                                  h=Hindi, i=Italian, j=Japanese,
                                  p=Brazilian Portuguese, z=Mandarin
                          gender: f=female, m=male
                        Examples: af_heart, af_bella, am_adam, bf_isabella,
                                  bm_george, jf_alpha
                        Full list: https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
    -d, --download      Allow Hugging Face downloads for this run by unsetting
                        HF_HUB_OFFLINE. Use this the first time you try a new
                        voice. Cached voices then work offline.
    -l, --list          List saved clips, most recently played first, and exit.
    -p, --play [N]      Replay saved clip N (as numbered by --list) instead of
                        reading the clipboard. With no N, pick interactively.
        --force         Re-synthesize even if a matching clip is saved.
        --no-save       Play from a temp file; don't add to audios/.
        --keep N        Retain only the N most recently played clips, pruning
                        least-recently-played first. 0 keeps everything.
                        Default: 50.

Examples:
    python speak_clipboard.py                        # default voice, 1.0x
    python speak_clipboard.py --speed 1.2            # start at 1.2x speed
    python speak_clipboard.py -v af_bella            # American female 'Bella'
    python speak_clipboard.py -v bm_george -s 1.1    # British male, 1.1x
    python speak_clipboard.py -v af_bella -d         # download new voice
    python speak_clipboard.py -l                     # list saved clips
    python speak_clipboard.py -p                     # pick a clip to replay
    python speak_clipboard.py -p 3 -s 1.2            # replay clip 3 at 1.2x

Interactive controls need a focused terminal. When launched without a TTY
(e.g. from a global shortcut), the script just plays the audio through.

Exit codes:
    0  success
    1  clipboard empty, invalid args, missing voice (offline), no audio,
       or no/invalid clip selected for --play
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import termios
import time
import tty
import warnings
from pathlib import Path

import soundfile as sf
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled


DEFAULT_VOICE = "bf_emma"
VALID_LANG_CODES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}

SAMPLE_RATE = 24000
SEEK_SECONDS = 5
SPEED_STEP = 0.1
MIN_SPEED = 0.1

AUDIO_DIR = Path(__file__).resolve().parent / "audios"
INDEX_PATH = AUDIO_DIR / "index.json"
DEFAULT_KEEP = 50
SNIPPET_CHARS = 60


def digest_for(text, voice):
    """Content address for a clip. Same text and voice always maps here."""
    return hashlib.sha256(f"{voice}\0{text}".encode()).hexdigest()[:8]


def snippet_for(text):
    collapsed = " ".join(text.split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[: SNIPPET_CHARS - 3] + "..."


def load_index():
    """Saved clips, least recently played first.

    Entries whose audio file is gone are dropped, so deleting from audios/ by
    hand stays consistent with the index.
    """
    try:
        entries = json.loads(INDEX_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [e for e in entries if (AUDIO_DIR / e["file"]).exists()]


def save_index(entries):
    AUDIO_DIR.mkdir(exist_ok=True)
    INDEX_PATH.write_text(json.dumps(entries, indent=2) + "\n")


def touch(entries, entry):
    """Move an entry to the most-recently-played end, preserving LRU order."""
    remaining = [e for e in entries if e["digest"] != entry["digest"]]
    return [*remaining, entry]


def store_audio(audio, text, voice, digest, entries):
    """Write a new clip to audios/ and return (path, updated entries)."""
    AUDIO_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = f"{stamp}_{voice}_{digest}.wav"
    path = AUDIO_DIR / name
    sf.write(path, audio, SAMPLE_RATE)
    entry = {
        "file": name,
        "voice": voice,
        "created": stamp,
        "digest": digest,
        "snippet": snippet_for(text),
    }
    return path, [*entries, entry]


def prune(entries, keep):
    """Drop all but the `keep` most recently played clips. Returns survivors."""
    if keep <= 0 or len(entries) <= keep:
        return entries
    for entry in entries[: len(entries) - keep]:
        with contextlib.suppress(FileNotFoundError):
            (AUDIO_DIR / entry["file"]).unlink()
    return entries[len(entries) - keep :]


def format_rows(entries):
    """Numbered rows, most recently played first. Index 1 is the newest."""
    return [
        (i, e, f"{i:>3}. {e['created']}  {e['voice']:<12}  {e['snippet']}")
        for i, e in enumerate(reversed(entries), start=1)
    ]


def print_list(entries):
    if not entries:
        print("No saved clips yet.")
        return
    for _, _, row in format_rows(entries):
        print(row)


def entry_at(entries, number):
    """Look up an entry by its --list number, or None if out of range."""
    for i, entry, _ in format_rows(entries):
        if i == number:
            return entry
    return None


def pick_entry(entries):
    """Prompt for a clip number. Returns the entry, or None if cancelled."""
    print_list(entries)
    if not entries:
        return None
    try:
        answer = input(f"Play which? [1-{len(entries)}, q to cancel]: ").strip()
    except EOFError:
        return None
    if answer.lower() in {"q", ""}:
        return None
    if not answer.isdigit():
        print(f"Not a number: {answer}", file=sys.stderr)
        return None
    entry = entry_at(entries, int(answer))
    if entry is None:
        print(f"No clip numbered {answer}", file=sys.stderr)
    return entry


def get_clipboard_text():
    result = subprocess.run(
        ["pbpaste"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def lang_code_for_voice(voice):
    """Derive Kokoro lang_code from the voice ID's first character."""
    if not voice or voice[0] not in VALID_LANG_CODES:
        raise ValueError(
            f"Invalid voice '{voice}'. Expected format <lang><gender>_<name>, "
            f"where lang is one of: {sorted(VALID_LANG_CODES)}"
        )
    return voice[0]


def synthesize(text, voice):
    """Render the whole text to a single float32 PCM buffer at SAMPLE_RATE.

    Synthesis is always at 1.0x; playback speed is applied later by mpv so it
    can be adjusted live without re-rendering.

    Kokoro (and through it, torch) is imported here rather than at module scope:
    it costs over a second, and the --list, --play, and cache-reuse paths never
    need it.
    """
    lang_code = lang_code_for_voice(voice)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"torch\.nn\.modules\.rnn"
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"torch\.nn\.utils\.weight_norm"
        )
        from kokoro import KPipeline

        import numpy as np

        pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")

    generator = pipeline(
        text,
        voice=voice,
        speed=1.0,
        split_pattern=r"\n+",
    )

    chunks = [audio.cpu().numpy().astype("float32") for _, _, audio in generator]
    if not chunks:
        raise RuntimeError("No audio generated")

    return np.concatenate(chunks)


def _ipc_send(sock, *command):
    """Send one mpv IPC command, best-effort. Responses are ignored.

    If mpv has already exited (e.g. playback finished, or a `quit` raced with
    EOF) the socket is gone; the command is simply a no-op.
    """
    with contextlib.suppress(OSError):
        sock.sendall(json.dumps({"command": list(command)}).encode() + b"\n")


def _connect_ipc(sock_path, proc, timeout=5.0):
    """Connect to mpv's IPC socket once it appears, or None if mpv exits first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(0.05)
    return None


def _read_key(fd):
    """Read one logical keypress; map arrow escape sequences to names."""
    ch = os.read(fd, 1)
    if ch != b"\x1b":
        return ch.decode("utf-8", "replace")
    # Arrow keys arrive as ESC [ A/B/C/D. A bare ESC has no follow-up bytes.
    if not select.select([fd], [], [], 0.05)[0]:
        return "\x1b"
    seq = os.read(fd, 2)
    return {b"[A": "UP", b"[B": "DOWN", b"[C": "RIGHT", b"[D": "LEFT"}.get(seq, "")


def _render(paused, speed):
    state = "Paused " if paused else "Playing"
    print(f"\r{state}   speed {speed:.1f}x   ", end="", flush=True)


def _control_loop(sock, proc, speed):
    """Translate keypresses to mpv IPC commands until playback ends or quits."""
    if not sys.stdin.isatty():
        proc.wait()
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    paused = False
    print(
        "Controls:  [space] pause/resume   [←/→] seek 5s   [↑/↓] speed ±0.1   [q] quit"
    )
    try:
        tty.setcbreak(fd)
        _render(paused, speed)
        while proc.poll() is None:
            if not select.select([fd], [], [], 0.2)[0]:
                continue
            key = _read_key(fd)
            if key == "q":
                _ipc_send(sock, "quit")
                break
            elif key == " ":
                paused = not paused
                _ipc_send(sock, "set", "pause", "yes" if paused else "no")
                _render(paused, speed)
            elif key == "RIGHT":
                _ipc_send(sock, "seek", SEEK_SECONDS, "relative")
            elif key == "LEFT":
                _ipc_send(sock, "seek", -SEEK_SECONDS, "relative")
            elif key == "UP":
                speed = round(speed + SPEED_STEP, 2)
                _ipc_send(sock, "set_property", "speed", speed)
                _render(paused, speed)
            elif key == "DOWN":
                speed = round(max(MIN_SPEED, speed - SPEED_STEP), 2)
                _ipc_send(sock, "set_property", "speed", speed)
                _render(paused, speed)
    except KeyboardInterrupt:
        with contextlib.suppress(OSError):
            _ipc_send(sock, "quit")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        print()


def play_with_controls(wav_path, initial_speed):
    """Play a WAV file through mpv with interactive transport controls."""
    sock_path = os.path.join(tempfile.gettempdir(), f"mpv-clipboard-{os.getpid()}.sock")
    proc = None
    try:
        proc = subprocess.Popen(
            [
                "mpv",
                "--no-video",
                "--no-terminal",
                "--no-config",
                "--audio-pitch-correction=yes",
                f"--speed={initial_speed}",
                f"--input-ipc-server={sock_path}",
                str(wav_path),
            ]
        )
        sock = _connect_ipc(sock_path, proc)
        if sock is None:
            proc.wait()
            return
        try:
            _control_loop(sock, proc, initial_speed)
        finally:
            sock.close()
        proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)


def main():
    parser = argparse.ArgumentParser(
        description="Speak clipboard text via Kokoro TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for the voice ID format and the full voice list URL.",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="Initial playback speed multiplier (e.g., 1.2 for 1.2x). Default: 1.0",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Kokoro voice ID (e.g., af_bella, bm_george). Default: {DEFAULT_VOICE}",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Allow HF downloads this run (unsets HF_HUB_OFFLINE). Needed for new voices.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List saved clips (most recently played first) and exit.",
    )
    parser.add_argument(
        "-p",
        "--play",
        nargs="?",
        type=int,
        const=0,
        metavar="N",
        help="Replay saved clip N instead of the clipboard. No N: pick interactively.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-synthesize even if a clip for this text and voice is saved.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Play from a temp file; don't add to audios/.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP,
        metavar="N",
        help=f"Retain only the N most recently played clips (0 = unlimited). Default: {DEFAULT_KEEP}",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        print("Speed must be greater than 0", file=sys.stderr)
        sys.exit(1)

    entries = load_index()

    if args.list:
        print_list(entries)
        return

    if args.play is not None:
        replay(entries, args.play, args.speed)
        return

    try:
        lang_code_for_voice(args.voice)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.download:
        os.environ.pop("HF_HUB_OFFLINE", None)

    text = get_clipboard_text()

    if not text:
        print("Clipboard is empty", file=sys.stderr)
        sys.exit(1)

    speak(text, entries, args)


def replay(entries, number, speed):
    """Play a saved clip by --list number, or interactively when number is 0."""
    if number:
        entry = entry_at(entries, number)
        if entry is None:
            print(
                f"No clip numbered {number}. Use --list to see saved clips.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif not sys.stdin.isatty():
        print(
            "--play needs a clip number when there's no terminal to pick from.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        entry = pick_entry(entries)
        if entry is None:
            sys.exit(1)

    save_index(touch(entries, entry))
    play_with_controls(AUDIO_DIR / entry["file"], speed)


def speak(text, entries, args):
    """Synthesize (or reuse) a clip for the clipboard text and play it."""
    digest = digest_for(text, args.voice)
    cached = next((e for e in entries if e["digest"] == digest), None)

    if cached is not None and not args.force and not args.no_save:
        print(f"Reusing saved clip {cached['file']} (--force to re-synthesize)")
        save_index(touch(entries, cached))
        play_with_controls(AUDIO_DIR / cached["file"], args.speed)
        return

    try:
        audio = synthesize(text, args.voice)
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as exc:
        print(
            f"\nError: voice '{args.voice}' is not cached and HF is offline.\n"
            f"Re-run with -d/--download to fetch it, e.g.:\n"
            f"    python speak_clipboard.py -d -v {args.voice}\n"
            f"\nUnderlying error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.no_save:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(wav_path, audio, SAMPLE_RATE)
            play_with_controls(wav_path, args.speed)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(wav_path)
        return

    # A re-synthesized clip replaces the stale file at the same digest.
    remaining = [e for e in entries if e["digest"] != digest]
    if cached is not None:
        with contextlib.suppress(FileNotFoundError):
            (AUDIO_DIR / cached["file"]).unlink()

    path, updated = store_audio(audio, text, args.voice, digest, remaining)
    save_index(prune(updated, args.keep))
    play_with_controls(path, args.speed)


if __name__ == "__main__":
    main()
