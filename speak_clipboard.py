"""Speak the macOS clipboard out loud using the Kokoro TTS model.

Usage:
    python speak_clipboard.py [-s SPEED] [-v VOICE]

Options:
    -s, --speed SPEED   Playback speed multiplier (must be > 0). Default: 1.0
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
                        Note: the first time you use a new voice you must run
                        with HF_HUB_OFFLINE=0 so it can be downloaded.

Examples:
    python speak_clipboard.py                        # default voice, 1.0x
    python speak_clipboard.py --speed 1.2            # 1.2x speed
    python speak_clipboard.py -v af_bella            # American female 'Bella'
    python speak_clipboard.py -v bm_george -s 1.1    # British male, 1.1x

Exit codes:
    0  success
    1  clipboard empty, invalid args, or no audio generated
"""

import argparse
import subprocess
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.rnn")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.nn\.utils\.weight_norm")
    from kokoro import KPipeline


DEFAULT_VOICE = "bf_emma"
VALID_LANG_CODES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}


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


def stream_audio(text, speed=1.0, voice=DEFAULT_VOICE):
    lang_code = lang_code_for_voice(voice)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.rnn")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.nn\.utils\.weight_norm")
        pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")

    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r"\n+",
    )

    proc = subprocess.Popen(
        [
            "ffplay",
            "-loglevel", "quiet",
            "-nodisp",
            "-autoexit",
            "-f", "f32le",
            "-ar", "24000",
            "-ch_layout", "mono",
            "-",
        ],
        stdin=subprocess.PIPE,
    )

    produced = False
    try:
        for _, _, audio in generator:
            produced = True
            proc.stdin.write(audio.cpu().numpy().astype("float32").tobytes())
            proc.stdin.flush()
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if not produced:
        raise RuntimeError("No audio generated")


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
        help="Playback speed multiplier (e.g., 1.2 for 1.2x). Default: 1.0",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Kokoro voice ID (e.g., af_bella, bm_george). Default: {DEFAULT_VOICE}",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        print("Speed must be greater than 0", file=sys.stderr)
        sys.exit(1)

    try:
        lang_code_for_voice(args.voice)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    text = get_clipboard_text()

    if not text:
        print("Clipboard is empty", file=sys.stderr)
        sys.exit(1)

    stream_audio(text, speed=args.speed, voice=args.voice)


if __name__ == "__main__":
    main()
