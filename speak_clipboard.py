import subprocess
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.rnn")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.nn\.utils\.weight_norm")
    from kokoro import KPipeline


def get_clipboard_text():
    result = subprocess.run(
        ["pbpaste"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def stream_audio(text):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.rnn")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.nn\.utils\.weight_norm")
        pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")

    generator = pipeline(
        text,
        voice="bf_emma",
        speed=1.0,
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
    text = get_clipboard_text()

    if not text:
        print("Clipboard is empty", file=sys.stderr)
        sys.exit(1)

    stream_audio(text)


if __name__ == "__main__":
    main()
