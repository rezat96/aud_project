import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import av
from faster_whisper import WhisperModel

from agent import DecisionError
from scheduler import utc_now
from session_manager import SessionManager

_original_av_open = av.open


def _compat_av_open(*args, **kwargs):
    """PyAV 19+ removed the metadata_errors argument expected by faster-whisper."""
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)


av.open = _compat_av_open


def transcribe_audio_file(
    audio_path: Path,
    *,
    model: str = "small",
    language: str | None = None,
    cpu_threads: int = 8,
) -> tuple[str, object, list[object]]:
    load_started = monotonic()
    model_handle = WhisperModel(
        model,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    load_seconds = monotonic() - load_started

    transcription_started = monotonic()
    segments, info = model_handle.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,
        vad_filter=True,
    )
    transcript_segments = list(segments)
    transcription_seconds = monotonic() - transcription_started
    transcript_text = " ".join(segment.text.strip() for segment in transcript_segments if segment.text.strip())
    return transcript_text.strip(), {"language": info.language,
                                     "confidence": info.language_probability,
                                     "model_load_seconds": load_seconds,
                                     "transcription_seconds": transcription_seconds}, transcript_segments


def _run_scheduler_agent(transcript: str, session_id: str | None = None) -> dict:
    if os.environ.get("DEMO_ALLOW_PAID_CALLS") != "1":
        raise DecisionError("Paid calls disabled. Set DEMO_ALLOW_PAID_CALLS=1 and OPENAI_API_KEY before routing speech to the scheduler agent.")
    manager = SessionManager()
    return manager.process_turn(transcript, session_id=session_id, now=utc_now())


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file locally with Whisper.")
    parser.add_argument("audio", type=Path, help="Input audio file, such as WAV, MP3, or M4A")
    parser.add_argument("--model", default="small", choices=("base", "small"))
    parser.add_argument("--language", choices=("en", "fr"), help="Auto-detect when omitted")
    parser.add_argument("--agent", action="store_true", help="Send the transcript to the scheduler agent after transcription")
    parser.add_argument("--session-id", default="default", help="Keep one live session alive across repeated turns in the same process")
    parser.add_argument("--interactive", action="store_true", help="After the first transcript, keep the same session open for follow-up text prompts")
    arguments = parser.parse_args()

    if not arguments.audio.is_file():
        parser.error(f"Audio file not found: {arguments.audio}")

    transcript, metrics, transcript_segments = transcribe_audio_file(
        arguments.audio,
        model=arguments.model,
        language=arguments.language,
    )

    print(f"Language: {metrics['language']} ({metrics['confidence']:.0%} confidence)")
    for segment in transcript_segments:
        print(f"[{segment.start:0.2f}-{segment.end:0.2f}] {segment.text.strip()}")
    print(f"Model load: {metrics['model_load_seconds']:.2f}s; transcription: {metrics['transcription_seconds']:.2f}s")
    print(f"Transcript: {transcript}")

    if arguments.agent:
        print("\nRouting transcript to the scheduler agent...")
        try:
            result = _run_scheduler_agent(transcript, session_id=arguments.session_id)
        except DecisionError as error:
            print(f"Scheduler agent unavailable: {error}")
            return 1
        print(f"Agent event: {result['event']}")
        print(result["reply"])
        payload = {key: value for key, value in result.items() if key != "reply"}
        if payload:
            print(json.dumps(payload, default=str))

        if arguments.interactive:
            print("\nSession active. Type follow-up text; enter quit to exit.")
            while True:
                try:
                    follow_up = input("You: ")
                except (EOFError, KeyboardInterrupt):
                    break
                if follow_up.strip().casefold() in {"quit", "exit"}:
                    break
                try:
                    follow_result = _run_scheduler_agent(follow_up, session_id=arguments.session_id)
                except DecisionError as error:
                    print(f"Scheduler agent unavailable: {error}")
                    break
                print(f"Agent event: {follow_result['event']}")
                print(follow_result["reply"])
                payload = {key: value for key, value in follow_result.items() if key != "reply"}
                if payload:
                    print(json.dumps(payload, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
