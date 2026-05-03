import json
import logging

from music_decoder.logging_setup import configure_logging, get_logger


def test_get_logger_returns_named_logger():
    log = get_logger("test_mod")
    assert log.name == "music_decoder.test_mod"


def test_configure_logging_writes_json(capsys):
    configure_logging(level="INFO")
    log = get_logger("emit")
    log.info("hello", extra={"job_id": 42, "stage": "transcription"})
    captured = capsys.readouterr()
    record = json.loads(captured.err.strip().splitlines()[-1])
    assert record["level"] == "INFO"
    assert record["message"] == "hello"
    assert record["logger"] == "music_decoder.emit"
    assert record["job_id"] == 42
    assert record["stage"] == "transcription"
    assert "timestamp" in record


def test_configure_logging_is_idempotent():
    configure_logging(level="INFO")
    configure_logging(level="DEBUG")
    root = logging.getLogger("music_decoder")
    handlers = root.handlers
    # Only one of our handlers should be attached.
    own = [h for h in handlers if getattr(h, "_md_marker", False)]
    assert len(own) == 1
