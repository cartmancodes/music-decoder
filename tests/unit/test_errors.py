from music_decoder.errors import (
    ChordRecognitionError,
    ClipTooShortError,
    CompositionError,
    CorruptAudioError,
    IngestError,
    InvalidScaleError,
    MusicDecoderError,
    SilentAudioError,
    YouTubeError,
)


def test_youtube_error_is_ingest_error():
    assert issubclass(YouTubeError, IngestError)
    assert issubclass(IngestError, MusicDecoderError)


def test_invalid_scale_is_composition_error():
    assert issubclass(InvalidScaleError, CompositionError)
    assert issubclass(CompositionError, MusicDecoderError)


def test_corrupt_silent_short_are_ingest_errors():
    for cls in (CorruptAudioError, SilentAudioError, ClipTooShortError):
        assert issubclass(cls, IngestError)


def test_each_error_carries_message():
    err = ChordRecognitionError("boom")
    assert str(err) == "boom"
    assert isinstance(err, MusicDecoderError)
