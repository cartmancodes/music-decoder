from music_decoder.progress import NullProgress, ProgressCallback


def test_null_progress_callable_does_nothing():
    p = NullProgress()
    p("loading", 0.0)
    p("loading", 0.5)
    p("done", 1.0)


def test_progress_callback_protocol_accepts_lambda():
    calls = []
    cb: ProgressCallback = lambda stage, fraction: calls.append((stage, fraction))
    cb("a", 0.5)
    assert calls == [("a", 0.5)]
