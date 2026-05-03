def test_warm_models_calls_each_loader(monkeypatch):
    calls: list[str] = []

    def fake_basic_pitch(): calls.append("basic-pitch")
    def fake_crepe(): calls.append("crepe")
    def fake_demucs(): calls.append("demucs")

    from music_decoder.cli import models_download
    monkeypatch.setattr(models_download, "_load_basic_pitch", fake_basic_pitch)
    monkeypatch.setattr(models_download, "_load_crepe", fake_crepe)
    monkeypatch.setattr(models_download, "_load_demucs", fake_demucs)

    models_download.warm_models()
    assert set(calls) == {"basic-pitch", "crepe", "demucs"}
