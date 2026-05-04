import importlib


def test_package_imports():
    mod = importlib.import_module("music_decoder")
    assert hasattr(mod, "__version__")


def test_version_is_string():
    import music_decoder

    assert isinstance(music_decoder.__version__, str)
    assert len(music_decoder.__version__) > 0
