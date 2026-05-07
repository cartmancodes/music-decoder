import importlib


def test_streamlit_app_imports():
    mod = importlib.import_module("music_decoder.ui.streamlit_app")
    # must define the page render function we'll call from streamlit run.
    assert hasattr(mod, "render")
