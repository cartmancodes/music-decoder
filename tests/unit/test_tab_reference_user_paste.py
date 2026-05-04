from music_decoder.tab_reference.base import RawTabInput, TabReferenceProvider
from music_decoder.tab_reference.user_paste import UserPasteProvider


def test_user_paste_url_records_source():
    provider: TabReferenceProvider = UserPasteProvider()
    out = provider.fetch(RawTabInput(text="https://www.ultimate-guitar.com/tab/x"))
    assert out.source == "user_pasted_url"
    assert "ultimate-guitar.com" in out.raw_text


def test_user_paste_text_records_source():
    raw = "e|---0---0---|\nB|---1---1---|\n"
    provider: TabReferenceProvider = UserPasteProvider()
    out = provider.fetch(RawTabInput(text=raw))
    assert out.source == "user_pasted_text"
    assert out.raw_text == raw
