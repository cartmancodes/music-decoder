from __future__ import annotations

from urllib.parse import urlparse

from .base import FetchedTab, RawTabInput


class UserPasteProvider:
    def fetch(self, payload: RawTabInput) -> FetchedTab:
        text = payload.text
        try:
            parsed = urlparse(text.strip())
            looks_like_url = bool(parsed.scheme and parsed.netloc)
        except Exception:
            looks_like_url = False
        return FetchedTab(
            source="user_pasted_url" if looks_like_url else "user_pasted_text",
            raw_text=text,
        )
