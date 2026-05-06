
from __future__ import annotations

from contextbench.extractors import treesitter


def test_get_parser_for_lang_uses_alias_for_c_sharp(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("contextbench.extractors.treesitter.available", lambda: True)
    monkeypatch.setattr("contextbench.extractors.treesitter._PARSERS", {})

    def fake_get_parser(lang: str):
        calls.append(lang)
        if lang == "csharp":
            return object()
        raise LookupError(lang)

    monkeypatch.setattr("contextbench.extractors.treesitter._get_parser", fake_get_parser)

    parser = treesitter._get_parser_for_lang("c_sharp")

    assert parser is not None
    assert calls == ["c_sharp", "csharp"]
