
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

import pytest


requires_tree_sitter = pytest.mark.skipif(
    not treesitter.available(), reason="tree-sitter parsers not installed"
)


def _named_defs(tmp_path, filename: str, source: str):
    path = tmp_path / filename
    path.write_text(source)
    return treesitter.extract_named_defs(str(path))


@requires_tree_sitter
def test_typescript_arrow_declarator_is_a_definition(tmp_path) -> None:
    defs = _named_defs(
        tmp_path,
        "util.ts",
        "export const toDisplayString = (val: unknown): string => String(val);\n",
    )

    assert [(kind, name) for kind, name, _, _ in defs] == [("variable_declarator", "toDisplayString")]


@requires_tree_sitter
def test_javascript_named_function_expression_is_a_definition(tmp_path) -> None:
    defs = _named_defs(
        tmp_path,
        "Select.js",
        "const Select = React.forwardRef(function Select(props, ref) { return null; });\n",
    )

    names = {(kind, name) for kind, name, _, _ in defs}
    assert ("function_expression", "Select") in names
    assert ("variable_declarator", "Select") in names


@requires_tree_sitter
def test_javascript_top_level_factory_call_is_a_definition(tmp_path) -> None:
    defs = _named_defs(
        tmp_path,
        "styled.tsx",
        "const SelectButton = styled('button', { name: 'JoySelect' });\n",
    )

    assert [(kind, name) for kind, name, _, _ in defs] == [("variable_declarator", "SelectButton")]


@requires_tree_sitter
def test_javascript_local_call_and_anonymous_callback_are_not_definitions(tmp_path) -> None:
    defs = _named_defs(
        tmp_path,
        "local.js",
        "function outer() {\n"
        "  const data = compute();\n"
        "  items.map(function (item) { return item; });\n"
        "  return data;\n"
        "}\n",
    )

    assert [(kind, name) for kind, name, _, _ in defs] == [("function_declaration", "outer")]


@requires_tree_sitter
def test_rust_macro_definition_is_a_definition(tmp_path) -> None:
    defs = _named_defs(tmp_path, "select.rs", "macro_rules! select { () => {}; }\n")

    assert [(kind, name) for kind, name, _, _ in defs] == [("macro_definition", "select")]


@requires_tree_sitter
def test_pony_actor_and_members_are_definitions(tmp_path) -> None:
    defs = _named_defs(
        tmp_path,
        "main.pony",
        "actor Main\n"
        "  new create(env: Env) =>\n"
        "    env.out.print(\"hi\")\n"
        "  be ping() =>\n"
        "    None\n",
    )

    kinds = {kind for kind, _, _, _ in defs}
    assert "actor_definition" in kinds
    assert {"constructor", "behavior"} <= kinds


@requires_tree_sitter
def test_python_module_constants_yield_no_definitions(tmp_path) -> None:
    path = tmp_path / "settings.py"
    path.write_text("SECURE_HSTS_SECONDS = 0\nSECURE_SSL_HOST = None\n")

    spans = {"settings.py": [(0, path.stat().st_size)]}
    assert treesitter.extract_def_set_in_spans(spans, str(tmp_path)) == set()
