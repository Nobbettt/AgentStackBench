# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-04-21 from upstream ContextBench.
# Summary of changes: add parser language alias support for C# tree-sitter extraction;
# recognize named function expressions, function-valued declarators, and top-level
# component factories in JS/TS; add Rust macro definitions and Pony support.

"""Definition extraction (tree-sitter required).

This project requires tree-sitter for symbol/definition extraction. If tree-sitter
is not available, symbol extraction raises a clear error instead of silently
falling back to best-effort heuristics.
"""

import os
from typing import Dict, List, Set, Tuple, Iterable

DefNode = Tuple[str, int, int]  # (kind, start_byte, end_byte)

# Language configuration
LANG_MAP = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "c_sharp", ".php": "php",
    ".rb": "ruby", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".pony": "pony"
}

PARSER_LANGUAGE_ALIASES = {
    "c_sharp": "csharp",
}

DEF_NODES = {
    "python": {"function_definition", "class_definition", "async_function_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "java": {"method_declaration", "class_declaration", "interface_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item", "macro_definition"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "ruby": {"method", "class", "module"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "kotlin": {"function_declaration", "class_declaration"},
    "scala": {"function_definition", "class_definition", "trait_definition"},
    "pony": {
        "actor_definition", "class_definition", "primitive_definition", "struct_definition",
        "trait_definition", "interface_definition", "method", "behavior", "constructor",
    },
}

# JS/TS definitions that need structural checks beyond a node-type match.
_JS_LANGS = {"javascript", "typescript", "tsx"}
_JS_FUNCTION_EXPRESSION_TYPES = {"function_expression", "function"}
_JS_FUNCTION_VALUE_TYPES = {"arrow_function", "function_expression", "function"}


def _node_field(node, field: str):
    try:
        return node.child_by_field_name(field)
    except Exception:
        return None


def _is_top_level_declarator(node) -> bool:
    declaration = getattr(node, "parent", None)
    scope = getattr(declaration, "parent", None) if declaration is not None else None
    return getattr(scope, "type", "") in {"program", "export_statement"}


def _is_definition_node(node, lang: str) -> bool:
    """Decide whether a named node counts as a symbol definition for `lang`.

    Beyond the per-language node-type sets, JS/TS code commonly defines
    components and utilities as named function expressions
    (`React.forwardRef(function Select(...) {...})`), function-valued
    declarators (`const f = () => {...}`), or top-level factory calls
    (`const Button = styled('button', {...})`). Anonymous inline callbacks
    stay excluded: they cannot be referenced by name.
    """
    node_type = getattr(node, "type", "")
    if node_type in DEF_NODES.get(lang, set()):
        return True
    if lang not in _JS_LANGS:
        return False
    if node_type in _JS_FUNCTION_EXPRESSION_TYPES:
        return _node_field(node, "name") is not None
    if node_type == "variable_declarator":
        value = _node_field(node, "value")
        value_type = getattr(value, "type", "")
        if value_type in _JS_FUNCTION_VALUE_TYPES:
            return True
        if value_type == "call_expression":
            return _is_top_level_declarator(node)
    return False

_TS_AVAILABLE = False
_PARSERS = {}

try:
    import tree_sitter  # noqa: F401
    # Prefer tree_sitter_languages when available (historical dependency),
    # otherwise fall back to tree_sitter_language_pack (Py>=3.12 friendly).
    try:
        from tree_sitter_languages import get_parser as _get_parser  # type: ignore
    except Exception:  # pragma: no cover
        from tree_sitter_language_pack import get_parser as _get_parser  # type: ignore
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False


def _require_tree_sitter() -> None:
    if _TS_AVAILABLE:
        return
    raise RuntimeError(
        "tree-sitter is required for symbol extraction. "
        "Install: pip install tree-sitter-language-pack"
    )

def available() -> bool:
    """Return True only if tree-sitter + parsers are usable."""
    if not _TS_AVAILABLE:
        return False
    try:
        return _get_parser("python") is not None
    except Exception:
        return False

def _get_parser_for_lang(lang: str):
    """Get parser for language (handles both API versions)."""
    if not available():
        return None
    
    if lang in _PARSERS:
        return _PARSERS[lang]
    
    candidates = [lang]
    alias = PARSER_LANGUAGE_ALIASES.get(lang)
    if alias and alias not in candidates:
        candidates.append(alias)

    for candidate in candidates:
        try:
            parser = _get_parser(candidate)
            _PARSERS[lang] = parser
            return parser
        except Exception:
            continue

    _PARSERS[lang] = None
    return None

def extract_defs(file_path: str) -> List[DefNode]:
    """Extract definition nodes from file."""
    _require_tree_sitter()
    
    lang = LANG_MAP.get(os.path.splitext(file_path.lower())[1])
    if not lang or lang not in DEF_NODES:
        return []
    
    parser = _get_parser_for_lang(lang)
    if not parser:
        raise RuntimeError(
            f"tree-sitter parser for language '{lang}' is not available. "
            "Install: pip install tree-sitter-language-pack"
        )
    
    try:
        with open(file_path, 'rb') as f:
            tree = parser.parse(f.read())
    except Exception:
        return []
    
    result = []

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if not getattr(node, "is_named", False):
            continue

        node_type = getattr(node, "type", "")
        if "comment" in node_type:
            continue

        # Add to result if it's a definition (but still traverse children)
        if _is_definition_node(node, lang):
            result.append((node_type, node.start_byte, node.end_byte))

        # Always traverse children (don't skip based on exclude list)
        for child in reversed(getattr(node, "children", [])):
            stack.append(child)

    return result

def _node_text(src: bytes, node) -> str:
    try:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


_IDENTIFIER_TYPES = {
    "identifier",
    "field_identifier",
    "property_identifier",
    "type_identifier",
    "scoped_identifier",
    "namespace_identifier",
    "method_identifier",
    "constant_identifier",
}


def _iter_descendants(node) -> Iterable:
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        for child in reversed(getattr(cur, "children", []) or []):
            stack.append(child)


def _best_name_for_def(def_node, src: bytes) -> str:
    """Best-effort extraction of a definition name for a tree-sitter def node."""
    # Many languages expose a name field directly.
    try:
        name_node = def_node.child_by_field_name("name")
    except Exception:
        name_node = None

    if name_node is not None:
        name = _node_text(src, name_node).strip()
        if name:
            return name

    # C/C++ often expose declarator.
    try:
        decl = def_node.child_by_field_name("declarator")
    except Exception:
        decl = None

    search_root = decl if decl is not None else def_node

    # Heuristic: prefer plain identifiers over type identifiers.
    best = ""
    best_rank = 10**9
    rank_map = {"identifier": 0, "field_identifier": 1, "property_identifier": 2, "method_identifier": 3}
    for n in _iter_descendants(search_root):
        if not getattr(n, "is_named", False):
            continue
        t = getattr(n, "type", "")
        if t not in _IDENTIFIER_TYPES:
            continue
        txt = _node_text(src, n).strip()
        if not txt:
            continue
        r = rank_map.get(t, 100)
        if r < best_rank:
            best = txt
            best_rank = r
            if best_rank == 0:
                break
    return best


def extract_named_defs(file_path: str) -> List[Tuple[str, str, int, int]]:
    """Extract named definitions from file.

    Returns [(kind, name, start_byte, end_byte)].
    """
    _require_tree_sitter()

    lang = LANG_MAP.get(os.path.splitext(file_path.lower())[1])
    if not lang or lang not in DEF_NODES:
        return []

    parser = _get_parser_for_lang(lang)
    if not parser:
        raise RuntimeError(
            f"tree-sitter parser for language '{lang}' is not available. "
            "Install: pip install tree-sitter-language-pack"
        )

    try:
        with open(file_path, "rb") as f:
            src = f.read()
        tree = parser.parse(src)
    except Exception:
        return []

    result: List[Tuple[str, str, int, int]] = []

    for node in _iter_descendants(tree.root_node):
        if not getattr(node, "is_named", False):
            continue
        node_type = getattr(node, "type", "")
        if "comment" in node_type:
            continue
        if not _is_definition_node(node, lang):
            continue

        name = _best_name_for_def(node, src)
        if not name:
            continue
        result.append((node_type, name, node.start_byte, node.end_byte))

    return result


def extract_def_set_from_symbol_names(
    pred_symbols_by_file: Dict[str, List[str]],
    repo_dir: str,
) -> Set[Tuple[str, str, int, int]]:
    """Map predicted symbol names to tree-sitter def byte ranges.

    Returns {(file, kind, start_byte, end_byte)}.
    """
    out: Set[Tuple[str, str, int, int]] = set()
    if not pred_symbols_by_file:
        return out

    for rel_path, symbols in pred_symbols_by_file.items():
        if not rel_path:
            continue
        if not isinstance(symbols, list) or not symbols:
            continue
        abs_path = os.path.join(repo_dir, rel_path)
        if not os.path.exists(abs_path):
            continue

        named_defs = extract_named_defs(abs_path)
        if not named_defs:
            continue

        by_name: Dict[str, List[Tuple[str, int, int]]] = {}
        for kind, name, s, e in named_defs:
            by_name.setdefault(name, []).append((kind, s, e))

        for raw in symbols:
            if not raw or not isinstance(raw, str):
                continue
            sym = raw.strip()
            if not sym:
                continue
            candidates = [sym]
            if "." in sym:
                candidates.append(sym.split(".")[-1])

            matched = False
            for cand in candidates:
                defs = by_name.get(cand)
                if not defs:
                    continue
                for kind, s, e in defs:
                    out.add((rel_path, kind, s, e))
                matched = True
                break

            if not matched:
                continue

    return out


def extract_def_set_in_spans(spans_by_file: Dict[str, List[Tuple[int, int]]], repo_dir: str) -> Set[Tuple[str, str, int, int]]:
    """
    Extract definitions that overlap with given byte spans.
    Returns {(file, kind, start_byte, end_byte)}.
    """
    result = set()
    for file_path, byte_intervals in spans_by_file.items():
        abs_path = os.path.join(repo_dir, file_path)
        if not os.path.exists(abs_path):
            continue
        
        # Get all definitions in this file
        all_defs = extract_defs(abs_path)
        
        # Keep only definitions that overlap with our spans
        for kind, def_start, def_end in all_defs:
            # Check if this definition overlaps any of our spans
            for span_start, span_end in byte_intervals:
                # Overlap check: def and span have any byte in common
                if not (def_end < span_start or def_start > span_end):
                    result.add((file_path, kind, def_start, def_end))
                    break  # Already added, no need to check other spans
    
    return result
