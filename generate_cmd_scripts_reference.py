#!/usr/bin/env python3
"""Generate an offline Markdown reference for Asgard console scripts.

Run this file from the Asgard workspace root with::

    python asgard_guis/generate_cmd_scripts_reference.py

The entry-point modules are parsed as source code. They are never imported or
executed, so documentation generation does not require their runtime, GUI, or
hardware dependencies.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "generate_cmd_scripts_reference.py requires Python 3.11+ or tomli"
        ) from error


DEFAULT_OUTPUT = Path("asgard_guis/docs/cmd_scripts_reference.md")
REPOSITORIES = (
    Path("asgard_guis"),
    Path("asgard-alignment"),
    Path("dcs"),
)
DEPLOYMENT_HOSTS = {
    "asgard_guis": "wag",
    "asgard-alignment": "mimir",
    "dcs": "mimir",
}
GITHUB_REPOSITORIES = {
    "asgard_guis": "https://github.com/asgard-vlti/asgard_guis",
    "asgard-alignment": "https://github.com/asgard-vlti/asgard-alignment",
    "dcs": "https://github.com/asgard-vlti/dcs",
}

UNKNOWN = object()


@dataclass(frozen=True)
class CliArgument:
    names: tuple[str, ...]
    destination: str
    type_name: str
    required: bool
    default: str | None
    choices: tuple[str, ...] | None
    choices_expression: str | None
    nargs: str | None
    metavar: tuple[str, ...] | None
    action: str | None
    description: str
    source_line: int
    manual: bool = False


@dataclass(frozen=True)
class ExclusiveGroup:
    destinations: tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class ScriptReference:
    command: str
    target: str
    source: Path
    source_line: int
    description: str
    arguments: tuple[CliArgument, ...]
    exclusive_groups: tuple[ExclusiveGroup, ...]
    nonstandard_resolution: bool


@dataclass(frozen=True)
class RepositoryReference:
    name: str
    root: Path
    pyproject: Path
    scripts: tuple[ScriptReference, ...]


class ExtractionError(RuntimeError):
    """Raised when source metadata cannot be extracted safely."""


def _node_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return "<dynamic expression>"


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    constants: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is None:
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
    return constants


def _safe_value(
    node: ast.AST,
    constants: dict[str, ast.expr],
    seen: frozenset[str] = frozenset(),
) -> Any:
    """Evaluate a deliberately small, side-effect-free expression subset."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in {"None", "True", "False"}:
            return {"None": None, "True": True, "False": False}[node.id]
        if node.id in seen or node.id not in constants:
            return UNKNOWN
        return _safe_value(constants[node.id], constants, seen | {node.id})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_value(node.operand, constants, seen)
        if value is UNKNOWN or not isinstance(value, (int, float, complex)):
            return UNKNOWN
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[Any] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                expanded = _safe_value(element.value, constants, seen)
                if expanded is UNKNOWN or not isinstance(expanded, (list, tuple, set)):
                    return UNKNOWN
                values.extend(expanded)
            else:
                value = _safe_value(element, constants, seen)
                if value is UNKNOWN:
                    return UNKNOWN
                values.append(value)
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                return UNKNOWN
            key = _safe_value(key_node, constants, seen)
            value = _safe_value(value_node, constants, seen)
            if key is UNKNOWN or value is UNKNOWN:
                return UNKNOWN
            result[key] = value
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _safe_value(node.left, constants, seen)
        right = _safe_value(node.right, constants, seen)
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        try:
            return left + right
        except TypeError:
            return UNKNOWN
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "sorted" and len(node.args) == 1 and not node.keywords:
            value = _safe_value(node.args[0], constants, seen)
            if value is not UNKNOWN:
                try:
                    return sorted(value)
                except (TypeError, ValueError):
                    return UNKNOWN
    return UNKNOWN


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, tuple):
        return "(" + ", ".join(_display_value(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_display_value(item) for item in value) + "]"
    if isinstance(value, set):
        return "{" + ", ".join(sorted(_display_value(item) for item in value)) + "}"
    if isinstance(value, dict):
        pairs = (
            f"{_display_value(key)}: {_display_value(item)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(pairs) + "}"
    return repr(value)


def _format_expression(node: ast.AST, constants: dict[str, ast.expr]) -> str:
    value = _safe_value(node, constants)
    return _node_text(node) if value is UNKNOWN else _display_value(value)


def _literal_string(
    node: ast.AST | None, constants: dict[str, ast.expr]
) -> str | None:
    if node is None:
        return None
    value = _safe_value(node, constants)
    return value if isinstance(value, str) else None


def _first_paragraph(value: str) -> str:
    paragraphs = re.split(r"\n\s*\n", value.strip(), maxsplit=1)
    return re.sub(r"\s+", " ", paragraphs[0]).strip() if paragraphs else ""


def _walk_scope(scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """Walk one lexical scope without entering nested functions or classes."""
    stack: list[ast.AST] = list(reversed(scope.body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _called_local_functions(
    entry: ast.FunctionDef | ast.AsyncFunctionDef,
    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    selected: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {entry.name: entry}
    pending = [entry]
    while pending:
        function = pending.pop()
        for node in _walk_scope(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            called = definitions.get(node.func.id)
            if called is not None and called.name not in selected:
                selected[called.name] = called
                pending.append(called)
    return tuple(sorted(selected.values(), key=lambda item: item.lineno))


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def _assignment_value(node: ast.AST) -> ast.expr | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _call_name(node: ast.Call) -> str:
    return _node_text(node.func)


def _is_argument_parser_call(node: ast.Call) -> bool:
    name = _call_name(node)
    return name == "ArgumentParser" or name.endswith(".ArgumentParser")


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}


def _receiver_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return call.func.value.id
    return None


def _destination(names: tuple[str, ...], keywords: dict[str, ast.expr], constants: dict[str, ast.expr]) -> str:
    explicit = _literal_string(keywords.get("dest"), constants)
    if explicit:
        return explicit
    if names and not names[0].startswith("-"):
        return names[0]
    long_names = [name for name in names if name.startswith("--")]
    selected = long_names[0] if long_names else names[0]
    return selected.lstrip("-").replace("-", "_")


def _argument_from_call(
    call: ast.Call,
    constants: dict[str, ast.expr],
) -> CliArgument:
    keywords = _keyword_map(call)
    names: list[str] = []
    for name_node in call.args:
        name = _literal_string(name_node, constants)
        names.append(name if name is not None else _node_text(name_node))
    if not names:
        raise ExtractionError(f"add_argument() at line {call.lineno} has no argument name")
    name_tuple = tuple(names)
    positional = not name_tuple[0].startswith("-")

    action = _literal_string(keywords.get("action"), constants)
    nargs_value = (
        _safe_value(keywords["nargs"], constants) if "nargs" in keywords else None
    )
    nargs = None if nargs_value is None else _display_value(nargs_value)
    if nargs_value is UNKNOWN:
        nargs = _node_text(keywords["nargs"])

    required_value = (
        _safe_value(keywords["required"], constants)
        if "required" in keywords
        else UNKNOWN
    )
    if isinstance(required_value, bool):
        required = required_value
    elif positional:
        required = nargs_value not in {"?", "*"}
    else:
        required = False

    if "default" in keywords:
        default = _format_expression(keywords["default"], constants)
    elif required:
        default = None
    elif action == "store_true":
        default = "False"
    elif action == "store_false":
        default = "True"
    else:
        default = "None"

    choices_values: tuple[str, ...] | None = None
    choices_expression: str | None = None
    if "choices" in keywords:
        choices = _safe_value(keywords["choices"], constants)
        if isinstance(choices, (list, tuple, set)):
            values = list(choices)
            if isinstance(choices, set):
                values.sort(key=lambda item: repr(item))
            choices_values = tuple(_display_value(item) for item in values)
        else:
            choices_expression = _node_text(keywords["choices"])

    metavar: tuple[str, ...] | None = None
    if "metavar" in keywords:
        value = _safe_value(keywords["metavar"], constants)
        if isinstance(value, (list, tuple)):
            metavar = tuple(_display_value(item) for item in value)
        elif value is not UNKNOWN:
            metavar = (_display_value(value),)
        else:
            metavar = (_node_text(keywords["metavar"]),)

    if action in {"store_true", "store_false"}:
        type_name = "bool"
    elif "type" in keywords:
        type_name = _node_text(keywords["type"])
    else:
        type_name = "str"

    help_node = keywords.get("help")
    description = _literal_string(help_node, constants)
    if description is None and help_node is not None:
        description = _node_text(help_node)

    return CliArgument(
        names=name_tuple,
        destination=_destination(name_tuple, keywords, constants),
        type_name=type_name,
        required=required,
        default=default,
        choices=choices_values,
        choices_expression=choices_expression,
        nargs=nargs,
        metavar=metavar,
        action=action,
        description=description or "",
        source_line=call.lineno,
    )


def _extract_argparse(
    tree: ast.Module,
    entry: ast.FunctionDef | ast.AsyncFunctionDef,
    constants: dict[str, ast.expr],
) -> tuple[str | None, tuple[CliArgument, ...], tuple[ExclusiveGroup, ...]]:
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scopes: tuple[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef, ...] = (
        tree,
        *_called_local_functions(entry, definitions),
    )

    parser_descriptions: list[tuple[int, str]] = []
    arguments: list[tuple[int, CliArgument, str | None]] = []
    group_required: dict[str, bool] = {}

    for scope in scopes:
        nodes = tuple(_walk_scope(scope))
        parser_names: set[str] = set()
        group_names: set[str] = set()

        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            name = _assignment_name(node)
            value = _assignment_value(node)
            if name is None or not isinstance(value, ast.Call):
                continue
            if _is_argument_parser_call(value):
                parser_names.add(name)
                description_node = _keyword_map(value).get("description")
                description = _literal_string(description_node, constants)
                if description:
                    parser_descriptions.append((value.lineno, description))

        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            name = _assignment_name(node)
            value = _assignment_value(node)
            if name is None or not isinstance(value, ast.Call):
                continue
            receiver = _receiver_name(value)
            if (
                receiver in parser_names
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "add_mutually_exclusive_group"
            ):
                group_names.add(name)
                required_node = _keyword_map(value).get("required")
                required = _safe_value(required_node, constants) if required_node else False
                group_required[name] = required if isinstance(required, bool) else False

        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue
            receiver = _receiver_name(node)
            if receiver not in parser_names and receiver not in group_names:
                continue
            group_name = receiver if receiver in group_names else None
            argument = _argument_from_call(node, constants)
            arguments.append((node.lineno, argument, group_name))

    unique_arguments: list[CliArgument] = []
    group_destinations: dict[str, list[str]] = {}
    seen_arguments: set[tuple[int, tuple[str, ...]]] = set()
    for _, argument, group_name in sorted(arguments, key=lambda item: item[0]):
        identity = (argument.source_line, argument.names)
        if identity in seen_arguments:
            continue
        seen_arguments.add(identity)
        unique_arguments.append(argument)
        if group_name is not None:
            group_destinations.setdefault(group_name, []).append(argument.destination)

    groups = tuple(
        ExclusiveGroup(tuple(destinations), group_required.get(name, False))
        for name, destinations in group_destinations.items()
    )
    description = (
        min(parser_descriptions, key=lambda item: item[0])[1]
        if parser_descriptions
        else None
    )
    return description, tuple(unique_arguments), groups


def _sys_argv_index(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if not (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "sys"
        and value.attr == "argv"
    ):
        return None
    index = _safe_value(node.slice, {})
    return index if isinstance(index, int) and index > 0 else None


def _contained_argv_indices(node: ast.AST) -> set[int]:
    indices: set[int] = set()
    for child in ast.walk(node):
        index = _sys_argv_index(child)
        if index is not None:
            indices.add(index)
    return indices


def _contains_exit(statements: Iterable[ast.stmt]) -> bool:
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Call) and _call_name(node) in {
                "exit",
                "sys.exit",
            }:
                return True
    return False


def _len_sys_argv_comparison(node: ast.AST) -> tuple[str, int] | None:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    left = node.left
    if not (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Name)
        and left.func.id == "len"
        and len(left.args) == 1
        and isinstance(left.args[0], ast.Attribute)
        and isinstance(left.args[0].value, ast.Name)
        and left.args[0].value.id == "sys"
        and left.args[0].attr == "argv"
    ):
        return None
    comparator = _safe_value(node.comparators[0], {})
    if not isinstance(comparator, int):
        return None
    operator_names = {
        ast.Eq: "eq",
        ast.NotEq: "ne",
        ast.Gt: "gt",
        ast.GtE: "ge",
        ast.Lt: "lt",
        ast.LtE: "le",
    }
    operator = operator_names.get(type(node.ops[0]))
    return (operator, comparator) if operator else None


def _usage_strings(entry: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    values: list[str] = []
    for node in _walk_scope(entry):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.value.strip().lower().startswith("usage:"):
            values.append(node.value.strip())
    return tuple(values)


def _manual_name_from_usage(usages: tuple[str, ...], index: int) -> str | None:
    for usage in usages:
        names = re.findall(r"<([A-Za-z_][A-Za-z0-9_-]*)>", usage)
        if 0 < index <= len(names):
            return names[index - 1]
    return None


def _assignment_target_for_index(
    entry: ast.FunctionDef | ast.AsyncFunctionDef, index: int
) -> str | None:
    for node in _walk_scope(entry):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = _assignment_name(node)
        value = _assignment_value(node)
        if target and value is not None and index in _contained_argv_indices(value):
            return target
    return None


def _manual_choices(
    entry: ast.FunctionDef | ast.AsyncFunctionDef,
    index: int,
    destination: str | None,
    constants: dict[str, ast.expr],
) -> tuple[str, ...] | None:
    values: list[Any] = []
    for node in _walk_scope(entry):
        if not isinstance(node, ast.Compare):
            continue
        left_matches = index in _contained_argv_indices(node.left)
        if isinstance(node.left, ast.Name) and node.left.id == destination:
            left_matches = True
        if not left_matches:
            continue
        for comparator in node.comparators:
            value = _safe_value(comparator, constants)
            if isinstance(value, (list, tuple, set)):
                values.extend(value)
            elif value is not UNKNOWN and not isinstance(value, bool):
                values.append(value)
    if not values:
        for usage in _usage_strings(entry):
            for choices_text in re.findall(r"\[([^\]]*\|[^\]]*)\]", usage):
                values.extend(part.strip() for part in choices_text.split("|"))
    if not values:
        return None
    result: list[str] = []
    for value in values:
        display = _display_value(value)
        if display not in result:
            result.append(display)
    return tuple(result)


def _manual_default(
    entry: ast.FunctionDef | ast.AsyncFunctionDef,
    index: int,
    assigned_name: str | None,
    constants: dict[str, ast.expr],
) -> tuple[bool, str | None]:
    if assigned_name:
        for node in _walk_scope(entry):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if _assignment_name(node) != assigned_name:
                continue
            value = _assignment_value(node)
            if value is None or index in _contained_argv_indices(value):
                continue
            return False, _format_expression(value, constants)

    for node in _walk_scope(entry):
        if not isinstance(node, ast.If):
            continue
        for comparison in ast.walk(node.test):
            result = _len_sys_argv_comparison(comparison)
            if result in {("gt", index), ("ge", index + 1)}:
                return False, "Not supplied"
        result = _len_sys_argv_comparison(node.test)
        if result == ("ne", index + 1) and _contains_exit(node.body):
            return True, None
    return True, None


def _extract_manual_arguments(
    entry: ast.FunctionDef | ast.AsyncFunctionDef,
    constants: dict[str, ast.expr],
) -> tuple[tuple[CliArgument, ...], tuple[str, ...]]:
    index_nodes: dict[int, list[ast.Subscript]] = {}
    for node in _walk_scope(entry):
        index = _sys_argv_index(node)
        if index is not None:
            index_nodes.setdefault(index, []).append(node)
    if not index_nodes:
        return (), ()

    usages = _usage_strings(entry)
    arguments: list[CliArgument] = []
    warnings: list[str] = []
    for index, nodes in sorted(index_nodes.items()):
        assigned_name = _assignment_target_for_index(entry, index)
        usage_name = _manual_name_from_usage(usages, index)
        choices = _manual_choices(entry, index, assigned_name, constants)
        destination = usage_name or assigned_name
        if destination is None and choices and len(choices) == 1:
            destination = choices[0]
        if destination is None:
            destination = f"argv[{index}]"
            warnings.append(
                f"line {nodes[0].lineno}: could not infer a name for sys.argv[{index}]"
            )
        required, default = _manual_default(entry, index, assigned_name, constants)
        usage_text = (
            re.sub(r"^Usage:\s*", "", usages[0], flags=re.IGNORECASE)
            if usages
            else ""
        )
        usage_note = f" Source usage: {usage_text}" if usage_text else ""
        arguments.append(
            CliArgument(
                names=(destination,),
                destination=destination,
                type_name="str",
                required=required,
                default=default,
                choices=choices,
                choices_expression=None,
                nargs=None,
                metavar=None,
                action=None,
                description=(
                    f"Manual positional argument read from sys.argv[{index}]."
                    f"{usage_note}"
                ),
                source_line=nodes[0].lineno,
                manual=True,
            )
        )
    return tuple(arguments), tuple(warnings)


def _package_names(data: dict[str, Any]) -> tuple[str, ...]:
    packages = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    if isinstance(packages, str):
        packages = [packages]
    if not isinstance(packages, list):
        return ()
    return tuple(
        Path(package).parts[0]
        for package in packages
        if isinstance(package, str) and Path(package).parts
    )


def _source_candidates(
    repository: Path, module: str, packages: tuple[str, ...]
) -> tuple[tuple[Path, bool], ...]:
    parts = module.split(".")
    candidates = [
        (repository.joinpath(*parts).with_suffix(".py"), False),
        (repository.joinpath(*parts, "__init__.py"), False),
    ]
    if parts and parts[0] in packages and len(parts) > 1:
        candidates.extend(
            [
                (repository.joinpath(*parts[1:]).with_suffix(".py"), True),
                (repository.joinpath(*parts[1:], "__init__.py"), True),
            ]
        )
    unique: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for path, fallback in candidates:
        if path not in seen:
            seen.add(path)
            unique.append((path, fallback))
    return tuple(unique)


def _resolve_source(
    repository: Path,
    command: str,
    module: str,
    packages: tuple[str, ...],
) -> tuple[Path, bool]:
    matches = [
        (path, fallback)
        for path, fallback in _source_candidates(repository, module, packages)
        if path.is_file()
    ]
    exact = [match for match in matches if not match[1]]
    selected = exact or matches
    if len(selected) == 1:
        return selected[0]
    if len(selected) > 1:
        paths = ", ".join(str(path) for path, _ in selected)
        raise ExtractionError(
            f"{repository.name}/{command}: ambiguous source for {module}: {paths}"
        )
    attempted = ", ".join(
        str(path) for path, _ in _source_candidates(repository, module, packages)
    )
    raise ExtractionError(
        f"{repository.name}/{command}: source for {module} was not found; tried {attempted}"
    )


def _load_pyproject(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file_object:
            data = tomllib.load(file_object)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExtractionError(f"could not read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ExtractionError(f"{path}: expected a TOML table")
    return data


def _extract_script(
    repository: Path,
    command: str,
    target: str,
    packages: tuple[str, ...],
) -> tuple[ScriptReference, tuple[str, ...]]:
    if target.count(":") != 1:
        raise ExtractionError(
            f"{repository.name}/{command}: invalid entry point {target!r}; "
            "expected module:function"
        )
    module, function_name = target.split(":", 1)
    if not module or not function_name.isidentifier():
        raise ExtractionError(
            f"{repository.name}/{command}: invalid entry point {target!r}; "
            "expected module:function"
        )

    source, nonstandard = _resolve_source(repository, command, module, packages)
    try:
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
    except (OSError, SyntaxError) as error:
        raise ExtractionError(f"{repository.name}/{command}: {error}") from error

    entries = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(entries) != 1:
        raise ExtractionError(
            f"{repository.name}/{command}: expected one top-level function "
            f"{function_name!r} in {source}, found {len(entries)}"
        )
    entry = entries[0]
    constants = _module_constants(tree)
    parser_description, parser_arguments, groups = _extract_argparse(
        tree, entry, constants
    )
    manual_arguments, manual_warnings = _extract_manual_arguments(entry, constants)

    function_description = ast.get_docstring(entry) or ""
    module_description = ast.get_docstring(tree) or ""
    description = (
        _first_paragraph(parser_description)
        if parser_description
        else _first_paragraph(function_description)
        or _first_paragraph(module_description)
        or "No description provided in source."
    )
    arguments = tuple(
        sorted((*parser_arguments, *manual_arguments), key=lambda item: item.source_line)
    )
    warnings = list(manual_warnings)
    if nonstandard:
        try:
            source_display = source.relative_to(repository.parent)
        except ValueError:
            source_display = source
        warnings.append(
            f"resolved {target} to {source_display} after stripping the declared "
            "package prefix"
        )

    return (
        ScriptReference(
            command=command,
            target=target,
            source=source,
            source_line=entry.lineno,
            description=description,
            arguments=arguments,
            exclusive_groups=groups,
            nonstandard_resolution=nonstandard,
        ),
        tuple(warnings),
    )


def load_repositories(
    workspace: Path,
    repository_paths: tuple[Path, ...] = REPOSITORIES,
) -> tuple[tuple[RepositoryReference, ...], tuple[str, ...]]:
    repositories: list[RepositoryReference] = []
    warnings: list[str] = []
    for relative_repository in repository_paths:
        repository = workspace / relative_repository
        pyproject = repository / "pyproject.toml"
        if not repository.is_dir():
            raise ExtractionError(
                f"required repository was not found: {relative_repository}; "
                "run this script from the Asgard workspace root"
            )
        data = _load_pyproject(pyproject)
        project = data.get("project")
        scripts = project.get("scripts") if isinstance(project, dict) else None
        if not isinstance(scripts, dict):
            raise ExtractionError(f"{pyproject}: [project.scripts] is missing")
        extracted: list[ScriptReference] = []
        for command, target in scripts.items():
            if not isinstance(command, str) or not isinstance(target, str):
                raise ExtractionError(
                    f"{pyproject}: every [project.scripts] entry must map strings to strings"
                )
            script, script_warnings = _extract_script(
                repository, command, target, _package_names(data)
            )
            extracted.append(script)
            warnings.extend(
                f"{relative_repository}/{command}: {warning}"
                for warning in script_warnings
            )
        repositories.append(
            RepositoryReference(
                name=relative_repository.name,
                root=repository,
                pyproject=pyproject,
                scripts=tuple(
                    sorted(extracted, key=lambda script: script.command.casefold())
                ),
            )
        )
    return tuple(repositories), tuple(warnings)


def _choices_text(argument: CliArgument) -> str:
    if argument.choices is not None:
        return ", ".join(argument.choices)
    return argument.choices_expression or ""


def _usage_value(argument: CliArgument) -> str:
    if argument.choices:
        value = (
            argument.choices[0]
            if len(argument.choices) == 1
            else "{" + ",".join(argument.choices) + "}"
        )
    elif argument.metavar:
        value = " ".join(argument.metavar)
    else:
        value = argument.destination.upper()

    if argument.action in {"store_true", "store_false"}:
        return ""
    if argument.nargs == "+":
        return f"{value} [{value} ...]"
    if argument.nargs == "*":
        return f"{value} ..."
    if argument.nargs == "?":
        return value
    if argument.nargs and argument.nargs.isdigit():
        if argument.metavar and len(argument.metavar) == int(argument.nargs):
            return value
        return " ".join(value for _ in range(int(argument.nargs)))
    return value


def _argument_token(argument: CliArgument) -> str:
    positional = not argument.names[0].startswith("-")
    value = _usage_value(argument)
    if positional:
        token = value
    else:
        token = argument.names[0]
        if value:
            token += f" {value}"
    return token if argument.required else f"[{token}]"


def _script_usage(script: ScriptReference) -> str:
    arguments_by_destination = {
        argument.destination: argument for argument in script.arguments
    }
    grouped_destinations = {
        destination
        for group in script.exclusive_groups
        for destination in group.destinations
    }
    tokens = [script.command]
    for argument in script.arguments:
        if argument.destination not in grouped_destinations:
            tokens.append(_argument_token(argument))
    for group in script.exclusive_groups:
        choices = [
            _argument_token(arguments_by_destination[destination]).strip("[]")
            for destination in group.destinations
            if destination in arguments_by_destination
        ]
        if choices:
            token = " | ".join(choices)
            tokens.append(f"({token})" if group.required else f"[{token}]")
    return " ".join(tokens)


def _markdown_cell(value: str) -> str:
    value = value.replace("|", "\\|").replace("\r\n", "\n")
    return value.replace("\n", "<br>")


def _github_link(
    repository: RepositoryReference,
    source_path: Path,
    line: int | None = None,
) -> str:
    try:
        relative = source_path.relative_to(repository.root).as_posix()
    except ValueError:
        relative = source_path.as_posix()
    github_repository = GITHUB_REPOSITORIES.get(repository.name)
    if github_repository is None:
        return f"{relative}#L{line}" if line is not None else relative
    link = f"{github_repository}/blob/main/{quote(relative)}"
    return f"{link}#L{line}" if line is not None else link


def _display_path(path: Path, repository: RepositoryReference) -> Path:
    try:
        return path.relative_to(repository.root.parent)
    except ValueError:
        return path


def _render_argument_table(arguments: tuple[CliArgument, ...]) -> list[str]:
    if not arguments:
        return ["_No command-line arguments._", ""]
    lines = [
        "| Argument | Type | Required | Default | Choices | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for argument in arguments:
        names = ", ".join(f"`{name}`" for name in argument.names)
        default = "—" if argument.default is None else f"`{_markdown_cell(argument.default)}`"
        choices = _choices_text(argument)
        choices_cell = f"`{_markdown_cell(choices)}`" if choices else "—"
        description = _markdown_cell(argument.description) or "—"
        lines.append(
            f"| {names} | `{_markdown_cell(argument.type_name)}` | "
            f"{'Yes' if argument.required else 'No'} | {default} | "
            f"{choices_cell} | {description} |"
        )
    lines.append("")
    return lines


def _render_repository(repository: RepositoryReference) -> list[str]:
    pyproject_link = _github_link(repository, repository.pyproject)
    pyproject_display = _display_path(repository.pyproject, repository)
    deployment_host = DEPLOYMENT_HOSTS.get(repository.name)
    heading = (
        f"{repository.name} (on {deployment_host})"
        if deployment_host
        else repository.name
    )
    lines = [
        f"## {heading}",
        "",
        f"Scripts declared in [`{pyproject_display.as_posix()}`]({pyproject_link}): "
        f"{len(repository.scripts)}.",
        "",
        "### Quick reference",
        "",
        "| Command | Invocation | Description |",
        "| --- | --- | --- |",
    ]
    for script in repository.scripts:
        lines.append(
            f"| `{script.command}` | `{_markdown_cell(_script_usage(script))}` | "
            f"{_markdown_cell(script.description)} |"
        )
    lines.extend(["", "### Script details", ""])

    for index, script in enumerate(repository.scripts):
        if index:
            lines.extend(["---", ""])
        source_link = _github_link(repository, script.source, script.source_line)
        source_display = _display_path(script.source, repository)
        lines.extend(
            [
                f"#### `{script.command}`",
                "",
                script.description,
                "",
                f"**Source:** [`{source_display.as_posix()}:{script.source_line}`]({source_link})",
                "",
            ]
        )
        if script.nonstandard_resolution:
            lines.extend(
                [
                    "_Resolution note: the declared module path has no matching file; "
                    "the source was found after stripping its declared package prefix._",
                    "",
                ]
            )
        lines.extend(
            [
                f"**Invocation:** `{_script_usage(script)}`",
                "",
                "**Arguments**",
                "",
            ]
        )
        lines.extend(_render_argument_table(script.arguments))
        if script.exclusive_groups:
            lines.extend(["**Argument constraints**", ""])
            for group in script.exclusive_groups:
                arguments_by_destination = {
                    argument.destination: argument for argument in script.arguments
                }
                names = ", ".join(
                    "/".join(
                        f"`{name}`"
                        for name in arguments_by_destination[destination].names
                    )
                    for destination in group.destinations
                    if destination in arguments_by_destination
                )
                requirement = (
                    "exactly one is required"
                    if group.required
                    else "at most one may be used"
                )
                lines.append(f"- {names}: {requirement}.")
            lines.append("")
    return lines


def render_document(
    repositories: tuple[RepositoryReference, ...], output_path: Path
) -> str:
    lines = [
        "# Asgard Command Script Reference",
        "",
        "<!-- Generated by asgard_guis/generate_cmd_scripts_reference.py. Do not edit manually. -->",
        "",
        "This document is generated offline from `[project.scripts]` declarations and "
        "Python source. Entry-point modules are parsed but never imported or executed.",
        "",
        "Regenerate it from the Asgard workspace root with:",
        "",
        "```bash",
        "python asgard_guis/generate_cmd_scripts_reference.py",
        "```",
        "",
    ]
    for repository in repositories:
        lines.extend(_render_repository(repository))
    return "\n".join(lines).rstrip() + "\n"


def _write_if_changed(output_path: Path, content: str) -> bool:
    if output_path.is_file() and output_path.read_text(encoding="utf-8") == content:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the offline Asgard command-script reference."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the output is current without modifying it.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace = Path.cwd()
    output = args.output if args.output.is_absolute() else workspace / args.output
    try:
        repositories, warnings = load_repositories(workspace)
        document = render_document(repositories, output)
    except (ExtractionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    display_output = args.output if not args.output.is_absolute() else output
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != document:
            print(f"out of date: {display_output}", file=sys.stderr)
            return 1
        print(f"up to date: {display_output}")
        return 0

    changed = _write_if_changed(output, document)
    state = "updated" if changed else "up to date"
    count = sum(len(repository.scripts) for repository in repositories)
    print(
        f"{state}: {display_output} "
        f"({count} scripts across {len(repositories)} repositories)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
