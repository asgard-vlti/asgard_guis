#!/usr/bin/env python3
"""Generate an offline Markdown reference for the Asgard command servers.
This program is not intended to be run on wag.
Instead, it should be run on a local machine with all asgard repos in a directory, and the file called using

python asgard_guis/generate_command_reference.py

The result is a Markdown file, located at asgard_guis/docs/dcs_command_reference.md by default, which can be viewed in a browser or text editor.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import string
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# TODO: headers at the top of each server that are manually written and point to additional docs
# e.g. a list of axes for the MDS server

# TODO: a generic command doc summarising all the script commands possible in each package too?

DEFAULT_OUTPUT = Path("asgard_guis/docs/dcs_command_reference.md")


@dataclass(frozen=True)
class Argument:
    name: str
    type_name: str = "unknown"
    default: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    arguments: tuple[Argument, ...]
    return_type: str
    source_line: int
    usage: str | None = None
    output_description: str = ""


@dataclass(frozen=True)
class Server:
    name: str
    source: Path | None
    protocol: str
    commands: tuple[Command, ...]


class ExtractionError(RuntimeError):
    """Raised when command metadata cannot be extracted safely."""


COMMANDER_BUILT_INS = (
    Command(
        name="help",
        description="List every command and its description.",
        arguments=(),
        return_type="std::string",
        source_line=0,
    ),
    Command(
        name="command_names",
        description="List all available command names.",
        arguments=(),
        return_type="std::vector<std::string>",
        source_line=0,
    ),
    Command(
        name="description",
        description="Describe one command, including its signature.",
        arguments=(
            Argument(
                "name",
                "std::string",
                description="Name of the command to describe.",
            ),
        ),
        return_type="std::string",
        source_line=0,
    ),
    Command(
        name="signature",
        description="Return the arguments and return type for one command.",
        arguments=(
            Argument(
                "name",
                "std::string",
                description="Name of the command to inspect.",
            ),
        ),
        return_type="JSON",
        source_line=0,
    ),
    Command(
        name="arguments",
        description="Return argument metadata for one command.",
        arguments=(
            Argument(
                "name",
                "std::string",
                description="Name of the command to inspect.",
            ),
        ),
        return_type="JSON",
        source_line=0,
    ),
    Command(
        name="return_type",
        description="Return the result type for one command.",
        arguments=(
            Argument(
                "name",
                "std::string",
                description="Name of the command to inspect.",
            ),
        ),
        return_type="JSON",
        source_line=0,
    ),
)


COMMANDER_SOURCES = (
    (
        "CRED1 camera server",
        Path("dcs/asgard-cred1-server/cred1_onsky_server.c"),
        (),
    ),
    (
        "Deformable mirror server",
        Path("dcs/asgard-dm-server/asgard_commander_MDM_high_perf_server.c"),
        (),
    ),
    (
        "Heimdallr",
        Path("dcs/heimdallr/heimdallr.cpp"),
        (Path("dcs/heimdallr/heimdallr.h"),),
    ),
)

MDS_SOURCE = Path("asgard-alignment/asgard_alignment/MultiDeviceServer.py")


def _cpp_literal_end(text: str, start: int) -> int | None:
    """Return the end of a C++ string/character literal at ``start``."""
    raw_prefixes = ('u8R"', 'uR"', 'UR"', 'LR"', 'R"')
    for prefix in raw_prefixes:
        if not text.startswith(prefix, start):
            continue
        delimiter_start = start + len(prefix)
        opening = text.find("(", delimiter_start)
        if opening == -1 or opening - delimiter_start > 16:
            return None
        delimiter = text[delimiter_start:opening]
        closing = text.find(")" + delimiter + '"', opening + 1)
        if closing == -1:
            raise ExtractionError(f"Unterminated raw C++ string at offset {start}")
        return closing + len(delimiter) + 2

    ordinary_prefixes = ('u8"', 'u"', 'U"', 'L"', '"', "u8'", "u'", "U'", "L'", "'")
    for prefix in ordinary_prefixes:
        if not text.startswith(prefix, start):
            continue
        quote = prefix[-1]
        pos = start + len(prefix)
        while pos < len(text):
            if text[pos] == "\\":
                pos += 2
            elif text[pos] == quote:
                return pos + 1
            else:
                pos += 1
        raise ExtractionError(f"Unterminated C++ literal at offset {start}")
    return None


def _mask_cpp_comments(text: str) -> str:
    """Replace C++ comments with spaces while preserving offsets and newlines."""
    result: list[str] = []
    pos = 0
    while pos < len(text):
        literal_end = _cpp_literal_end(text, pos)
        if literal_end is not None:
            result.append(text[pos:literal_end])
            pos = literal_end
            continue
        if text.startswith("//", pos):
            end = text.find("\n", pos + 2)
            if end == -1:
                end = len(text)
            result.append(" " * (end - pos))
            pos = end
            continue
        if text.startswith("/*", pos):
            end = text.find("*/", pos + 2)
            if end == -1:
                raise ExtractionError("Unterminated C++ block comment")
            end += 2
            comment = text[pos:end]
            result.append("".join("\n" if char == "\n" else " " for char in comment))
            pos = end
            continue
        result.append(text[pos])
        pos += 1
    return "".join(result)


def _matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 1
    pos = start + 1
    while pos < len(text):
        literal_end = _cpp_literal_end(text, pos)
        if literal_end is not None:
            pos = literal_end
            continue
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise ExtractionError(f"Unterminated {opening}{closing} block at offset {start}")


def _split_cpp_arguments(text: str) -> list[str]:
    arguments: list[str] = []
    start = 0
    pos = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    while pos < len(text):
        literal_end = _cpp_literal_end(text, pos)
        if literal_end is not None:
            pos = literal_end
            continue
        char = text[pos]
        if char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            arguments.append(text[start:pos].strip())
            start = pos + 1
        pos += 1
    if stack:
        raise ExtractionError("Unbalanced delimiters in C++ command registration")
    arguments.append(text[start:].strip())
    return arguments


def _decode_cpp_string_token(token: str) -> str:
    raw_match = re.fullmatch(
        r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]*)\((.*)\)\1"', token, re.DOTALL
    )
    if raw_match:
        return raw_match.group(2)

    token = re.sub(r"^(?:u8|u|U|L)", "", token)
    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError) as error:
        raise ExtractionError(f"Unsupported C++ string literal: {token!r}") from error
    if not isinstance(value, str):
        raise ExtractionError(f"Expected a C++ string literal, got: {token!r}")
    return value


def _cpp_string_prefix(expression: str) -> tuple[str, int] | None:
    values: list[str] = []
    pos = 0
    while True:
        while pos < len(expression) and expression[pos].isspace():
            pos += 1
        end = _cpp_literal_end(expression, pos)
        if end is None or expression[end - 1] == "'":
            break
        values.append(_decode_cpp_string_token(expression[pos:end]))
        pos = end
    if not values:
        return None
    return "".join(values), pos


def _cpp_string(expression: str) -> str | None:
    parsed = _cpp_string_prefix(expression)
    if parsed is None:
        return None
    value, pos = parsed
    if expression[pos:].strip():
        return None
    return value


def _normalise_cpp_type(type_name: str) -> str:
    type_name = re.sub(r"\b(?:static|inline|constexpr|extern)\b", "", type_name)
    type_name = re.sub(r"\s+", " ", type_name).strip()
    type_name = re.sub(r"\s*([<>,*&])\s*", r"\1", type_name)
    return type_name or "unknown"


def _strip_cpp_default(parameter: str) -> str:
    parts = _split_cpp_arguments(parameter.replace("=", ",", 1))
    return parts[0].strip()


def _cpp_parameter_type(parameter: str) -> str:
    parameter = _strip_cpp_default(parameter).strip()
    if parameter == "...":
        return parameter

    match = re.search(r"\b[A-Za-z_]\w*\s*(?:\[[^]]*\])?\s*$", parameter)
    if not match:
        return _normalise_cpp_type(parameter)

    candidate = parameter[: match.start()].strip()
    if not candidate:
        return _normalise_cpp_type(parameter)
    return _normalise_cpp_type(candidate)


def _find_cpp_function_signature(
    source: str, masked_source: str, function_name: str
) -> tuple[tuple[str, ...], str] | None:
    pattern = re.compile(
        rf"(?m)^[ \t]*(?P<return>[A-Za-z_][\w:<>, \t*&]*?)\s+{re.escape(function_name)}\s*\("
    )
    declarations: list[tuple[bool, tuple[str, ...], str]] = []
    for match in pattern.finditer(masked_source):
        opening = masked_source.find("(", match.start())
        closing = _matching_delimiter(masked_source, opening, "(", ")")
        tail = masked_source[closing + 1 : closing + 80].lstrip()
        is_definition = tail.startswith("{") or bool(
            re.match(r"(?:const\s*)?(?:noexcept\s*)?\{", tail)
        )
        if not is_definition and not tail.startswith(";"):
            continue

        parameter_text = source[opening + 1 : closing].strip()
        if not parameter_text or parameter_text == "void":
            parameter_types: tuple[str, ...] = ()
        else:
            parameter_types = tuple(
                _cpp_parameter_type(parameter)
                for parameter in _split_cpp_arguments(parameter_text)
            )
        return_type = _normalise_cpp_type(match.group("return"))
        declarations.append((is_definition, parameter_types, return_type))

    if not declarations:
        return None
    declarations.sort(key=lambda declaration: declaration[0], reverse=True)
    _, parameter_types, return_type = declarations[0]
    return parameter_types, return_type


def _format_cpp_default(expression: str) -> str:
    string_value = _cpp_string(expression)
    if string_value is not None:
        return json.dumps(string_value)
    return re.sub(r"\s+", " ", expression).strip()


def _cpp_named_argument(expression: str) -> Argument | None:
    helper_match = re.match(r"(?:(?:commander|co)::)?arg\s*\(", expression)
    if helper_match:
        opening = expression.find("(", helper_match.start())
        closing = _matching_delimiter(expression, opening, "(", ")")
        if expression[closing + 1 :].strip():
            raise ExtractionError(
                f"Unsupported Commander argument metadata: {expression}"
            )
        helper_arguments = _split_cpp_arguments(expression[opening + 1 : closing])
        if len(helper_arguments) not in (2, 3):
            raise ExtractionError(
                f"Commander arg() expects a name, description, and optional default: {expression}"
            )
        name = _cpp_string(helper_arguments[0])
        description = _cpp_string(helper_arguments[1])
        if name is None or description is None:
            raise ExtractionError(
                f"Commander arg() name and description must be string literals: {expression}"
            )
        default = (
            _format_cpp_default(helper_arguments[2])
            if len(helper_arguments) == 3
            else None
        )
        return Argument(name=name, default=default, description=description)

    parsed = _cpp_string_prefix(expression)
    if parsed is None:
        return None
    name, pos = parsed
    remainder = expression[pos:].strip()
    if not remainder.startswith("_arg"):
        return None
    remainder = remainder[len("_arg") :].strip()
    if not remainder:
        return Argument(name=name)
    if not remainder.startswith("="):
        raise ExtractionError(f"Unsupported Commander argument metadata: {expression}")
    return Argument(name=name, default=_format_cpp_default(remainder[1:].strip()))


def _callable_name(expression: str) -> str | None:
    expression = expression.strip().lstrip("&")
    match = re.fullmatch(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)", expression)
    return match.group(1) if match else None


def _extract_commander_commands(
    source_path: Path, signature_sources: tuple[Path, ...]
) -> tuple[Command, ...]:
    source = source_path.read_text(encoding="utf-8")
    masked_source = _mask_cpp_comments(source)
    supplemental_sources = [
        (path.read_text(encoding="utf-8"), path) for path in signature_sources
    ]
    registration_pattern = re.compile(r"\bm\s*\.\s*def\s*\(")
    commands: list[Command] = []

    for match in registration_pattern.finditer(masked_source):
        opening = masked_source.find("(", match.start())
        closing = _matching_delimiter(masked_source, opening, "(", ")")
        raw_arguments = _split_cpp_arguments(masked_source[opening + 1 : closing])
        if len(raw_arguments) < 2:
            raise ExtractionError(
                f"Invalid m.def registration at {source_path}:{source.count(chr(10), 0, match.start()) + 1}"
            )

        name = _cpp_string(raw_arguments[0])
        if name is None:
            raise ExtractionError(
                f"Non-literal command name in {source_path}:{source.count(chr(10), 0, match.start()) + 1}"
            )
        description = _cpp_string(raw_arguments[2]) if len(raw_arguments) >= 3 else ""
        if description is None:
            raise ExtractionError(
                f"Non-literal description for {name!r} in {source_path}"
            )

        named_arguments = [
            argument
            for expression in raw_arguments[3:]
            if (argument := _cpp_named_argument(expression)) is not None
        ]
        if len(named_arguments) != len(raw_arguments[3:]):
            raise ExtractionError(
                f"Unsupported argument metadata for {name!r} in {source_path}"
            )

        callable_name = _callable_name(raw_arguments[1])
        signature = (
            _find_cpp_function_signature(source, masked_source, callable_name)
            if callable_name
            else None
        )
        if callable_name and signature is None:
            for supplemental_source, _ in supplemental_sources:
                signature = _find_cpp_function_signature(
                    supplemental_source,
                    _mask_cpp_comments(supplemental_source),
                    callable_name,
                )
                if signature is not None:
                    break
        if callable_name and signature is None and not named_arguments:
            raise ExtractionError(
                f"Could not find the signature of {callable_name!r}, registered as {name!r} in {source_path}"
            )

        parameter_types, return_type = signature or (
            ("unknown",) * len(named_arguments),
            "unknown",
        )
        if named_arguments and len(named_arguments) != len(parameter_types):
            raise ExtractionError(
                f"Argument count mismatch for {name!r}: metadata has {len(named_arguments)}, "
                f"callable has {len(parameter_types)}"
            )

        if named_arguments:
            command_arguments = tuple(
                Argument(
                    argument.name,
                    type_name,
                    argument.default,
                    argument.description,
                )
                for argument, type_name in zip(named_arguments, parameter_types)
            )
        else:
            command_arguments = tuple(
                Argument(f"arg_{index}", type_name)
                for index, type_name in enumerate(parameter_types)
            )

        commands.append(
            Command(
                name=name,
                description=description.strip(),
                arguments=command_arguments,
                return_type=return_type,
                source_line=source.count("\n", 0, match.start()) + 1,
            )
        )

    if not commands:
        raise ExtractionError(f"No Commander registrations found in {source_path}")

    all_commands = [*commands, *COMMANDER_BUILT_INS]
    _validate_unique_names(source_path, all_commands)
    return tuple(sorted(all_commands, key=lambda command: command.name.casefold()))


def _literal_keyword(call: ast.Call, keyword_name: str) -> str:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError) as error:
                raise ExtractionError(
                    f"MDS Command.{keyword_name} must be a string literal at line {call.lineno}"
                ) from error
            if not isinstance(value, str):
                raise ExtractionError(
                    f"MDS Command.{keyword_name} must be a string at line {call.lineno}"
                )
            return value
    raise ExtractionError(
        f"MDS Command is missing {keyword_name!r} at line {call.lineno}"
    )


def _optional_literal_keyword(call: ast.Call, keyword_name: str, default: str) -> str:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError) as error:
                raise ExtractionError(
                    f"MDS Command.{keyword_name} must be a string literal at line {call.lineno}"
                ) from error
            if not isinstance(value, str):
                raise ExtractionError(
                    f"MDS Command.{keyword_name} must be a string at line {call.lineno}"
                )
            return value
    return default


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    raise ExtractionError(
        f"MDS Command is missing {keyword_name!r} at line {call.lineno}"
    )


def _mds_arguments(
    call: ast.Call, info: str, format_string: str
) -> tuple[Argument, ...]:
    usage = info.partition(" - ")[0]
    usage_names = re.findall(r"\{([^}:]+)(?::[^}]*)?\}", usage)
    format_specs = [
        format_spec
        for _, field_name, format_spec, _ in string.Formatter().parse(format_string)
        if field_name is not None
    ]
    if len(usage_names) != len(format_specs):
        raise ExtractionError(
            f"MDS usage and format string disagree at line {call.lineno}: "
            f"{len(usage_names)} named arguments versus {len(format_specs)} fields"
        )

    arguments_node = _keyword_value(call, "arguments")
    if not isinstance(arguments_node, (ast.Tuple, ast.List)):
        raise ExtractionError(
            f"MDS Command.arguments must be a tuple or list at line {call.lineno}"
        )

    arguments: list[Argument] = []
    for argument_node in arguments_node.elts:
        if (
            not isinstance(argument_node, ast.Call)
            or not isinstance(argument_node.func, ast.Name)
            or argument_node.func.id != "CommandArgument"
            or argument_node.keywords
            or len(argument_node.args) != 3
        ):
            raise ExtractionError(
                f"Invalid MDS CommandArgument at line {argument_node.lineno}"
            )
        try:
            name, type_name, description = (
                ast.literal_eval(value) for value in argument_node.args
            )
        except (ValueError, SyntaxError) as error:
            raise ExtractionError(
                f"MDS CommandArgument values must be string literals at line {argument_node.lineno}"
            ) from error
        if not all(isinstance(value, str) for value in (name, type_name, description)):
            raise ExtractionError(
                f"MDS CommandArgument values must be strings at line {argument_node.lineno}"
            )
        arguments.append(
            Argument(name=name, type_name=type_name, description=description)
        )

    metadata_names = [argument.name for argument in arguments]
    if metadata_names != usage_names:
        raise ExtractionError(
            f"MDS argument metadata disagrees with usage at line {call.lineno}: "
            f"{metadata_names} versus {usage_names}"
        )
    return tuple(arguments)


def _extract_mds_commands(source_path: Path) -> tuple[Command, ...]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    command_dicts: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "commands"
            for target in node.targets
        ):
            command_dicts.append(node.value)

    if not command_dicts:
        raise ExtractionError(
            f"Could not find the MDS commands dictionary in {source_path}"
        )
    command_dict = max(command_dicts, key=lambda node: len(node.keys))
    commands: list[Command] = []

    for key_node, value_node in zip(command_dict.keys, command_dict.values):
        if key_node is None:
            raise ExtractionError(
                f"Dictionary unpacking is not supported in {source_path}"
            )
        try:
            name = ast.literal_eval(key_node)
        except (ValueError, SyntaxError) as error:
            raise ExtractionError(
                f"MDS command names must be string literals at line {key_node.lineno}"
            ) from error
        if not isinstance(name, str) or not isinstance(value_node, ast.Call):
            raise ExtractionError(
                f"Unsupported MDS command entry at line {key_node.lineno}"
            )

        info = _literal_keyword(value_node, "info")
        format_string = _literal_keyword(value_node, "format_str")
        output = _literal_keyword(value_node, "output")
        output_type = _optional_literal_keyword(value_node, "output_type", "str")
        usage, separator, description = info.partition(" - ")
        commands.append(
            Command(
                name=name,
                description=description.strip() if separator else info.strip(),
                arguments=_mds_arguments(value_node, info, format_string),
                return_type=output_type,
                source_line=key_node.lineno,
                usage=usage.strip(),
                output_description=output,
            )
        )

    _validate_unique_names(source_path, commands)
    return tuple(sorted(commands, key=lambda command: command.name.casefold()))


def _validate_unique_names(source_path: Path, commands: list[Command]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for command in commands:
        if command.name in seen:
            duplicates.add(command.name)
        seen.add(command.name)
    if duplicates:
        duplicate_names = ", ".join(sorted(duplicates))
        raise ExtractionError(f"Duplicate commands in {source_path}: {duplicate_names}")


def _command_usage(command: Command, protocol: str) -> str:
    if command.usage is not None:
        return command.usage
    if not command.arguments:
        return command.name

    arguments = []
    for argument in command.arguments:
        label = argument.name
        if argument.type_name != "unknown":
            label += f": {argument.type_name}"
        if argument.default is not None:
            label += f" = {argument.default}"
        arguments.append(label)
    if protocol == "Commander":
        return f"{command.name} [{', '.join(arguments)}]"
    return f"{command.name} {' '.join(arguments)}"


def _markdown_cell(value: str) -> str:
    value = value.replace("|", "\\|").replace("\r\n", "\n")
    return value.replace("\n", "<br>")


def _source_link(output_path: Path, source_path: Path) -> str:
    relative = Path(os.path.relpath(source_path, output_path.parent))
    return relative.as_posix()


def _render_command_details(command: Command, protocol: str) -> list[str]:
    description = re.sub(r"\s+", " ", command.description).strip()
    lines = [
        f"#### `{command.name}`",
        "",
        description or "No description provided.",
        "",
        f"**Invocation:** `{_command_usage(command, protocol)}`",
        "",
        "**Arguments**",
        "",
    ]

    if command.arguments:
        lines.extend(
            [
                "| Name | Type | Default | Description |",
                "| --- | --- | --- | --- |",
            ]
        )
        for argument in command.arguments:
            default = (
                "Required"
                if argument.default is None
                else f"`{_markdown_cell(argument.default)}`"
            )
            argument_description = _markdown_cell(argument.description) or "—"
            lines.append(
                f"| `{argument.name}` | `{_markdown_cell(argument.type_name)}` "
                f"| {default} | {argument_description} |"
            )
    else:
        lines.append("_No arguments._")

    output_description = _markdown_cell(command.output_description) or "—"
    lines.extend(
        [
            "",
            "**Output**",
            "",
            "| Type | Description |",
            "| --- | --- |",
            f"| `{_markdown_cell(command.return_type)}` | {output_description} |",
            "",
        ]
    )
    return lines


def _render_server(server: Server, output_path: Path) -> list[str]:
    lines = [f"## {server.name}", ""]
    if server.source is None:
        lines.extend(
            [
                "Baldr commands are intentionally excluded for now.",
                "",
                "_No commands documented._",
                "",
            ]
        )
        return lines

    source_link = _source_link(output_path, server.source)
    lines.extend(
        [
            f"Source: [`{server.source.as_posix()}`]({source_link})",
            "",
            f"Protocol: {server.protocol}. Commands documented: {len(server.commands)}.",
            "",
        ]
    )
    if server.protocol == "Commander":
        lines.extend(
            [
                "Arguments are shown in the positional JSON array accepted by Commander. "
                "Names such as `arg_0` match Commander's generated argument names.",
                "",
            ]
        )

    lines.extend(
        [
            "### Quick reference",
            "",
            "| Command | Invocation | Description |",
            "| --- | --- | --- |",
        ]
    )
    for command in server.commands:
        usage = _markdown_cell(_command_usage(command, server.protocol))
        description = _markdown_cell(command.description or "No description provided.")
        lines.append(f"| `{command.name}` | `{usage}` | {description} |")

    lines.extend(["", "### Command details", ""])
    for command in server.commands:
        lines.extend(_render_command_details(command, server.protocol))
    return lines


def _render_document(servers: tuple[Server, ...], output_path: Path) -> str:
    lines = [
        "# Asgard DCS Command Reference",
        "",
        "<!-- Generated by asgard_guis/generate_command_reference.py. Do not edit manually. -->",
        "",
        "This document is generated offline from the server command registrations.",
        "",
        "Regenerate it from the Asgard base folder with:",
        "",
        "```bash",
        "python asgard_guis/generate_command_reference.py",
        "```",
        "",
    ]
    for server in servers:
        lines.extend(_render_server(server, output_path))
    return "\n".join(lines).rstrip() + "\n"


def _load_servers() -> tuple[Server, ...]:
    missing = [path for _, path, _ in COMMANDER_SOURCES if not path.is_file()]
    missing.extend(
        path
        for _, _, signature_sources in COMMANDER_SOURCES
        for path in signature_sources
        if not path.is_file()
    )
    if not MDS_SOURCE.is_file():
        missing.append(MDS_SOURCE)
    if missing:
        missing_paths = "\n".join(f"  - {path}" for path in missing)
        raise ExtractionError(
            "Required source files were not found. Run this script from the Asgard base folder:\n"
            f"{missing_paths}"
        )

    servers = [
        Server(
            name=name,
            source=source_path,
            protocol="Commander",
            commands=_extract_commander_commands(source_path, signature_sources),
        )
        for name, source_path, signature_sources in COMMANDER_SOURCES
    ]
    servers.append(
        Server(
            name="Multi-device server",
            source=MDS_SOURCE,
            protocol="MDS custom text protocol",
            commands=_extract_mds_commands(MDS_SOURCE),
        )
    )
    servers.append(
        Server(
            name="Baldr",
            source=None,
            protocol="Not documented",
            commands=(),
        )
    )
    return tuple(servers)


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
        description="Generate the offline Asgard DCS command reference."
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
    try:
        servers = _load_servers()
        document = _render_document(servers, args.output)
    except (ExtractionError, OSError, SyntaxError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != document
        ):
            print(f"out of date: {args.output}", file=sys.stderr)
            return 1
        print(f"up to date: {args.output}")
        return 0

    changed = _write_if_changed(args.output, document)
    state = "updated" if changed else "up to date"
    count = sum(len(server.commands) for server in servers)
    print(f"{state}: {args.output} ({count} commands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
