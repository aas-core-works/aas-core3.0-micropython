"""Patch the original aas-core-python SDK so that it can run on Micropython."""

import argparse
import ast
import io
import itertools
import os
import pathlib
import re
import sys
import tokenize
from typing import Optional, Tuple, List, Sequence, Union

import asttokens
from icontract import ensure

# NOTE (empwilli):
# We need to include the repository root in the python search path since newer
# versions of Python (such as 3.11 and 3.12) exclude ``dev_scripts/`` from it
# -- they rely on setup.py excluding them in ``find_package``:
#
# ``packages=find_packages(exclude=["dev_scripts", ...]),``
#
# . This means that the ``dev_scripts`` module will not be on the Python path,
# as newer versions of setuptools only put packages explicitly found by
# ``find_packages``.

sys.path.append(str(pathlib.Path(os.path.realpath(__file__)).parent.parent))

from dev_scripts import patching


def check_source_file_exists_and_panic_if_not(path: pathlib.Path) -> None:
    """Check that the file exists.

    Otherwise, print an error to STDERR and ``sys.exit`` with 1.
    """
    if path.exists():
        return

    print(f"The source file does not exist: {path}", file=sys.stderr)
    sys.exit(1)


def _find_keyword_arg(call: ast.Call, name: str) -> Optional[ast.keyword]:
    """Find the keyword argument in the given call."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword

    return None


def patch_setup_py(
    source_repo_dir: pathlib.Path, target_repo_dir: pathlib.Path, package_name: str
) -> Optional[patching.Error]:
    """
    Patch the ``setup.py`` for Micropython.

    The package name designates the package name in PyPI.
    """
    src_pth = source_repo_dir / "setup.py"
    tgt_pth = target_repo_dir / "setup.py"

    atok, error = patching.parse_file(src_pth)
    if error is not None:
        return error

    setup_call = None  # type: Optional[ast.Call]
    for body_stmt in atok.tree.body:
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and isinstance(body_stmt.value.func, ast.Name)
            and body_stmt.value.func.id == "setup"
        ):
            setup_call = body_stmt.value

    if setup_call is None:
        return patching.Error(
            f"Failed to patch {src_pth}",
            underlying_errors=[patching.Error("No `setup(.)` call found")],
        )

    patches = []  # type: List[patching.Patch]
    setup_call_errors = []  # type: List[patching.Error]

    keyword = _find_keyword_arg(call=setup_call, name="name")
    if keyword is None:
        setup_call_errors.append(patching.Error("No `name` keyword argument found"))
    else:
        patches.append(
            patching.Patch(
                patching.cast_node_to_range(keyword.value),
                replacement=repr(package_name),
            )
        )

    keyword = _find_keyword_arg(call=setup_call, name="description")
    if keyword is None:
        setup_call_errors.append(
            patching.Error("No `description` keyword argument found")
        )
    else:
        patches.append(
            patching.Patch(
                patching.cast_node_to_range(keyword.value),
                replacement=(
                    '"Manipulate and de/serialize '
                    'Asset Administration Shells in Micropython."'
                ),
            )
        )

    keyword = _find_keyword_arg(call=setup_call, name="url")
    if keyword is None:
        setup_call_errors.append(patching.Error("No `url` keyword argument found"))
    else:
        patches.append(
            patching.Patch(
                patching.cast_node_to_range(keyword.value),
                replacement=(
                    '"https://github.com/aas-core-works/aas-core3.0-micropython"'
                ),
            )
        )

    keyword = _find_keyword_arg(call=setup_call, name="classifiers")
    if keyword is None:
        setup_call_errors.append(
            patching.Error("No `classifiers` keyword argument found")
        )
    else:
        patches.append(
            patching.Patch(
                patching.cast_node_to_range(keyword.value),
                replacement=(
                    """[
"Programming Language :: Python :: Implementation :: MicroPython",
"Development Status :: 5 - Production/Stable",
"License :: OSI Approved :: MIT License",
]"""
                ),
            )
        )

    if len(setup_call_errors) > 0:
        return patching.Error(
            f"Failed to patch {src_pth}",
            underlying_errors=[
                patching.Error(
                    "Unexpected `setup(.)` call", underlying_errors=setup_call_errors
                )
            ],
        )

    new_text = patching.apply_patches(patches=patches, text=atok.text)
    tgt_pth.write_text(new_text, encoding="utf-8")
    return None


def _find_docstring(body: Sequence[ast.AST]) -> Optional[ast.Constant]:
    """Find the docstring as the first statement in the body."""
    if len(body) == 0:
        return None

    stmt = body[0]
    if not isinstance(stmt, ast.Expr):
        return None

    if not isinstance(stmt.value, ast.Constant):
        return None

    if not isinstance(stmt.value.value, str):
        return None

    return stmt.value


def _find_variable_assign(body: Sequence[ast.AST], name: str) -> Optional[ast.Assign]:
    """Find the assignment in the ``body`` to the variable called ``name``."""
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue

        if len(stmt.targets) != 1:
            continue

        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue

        if target.id == name:
            return stmt

    return None


def patch_init_py(
    source_repo_dir: pathlib.Path, target_repo_dir: pathlib.Path, module_name: str
) -> Optional[patching.Error]:
    """
    Patch ``__init__.py`` in the SDK.

    The ``module_name`` designates the main SDK module, *e.g.*, ``aas_core3``.
    """
    src_pth = source_repo_dir / module_name / "__init__.py"
    tgt_pth = target_repo_dir / module_name / "__init__.py"

    atok, error = patching.parse_file(src_pth)
    if error is not None:
        return error

    patches = []  # type: List[patching.Patch]
    errors = []  # type: List[patching.Error]

    docstring = _find_docstring(atok.tree.body)
    if docstring is None:
        errors.append(patching.Error("No module docstring found"))
    else:
        patches.append(
            patching.Patch(
                patching.cast_node_to_range(docstring),
                replacement=(
                    '"""Manipulate and de/serialize '
                    'Asset Administration Shells in Micropython."""'
                ),
            )
        )

    if len(errors) > 0:
        return patching.Error(f"Failed to patch: {src_pth}", underlying_errors=errors)

    tgt_pth.parent.mkdir(exist_ok=True)

    new_text = patching.apply_patches(patches, atok.text)
    tgt_pth.write_text(new_text, encoding="utf-8")

    return None


def _patch_function_def_to_stub(
    function_def: ast.FunctionDef, atok: asttokens.ASTTokens
) -> List[patching.Patch]:
    """Determine the patches required to map the given function definition to a stub."""
    patches = []  # type: List[patching.Patch]

    assert (
        len(function_def.body) >= 1
    ), f"Unexpected FunctionDef with empty body: {ast.dump(function_def)}"

    first_stmt = function_def.body[0]
    first_body_stmt_index = 0
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        first_body_stmt_index = 1

    if first_body_stmt_index < len(function_def.body):
        first_body_stmt = function_def.body[first_body_stmt_index]
        first_body_stmt_as_range = patching.cast_node_to_range(first_body_stmt)

        function_def_as_range = patching.cast_node_to_range(function_def)

        patches.append(
            patching.Patch(
                patching.Range(
                    first_token=first_body_stmt_as_range.first_token,
                    last_token=function_def_as_range.last_token,
                ),
                replacement="...",
            )
        )

    return patches


def _patch_class_def_to_stub(
    class_def: ast.ClassDef, atok: asttokens.ASTTokens
) -> List[patching.Patch]:
    """Determine the patches needed to create a stub for the given class definition."""
    patches = []  # type: List[patching.Patch]

    for stmt in class_def.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                value_as_range = patching.cast_node_to_range(stmt.value)
                patches.append(
                    patching.Patch(
                        patching.Range(
                            first_token=atok.tokens[
                                value_as_range.first_token.index - 1
                            ],
                            last_token=value_as_range.last_token,
                        ),
                        replacement="",
                    )
                )
        elif isinstance(stmt, ast.FunctionDef):
            patches.extend(_patch_function_def_to_stub(stmt, atok=atok))
        elif isinstance(stmt, ast.Assign):
            continue
        else:
            raise NotImplementedError(
                f"Unhandled node in the body "
                f"of the class definition {class_def.name!r}: {ast.dump(stmt)}"
            )

    return patches


_TYPE_COMMENT_RE = re.compile(r"^#\s*type\s*:")


def _strip_type_comments(text: str) -> str:
    """Strip all the ``# type: `` comments."""
    stream = io.StringIO(text)

    accepted_tokens = []  # type: List[tokenize.Token]

    for token in tokenize.generate_tokens(stream.readline):
        if token.type is tokenize.COMMENT:
            if _TYPE_COMMENT_RE.match(token.string) is not None:
                continue

        accepted_tokens.append(token)

    return tokenize.untokenize(accepted_tokens)


def _map_module_to_stub(
    module: ast.Module, atok: asttokens.ASTTokens
) -> Tuple[Optional[str], Optional[List[patching.Error]]]:
    """Generate the stub text for the given module."""
    patches = []  # type: List[patching.Patch]
    errors = []  # type: List[patching.Error]

    for stmt in module.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue

        elif isinstance(stmt, (ast.ImportFrom, ast.Import)):
            continue

        elif isinstance(stmt, ast.FunctionDef):
            patches.extend(_patch_function_def_to_stub(stmt, atok=atok))

        elif isinstance(stmt, ast.AnnAssign):
            value_as_range = patching.cast_node_to_range(stmt.value)
            patches.append(
                patching.Patch(
                    patching.Range(
                        # NOTE (mristin, 2024-04-05):
                        # We need to also delete the ``:`` before the annotation.
                        first_token=atok.tokens[value_as_range.first_token.index - 1],
                        last_token=value_as_range.last_token,
                    ),
                    replacement="",
                )
            )

        elif isinstance(stmt, ast.Assign):
            continue

        elif isinstance(stmt, ast.If):
            continue

        elif isinstance(stmt, ast.ClassDef):
            patches.extend(_patch_class_def_to_stub(class_def=stmt, atok=atok))
        elif isinstance(stmt, ast.Assert):
            # NOTE (mristin, 2024-04-05):
            # Assertions are meaningless in the stubs as mypy will not interpret them.
            # There is some discussion on stub-tests, but we can assume here that
            # our typing is correct in the original code.
            #
            # See, for example:
            # https://stackoverflow.com/questions/51716200/how-do-you-check-if-a-typeshed-stub-pyi-file-matches-the-implementation
            patches.append(
                patching.Patch(patching.cast_node_to_range(stmt), replacement="")
            )
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            # NOTE (mristin, 2024-04-05):
            # The calls are to be executed, and add no typing information, so we simply
            # exclude the calls from the stubs.
            patches.append(
                patching.Patch(patching.cast_node_to_range(stmt), replacement="")
            )
        else:
            errors.append(patching.Error(f"Unhandled AST node: {ast.dump(stmt)}"))

    if len(errors) > 0:
        return None, errors

    new_text = patching.apply_patches(patches, text=atok.text)

    # NOTE (mristin, 2024-04-03):
    # We need to strip ``# type: ...` comments as the last token in the bodies
    # does not include the comments.
    new_text = _strip_type_comments(new_text)

    return new_text, None


def _replace_docstring_if_any(
    body: Sequence[Union[ast.AST, ast.expr, ast.stmt]]
) -> List[patching.Patch]:
    """Replace the docstring with an empty string, if there is any docstring."""
    if len(body) == 0:
        return []

    first_stmt = body[0]

    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        return [patching.Patch(patching.cast_node_to_range(first_stmt), replacement="")]

    return []


def _patch_function_def_to_implementation(
    function_def: ast.FunctionDef, atok: asttokens.ASTTokens
) -> List[patching.Patch]:
    """Strip the function definition of any type annotations."""
    patches = []  # type: List[patching.Patch]

    assert (
        len(function_def.args.posonlyargs) == 0
    ), "No positional-only arguments expected"

    assert len(function_def.args.kwonlyargs) == 0, "No keyword-only arguments expected"

    for decorator in function_def.decorator_list:
        if (
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "abc"
        ):
            decorator_as_range = patching.cast_node_to_range(decorator)

            patches.append(
                patching.Patch(
                    range=patching.Range(
                        # NOTE (mristin, 2024-04-05):
                        # We have to include ``@`` in the deletion as well.
                        first_token=atok.tokens[
                            decorator_as_range.first_token.index - 1
                        ],
                        last_token=decorator_as_range.last_token,
                    ),
                    replacement="",
                )
            )

    for arg in itertools.chain(
        function_def.args.posonlyargs,
        function_def.args.args,
        function_def.args.kwonlyargs,
    ):
        if arg.annotation is not None:
            patches.append(
                patching.Patch(patching.cast_node_to_range(arg), replacement=arg.arg)
            )

    if function_def.returns is not None:
        # NOTE (mristin, 2024-04-03):
        # We go one token back to also include the ``->`` token.

        returns_as_range = patching.cast_node_to_range(function_def.returns)

        prev_token = atok.tokens[returns_as_range.first_token.index - 1]

        patches.append(
            patching.Patch(
                patching.Range(prev_token, returns_as_range.last_token), replacement=""
            )
        )

    patches.extend(_replace_docstring_if_any(function_def.body))

    for stmt in function_def.body:
        if isinstance(stmt, ast.AnnAssign):
            annotation_as_range = patching.cast_node_to_range(stmt.annotation)
            patches.append(
                patching.Patch(
                    patching.Range(
                        first_token=atok.tokens[
                            annotation_as_range.first_token.index - 1
                        ],
                        last_token=annotation_as_range.last_token,
                    ),
                    replacement="",
                )
            )

    return patches


def _patch_class_def_to_implementation(
    class_def: ast.ClassDef, atok: asttokens.ASTTokens
) -> List[patching.Patch]:
    """Strip the class definition of any type annotations."""
    patches = []  # type: List[patching.Patch]

    for base in class_def.bases:
        if (
            isinstance(base, ast.Subscript)
            and isinstance(base.value, ast.Name)
            and base.value.id == "Generic"
        ) or (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "abc"
        ):
            patches.append(
                patching.Patch(range=patching.cast_node_to_range(base), replacement="")
            )
        elif (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "enum"
        ):
            # NOTE (mristin, 2024-04-5):
            # We decorate with our own enum. decorator as we need to mimic the protocol
            # of ``enum.Enum`` in jsonization.
            patches.append(
                patching.Patch(range=patching.cast_node_to_range(base), replacement="")
            )
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(class_def),
                    prefix="@aas_enum.decorator\n",
                )
            )
        elif isinstance(base, ast.Subscript):
            # NOTE (mristin, 2024-04-05):
            # We assume here that subscripts in the base classes imply generic
            # parameters, so we simply remove them.
            value_text = atok.get_text(base.value)
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(base), replacement=value_text
                )
            )

    for stmt in class_def.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            if len(class_def.body) == 1:
                # NOTE (mristin, 2024-04-05):
                # We need to replace the docstrings with ``pass`` so that we
                # do not invalidate the classes with an empty body.
                patches.append(
                    patching.Patch(
                        range=patching.cast_node_to_range(stmt), replacement="pass"
                    )
                )
            else:
                patches.append(
                    patching.Patch(
                        range=patching.cast_node_to_range(stmt), replacement=""
                    )
                )
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is None:
            patches.append(
                patching.Patch(range=patching.cast_node_to_range(stmt), replacement="")
            )
        elif isinstance(stmt, ast.FunctionDef):
            patches.extend(
                _patch_function_def_to_implementation(function_def=stmt, atok=atok)
            )
        elif isinstance(stmt, ast.Assign):
            continue
        else:
            raise NotImplementedError(
                f"Unhandled AST statement "
                f"in the class {class_def.name!r}: {ast.dump(stmt)}"
            )

    return patches


_CONDITIONAL_IMPORT_OF_TYPING_FINAL = """\
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final"""

_CONDITIONAL_DEFINITION_OF_PATH_LIKE_ON_TYPE_CHECKING = """\
if TYPE_CHECKING:
    PathLike = os.PathLike[Any]
else:
    PathLike = os.PathLike"""

_CONDITIONAL_IMPORT_OF_FINAL_AND_PROTOCOL = """\
if sys.version_info >= (3, 8):
    from typing import Final, Protocol
else:
    from typing_extensions import Final, Protocol"""


def _is_conditional_header_for_typing(
    node: Union[ast.AST], atok: asttokens.ASTTokens
) -> bool:
    """Check if the node is an ``if``-statement importing ``typing.Final``."""
    if not isinstance(node, ast.If):
        return False

    # NOTE (mristin, 2024-04-03):
    # This is not a very robust test, but will probably work for a long time.

    # noinspection PyTypeChecker
    text = atok.get_text(node)

    return text in (
        _CONDITIONAL_IMPORT_OF_TYPING_FINAL,
        _CONDITIONAL_DEFINITION_OF_PATH_LIKE_ON_TYPE_CHECKING,
        _CONDITIONAL_IMPORT_OF_FINAL_AND_PROTOCOL,
    )


def _strip_comments(text: str) -> str:
    """Strip all the comments."""
    stream = io.StringIO(text)

    accepted_tokens = []  # type: List[tokenize.Token]

    for token in tokenize.generate_tokens(stream.readline):
        if token.type is tokenize.COMMENT:
            continue

        accepted_tokens.append(token)

    return tokenize.untokenize(accepted_tokens)


class _StringAndJoinedStrCollector(ast.NodeVisitor):
    """Collect all the string constants and joined strings."""

    # pylint: disable=missing-docstring

    def __init__(self) -> None:
        self.string_constants = []  # type: List[ast.Constant]
        self.joined_strings = []  # type: List[ast.JoinedStr]

    def visit_Constant(self, node):
        assert isinstance(node, ast.Constant)
        if isinstance(node.value, str):
            self.string_constants.append(node)

    def visit_JoinedStr(self, node):
        assert isinstance(node, ast.JoinedStr)
        self.joined_strings.append(node)


class _RendererInFormattedValue(ast.NodeVisitor):
    """Render the formatted values in a joined string."""

    # NOTE (mristin):
    # This class is needed since atok does not hold the strings corresponding to
    # the nodes of values in joined strings. Hence, we have to render them ourselves.

    # pylint: disable=missing-function-docstring

    def __init__(self) -> None:
        self._writer = io.StringIO()

    def get_text(self) -> str:
        return self._writer.getvalue()

    def visit_JoinedStr(self, node):
        raise ValueError("Unexpected JoinedStr in a FormattedValue")

    def visit_Name(self, node):
        assert isinstance(node, ast.Name)
        self._writer.write(node.id)

    def visit_Constant(self, node):
        return self._writer.write(repr(node))

    def visit_Attribute(self, node):
        assert isinstance(node, ast.Attribute)

        needs_parentheses = not isinstance(
            node.value, (ast.Call, ast.Attribute, ast.Name)
        )

        if needs_parentheses:
            self._writer.write("(")

        self.visit(node.value)

        if needs_parentheses:
            self._writer.write(")")

        self._writer.write(f".{node.attr}")

    def visit_Call(self, node):
        assert isinstance(node, ast.Call)

        needs_parentheses = not isinstance(
            node.func, (ast.Call, ast.Attribute, ast.Name)
        )

        if needs_parentheses:
            self._writer.write("(")

        self.visit(node.func)
        self._writer.write("(")

        for i, arg in enumerate(node.args):
            if i > 0:
                self._writer.write(", ")

            self.visit(arg)

        self._writer.write(")")

        if needs_parentheses:
            self._writer.write(")")

    def generic_visit(self, node):
        raise NotImplementedError(
            f"Unhandled node in a FormattedValue: {ast.dump(node)}"
        )


def _join_joined_strs(node: ast.JoinedStr, atok: asttokens.ASTTokens) -> str:
    """Render the joined string in a single literal."""
    parts = ['f"']  # type: List[str]
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(
                value.value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\t", "\\t")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\b", "\\b")
                .replace("\f", "\\f")
                .replace("{", "{{")
                .replace("}", "}}")
            )
        elif isinstance(value, ast.FormattedValue):
            renderer = _RendererInFormattedValue()
            renderer.visit(value.value)
            parts.append("{")
            parts.append(renderer.get_text())
            parts.append("}")
        else:
            raise NotImplementedError(
                f"Unhandled value in a joined str: {ast.dump(value)}"
            )

    parts.append('"')

    parts_joined = "".join(parts)

    return parts_joined


def _join_strings(node: ast.Constant) -> str:
    """Render the string literal on a single line."""
    return repr(node.value)


def _join_strings_and_joined_strings(text: str) -> str:
    """
    Join the strings since Micropython does not support consecutive literals.

    >>> _join_strings_and_joined_strings('"testme"')
    '"testme"'

    >>> _join_strings_and_joined_strings('"test" "me"')
    "'testme'"

    >>> _join_strings_and_joined_strings('f"test{x}me"')
    'f"test{x}me"'

    >>> _join_strings_and_joined_strings('f"test{x}" f"{y}me"')
    'f"test{x}{y}me"'

    >>> _join_strings_and_joined_strings('f"test{x.y}" f"me"')
    'f"test{x.y}me"'

    >>> _join_strings_and_joined_strings('f"test{x()}" f"me"')
    'f"test{x()}me"'

    >>> _join_strings_and_joined_strings('f"test{type(variable).__name__}" f"me"')
    'f"test{type(variable).__name__}me"'
    """
    atok = asttokens.ASTTokens(text, parse=True)
    assert isinstance(atok.tree, ast.Module)

    collector = _StringAndJoinedStrCollector()
    collector.visit(atok.tree)

    nodes = sorted(
        collector.string_constants + collector.joined_strings,
        key=lambda node: node.first_token.index,
    )

    patches = []  # type: List[patching.Patch]

    for node in nodes:
        if node.first_token.index != node.last_token.index:
            if isinstance(node, ast.JoinedStr):
                joined = _join_joined_strs(node=node, atok=atok)
            else:
                joined = _join_strings(node=node)

            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(node), replacement=joined
                )
            )

    new_text = patching.apply_patches(patches, text)

    return new_text


class _ASTCollectionsAbcCollector(ast.NodeVisitor):
    """Collect all the ``ast.Attribute``'s referring to ``collections.abc``."""

    # pylint: disable=missing-function-docstring

    def __init__(self) -> None:
        self.collection_attributes = []  # type: List[ast.Attribute]

    def visit_Attribute(self, node):
        assert isinstance(node, ast.Attribute)

        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "collections"
            and node.value.attr == "abc"
        ):
            self.collection_attributes.append(node)


def _replace_collections_abc_with_native_types(text: str) -> str:
    """Replace ``collections.abc.*`` with native types."""
    atok = asttokens.ASTTokens(text, parse=True)
    assert isinstance(atok.tree, ast.Module)

    collector = _ASTCollectionsAbcCollector()
    collector.visit(atok.tree)

    patches = []  # type: List[patching.Patch]

    for node in collector.collection_attributes:
        if node.attr == "Mapping":
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(node), replacement="dict"
                )
            )
        elif node.attr == "Iterable":
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(node), replacement="list"
                )
            )
        elif node.attr == "Set":
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(node), replacement="set"
                )
            )
        else:
            raise NotImplementedError(f"Unhandled collections.abc: {ast.dump(node)}")

    return patching.apply_patches(patches, text)


class _ASTBase64Collector(ast.NodeVisitor):
    """Collect all the ``ast.Attribute``'s related to ``base64`` module."""

    # pylint: disable=missing-function-docstring

    def __init__(self) -> None:
        self.attributes = []  # type: List[ast.Attribute]

    def visit_Attribute(self, node):
        assert isinstance(node, ast.Attribute)

        if isinstance(node.value, ast.Name) and node.value.id == "base64":
            self.attributes.append(node)

        self.visit(node.value)
        self.visit(node.ctx)


def _replace_base64_with_binascii(text: str) -> str:
    """
    Replace the encoding/decoding of Base64 using ``binascii`` module.

    See: https://github.com/micropython/micropython/issues/3862
    """
    atok = asttokens.ASTTokens(text, parse=True)
    assert isinstance(atok.tree, ast.Module)

    collector = _ASTBase64Collector()
    collector.visit(atok.tree)

    patches = []  # type: List[patching.Patch]

    for node in collector.attributes:
        if node.attr == "b64decode":
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(node),
                    replacement="binascii.a2b_base64",
                )
            )
        elif node.attr == "b64encode":
            patches.append(
                patching.Patch(
                    range=patching.cast_node_to_range(node),
                    replacement="binascii.b2a_base64",
                )
            )
        else:
            raise NotImplementedError(
                f"Unhandled base64-related statement: {ast.dump(node)}"
            )

    return patching.apply_patches(patches, text)


def _map_module_to_implementation(
    module: ast.Module, atok: asttokens.ASTTokens
) -> Tuple[Optional[str], Optional[List[patching.Error]]]:
    """Generate the stub text for the given module."""
    patches = []  # type: List[patching.Patch]
    errors = []  # type: List[patching.Error]

    for stmt in module.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            patches.append(
                patching.Patch(patching.cast_node_to_range(stmt), replacement="")
            )

        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module == "typing":
                patches.append(
                    patching.Patch(patching.cast_node_to_range(stmt), replacement="")
                )

        elif isinstance(stmt, ast.Import):
            if len(stmt.names) == 1 and stmt.names[0].name in (
                "collections.abc",
                "abc",
            ):
                patches.append(
                    patching.Patch(patching.cast_node_to_range(stmt), replacement="")
                )
            elif len(stmt.names) == 1 and stmt.names[0].name == "base64":
                # NOTE (mristin, 2024-04-5):
                # We have to replace base64 with binascii for Micropython.
                patches.append(
                    patching.Patch(
                        patching.cast_node_to_range(stmt), replacement="import binascii"
                    )
                )
            elif len(stmt.names) == 1 and stmt.names[0].name == "enum":
                # NOTE (mristin, 2024-04-5):
                # We replace ``enum.Enum`` with our decorator.
                patches.append(
                    patching.Patch(
                        patching.cast_node_to_range(stmt),
                        replacement="from aas_core3 import enum as aas_enum",
                    )
                )

        elif isinstance(stmt, ast.FunctionDef):
            # NOTE (mristin, 2024-04-5):
            # We exclude the assertions encapsulated in functions since they are
            # not critical for the implementation.
            #
            # We rely on that they are checked in the original code, but do not matter
            # for the Micropython implementation.
            if stmt.name.startswith("_assert_"):
                patches.append(
                    patching.Patch(
                        range=patching.cast_node_to_range(stmt), replacement=""
                    )
                )
            else:
                patches.extend(
                    _patch_function_def_to_implementation(function_def=stmt, atok=atok)
                )
        elif isinstance(stmt, ast.AnnAssign):
            annotation_as_range = patching.cast_node_to_range(stmt.annotation)
            patches.append(
                patching.Patch(
                    patching.Range(
                        first_token=atok.tokens[
                            annotation_as_range.first_token.index - 1
                        ],
                        last_token=annotation_as_range.last_token,
                    ),
                    replacement="",
                )
            )

        elif isinstance(stmt, ast.Assign):
            if (
                isinstance(stmt.value, ast.Subscript)
                and isinstance(stmt.value.value, ast.Name)
                and stmt.value.value.id == "Union"
            ) or (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "TypeVar"
            ):
                patches.append(
                    patching.Patch(
                        range=patching.cast_node_to_range(stmt), replacement=""
                    )
                )
            else:
                pass

        elif _is_conditional_header_for_typing(node=stmt, atok=atok):
            patches.append(
                patching.Patch(range=patching.cast_node_to_range(stmt), replacement="")
            )

        elif isinstance(stmt, ast.ClassDef):
            patches.extend(
                _patch_class_def_to_implementation(class_def=stmt, atok=atok)
            )

        elif isinstance(stmt, ast.Assert):
            # NOTE (mristin, 2024-04-5):
            # We exclude the assertions since they are not critical for the
            # implementation.
            #
            # We rely on that they are checked in the original code, but do not matter
            # for the Micropython implementation.
            patches.append(
                patching.Patch(range=patching.cast_node_to_range(stmt), replacement="")
            )

        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            # NOTE (mristin, 2024-04-5):
            # We exclude the assertions encapsulated in functions since they are
            # not critical for the implementation.
            #
            # We rely on that they are checked in the original code, but do not matter
            # for the Micropython implementation.
            if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id.startswith(
                "_assert_"
            ):
                patches.append(
                    patching.Patch(
                        range=patching.cast_node_to_range(stmt), replacement=""
                    )
                )
        else:
            errors.append(patching.Error(f"Unhandled AST node: {ast.dump(stmt)}"))

    if len(errors) > 0:
        return None, errors

    new_text = patching.apply_patches(patches, text=atok.text)

    # NOTE (mristin, 2024-04-03):
    # We remove all the comments as we want to optimize the loading time.
    new_text = _strip_comments(new_text)

    new_text = _join_strings_and_joined_strings(new_text)
    new_text = _replace_collections_abc_with_native_types(new_text)
    new_text = _replace_base64_with_binascii(new_text)

    return new_text, None


def _produce_stub_file(
    atok: asttokens.ASTTokens, target_path: pathlib.Path
) -> Optional[List[patching.Error]]:
    """Generate the stub file for mypy."""
    text, errors = _map_module_to_stub(module=atok.tree, atok=atok)
    if errors is not None:
        return errors

    assert text is not None
    target_path.write_text(text, encoding="utf-8")
    return None


def _produce_implementation_file(
    atok: asttokens.ASTTokens, target_path: pathlib.Path
) -> Optional[List[patching.Error]]:
    """Generate the implementation file stripped of any type annotations."""
    text, errors = _map_module_to_implementation(module=atok.tree, atok=atok)
    if errors is not None:
        return errors

    assert text is not None
    target_path.write_text(text, encoding="utf-8")
    return None


def _patch_to_stub_and_implementation_file(
    source_path: pathlib.Path,
    target_stub_path: pathlib.Path,
    target_implementation_path: pathlib.Path,
) -> Optional[patching.Error]:
    """Patch the file and split it into a stub and the implementation file."""
    errors = []  # type: List[patching.Error]

    atok, error = patching.parse_file(source_path)
    if error is not None:
        return error

    stub_errors = _produce_stub_file(atok=atok, target_path=target_stub_path)
    if stub_errors is not None:
        errors.append(
            patching.Error("Failed to produce the stub", underlying_errors=stub_errors)
        )

    impl_errors = _produce_implementation_file(
        atok=atok, target_path=target_implementation_path
    )
    if impl_errors is not None:
        errors.append(
            patching.Error(
                "Failed to produce the implementation", underlying_errors=impl_errors
            )
        )

    if len(errors) > 0:
        return patching.Error(
            f"Failed to patch: {source_path}", underlying_errors=errors
        )

    return None


def _remove_global_level_comments(text: str) -> str:
    """Strip all the comments at the global level, *i.e.*, at no indention."""
    stream = io.StringIO(text)

    accepted_tokens = []  # type: List[tokenize.Token]

    prev_token = None
    for token in tokenize.generate_tokens(stream.readline):
        if not (
            token.type is tokenize.COMMENT
            and (
                prev_token is None
                or prev_token.type is tokenize.NL
                or prev_token.type is tokenize.NEWLINE
            )
        ):
            accepted_tokens.append(token)

        prev_token = token

    return tokenize.untokenize(accepted_tokens)


@ensure(lambda result: (result[0] is not None) ^ (result[1] is not None))
def _remove_xml_deserialization(
    atok: asttokens.ASTTokens,
) -> Tuple[Optional[str], Optional[patching.Error]]:
    """Remove all the elements from the module related to the XML de-serialization."""
    patches = []  # type: List[patching.Patch]

    docstring = _find_docstring(atok.tree.body)
    if docstring is None:
        return None, patching.Error("Expected to find a docstring, but found none")

    patches.append(
        patching.Patch(
            range=patching.cast_node_to_range(docstring),
            replacement='"""Serialize AAS models to XML."""',
        )
    )

    class_set_to_ignore = {
        "Element",
        "HasIterparse",
        "ElementSegment",
        "IndexSegment",
        "Segment",
        "Path",
        "DeserializationException",
    }

    function_set_to_ignore = {
        "_with_elements_cleared_after_yield",
        "_parse_element_tag",
        "_raise_if_has_tail_or_attrib",
        "_read_end_element",
        "_read_text_from_element",
        "_read_bool_from_element_text",
        "_read_int_from_element_text",
        "_read_float_from_element_text",
        "_read_str_from_element_text",
        "_read_bytes_from_element_text",
    }

    variable_set_to_ignore = {
        "_XS_BOOLEAN_LITERAL_SET",
        "_NAMESPACE_IN_CURLY_BRACKETS",
        "Segment",
        "_TEXT_TO_XS_DOUBLE_LITERALS",
    }

    for stmt in atok.tree.body:
        ignore = False
        if isinstance(stmt, ast.ClassDef):
            if stmt.name in class_set_to_ignore:
                ignore = True
            elif stmt.name.startswith("_ReaderAndSetterFor"):
                ignore = True
            else:
                pass
        elif isinstance(stmt, ast.FunctionDef):
            if stmt.name in function_set_to_ignore:
                ignore = True
            elif (
                stmt.name.endswith("_from_iterparse")
                or stmt.name.endswith("_from_stream")
                or stmt.name.endswith("_from_file")
                or stmt.name.endswith("_from_str")
                or re.match(r"^_read_.*_as_element$", stmt.name) is not None
                or re.match(r"^_read_.*_as_sequence$", stmt.name) is not None
                or re.match(r"^_read_.*_from_element_text$", stmt.name) is not None
            ):
                ignore = True
            else:
                pass
        elif isinstance(stmt, (ast.AnnAssign, ast.Assign)):
            target = None  # type: Optional[ast.AST]

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
            elif isinstance(stmt, ast.AnnAssign):
                target = stmt.target
            else:
                pass

            if (
                target is not None
                and isinstance(target, ast.Name)
                and (
                    target.id in variable_set_to_ignore
                    or target.id.startswith("_DISPATCH_FOR_")
                    or target.id.startswith("_READ_AND_SET_DISPATCH_FOR_")
                )
            ):
                ignore = True

        elif isinstance(stmt, ast.Import):
            if len(stmt.names) == 1:
                import_name = stmt.names[0]
                assert isinstance(import_name, ast.alias), (
                    f"Expected import names to be alias, "
                    f"but got something else in the statement: {ast.dump(stmt)}"
                )

                if import_name.name == "xml.etree.ElementTree":
                    ignore = True
        else:
            pass

        if ignore:
            patches.append(
                patching.Patch(range=patching.cast_node_to_range(stmt), replacement="")
            )

    new_text = patching.apply_patches(patches, atok.text)

    # NOTE (mristin, 2024-04-12):
    # We reformat only a slight bit so that the result is easier for debugging.
    new_text = re.sub(r"\n\n\n+", "\n\n", new_text)

    # NOTE (mristin, 2024-04-12):
    # We remove the comments at the global level as it is very cumbersome to tie them
    # to the definitions which we are removing. Hence, we leave the indented comments,
    # but make a blanket removal of all the global comments even if they would have been
    # informative.
    new_text = _remove_global_level_comments(new_text)

    return new_text, None


def _patch_xml_serialization(
    source_path: pathlib.Path,
    target_stub_path: pathlib.Path,
    target_implementation_path: pathlib.Path,
) -> Optional[patching.Error]:
    """Patch only the XML serialization as Micropython does not include XML library."""
    atok, error = patching.parse_file(source_path)
    if error is not None:
        return error

    assert atok is not None
    text_wo_deserialization, error = _remove_xml_deserialization(atok=atok)
    if error is not None:
        return patching.Error(
            f"Failed to remove the XML de-serialization from: {source_path}",
            underlying_errors=[error],
        )

    try:
        atok_wo_deserialization = asttokens.ASTTokens(
            text_wo_deserialization, parse=True
        )
    except Exception as exception:
        return patching.Error(
            f"Failed to parse the code where the XML de-serialization "
            f"is removed: {exception} from: {source_path}"
        )

    assert atok_wo_deserialization is not None

    errors = []  # type: List[patching.Error]

    stub_errors = _produce_stub_file(
        atok=atok_wo_deserialization, target_path=target_stub_path
    )
    if stub_errors is not None:
        errors.append(
            patching.Error("Failed to produce the stub", underlying_errors=stub_errors)
        )

    impl_errors = _produce_implementation_file(
        atok=atok_wo_deserialization, target_path=target_implementation_path
    )
    if impl_errors is not None:
        errors.append(
            patching.Error(
                "Failed to produce the implementation", underlying_errors=impl_errors
            )
        )

    if len(errors) > 0:
        return patching.Error(
            f"Failed to patch: {source_path}", underlying_errors=errors
        )

    return None


def main() -> int:
    """Execute the main routine."""
    this_path = pathlib.Path(os.path.realpath(__file__))
    this_dir = this_path.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_repo_dir",
        help="Path to the repository containing aas-core-python SDK",
        default=str(this_dir.parent.parent / "aas-core3.0-python"),
    )
    parser.add_argument(
        "--target_repo_dir",
        help="Path to the repository containing aas-core-micropython SDK",
        default=str(this_dir.parent),
    )
    args = parser.parse_args()

    source_repo_dir = pathlib.Path(args.source_repo_dir)
    target_repo_dir = pathlib.Path(args.target_repo_dir)

    if not source_repo_dir.exists():
        print(
            f"The --source_repo_dir does not exist: {source_repo_dir}", file=sys.stderr
        )
        sys.exit(1)

    if not source_repo_dir.is_dir():
        print(
            f"The --source_repo_dir is not a directory: {source_repo_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not target_repo_dir.exists():
        print(
            f"The --target_repo_dir does not exist: {target_repo_dir}", file=sys.stderr
        )
        sys.exit(1)

    if not target_repo_dir.is_dir():
        print(
            f"The --target_repo_dir is not a directory: {target_repo_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"""\
Patching from:
  {source_repo_dir}
to:
  {target_repo_dir}"""
    )

    errors = []  # type: List[patching.Error]

    error = patch_setup_py(
        source_repo_dir=source_repo_dir,
        target_repo_dir=target_repo_dir,
        package_name="aas-core3.0-micropython",
    )
    if error is not None:
        errors.append(error)

    module_name = "aas_core3"

    error = patch_init_py(
        source_repo_dir=source_repo_dir,
        target_repo_dir=target_repo_dir,
        module_name=module_name,
    )
    if error is not None:
        errors.append(error)

    for submodule in ["common", "constants", "jsonization", "stringification", "types"]:
        error = _patch_to_stub_and_implementation_file(
            source_path=source_repo_dir / module_name / f"{submodule}.py",
            target_stub_path=target_repo_dir / module_name / f"{submodule}.pyi",
            target_implementation_path=target_repo_dir
            / module_name
            / f"{submodule}.py",
        )
        if error is not None:
            errors.append(error)

    error = _patch_xml_serialization(
        source_path=source_repo_dir / module_name / "xmlization.py",
        target_stub_path=target_repo_dir / module_name / "xmlization.pyi",
        target_implementation_path=target_repo_dir / module_name / "xmlization.py",
    )
    if error is not None:
        errors.append(error)

    if len(errors) > 0:
        print(
            patching.Error(
                f"Failed to patch {source_repo_dir}", underlying_errors=errors
            ),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
