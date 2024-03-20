"""Provide basic blocks for patching Python code."""

import ast
import dataclasses
import itertools
import pathlib
import textwrap
from typing import (
    Optional,
    Union,
    Sequence,
    List,
    TypeVar,
    Iterable,
    Iterator,
    Tuple,
    NoReturn,
    cast,
)

import asttokens
from icontract import require, ensure


@dataclasses.dataclass
class Range:
    """
    Represent a token range.

    The AST nodes parsed with asttokens will satisfy this protocol.
    """

    first_token: asttokens.asttokens.Token
    last_token: asttokens.asttokens.Token


def cast_node_to_range(node: ast.AST) -> Range:
    """
    Assert that the node can be cast to a token range.

    The AST nodes parsed with asttokens will satisfy the :ref:`Range` protocol.
    """
    if not hasattr(node, "first_token"):
        raise ValueError(f"Expected an AST node parsed with asttokens, but got: {node}")

    assert hasattr(node, "first_token")
    assert hasattr(node, "last_token")
    # noinspection PyUnresolvedReferences
    assert isinstance(node.first_token, asttokens.asttokens.Token)
    # noinspection PyUnresolvedReferences
    assert isinstance(node.last_token, asttokens.asttokens.Token)

    return cast(Range, node)


@dataclasses.dataclass
class Patch:
    """Represent a code patch anchored at a range of tokens."""

    range: Range
    prefix: Optional[str] = None
    replacement: Optional[str] = None
    suffix: Optional[str] = None


@dataclasses.dataclass
class _InsertPrefix:
    text: str
    position: int

    @property
    def end(self) -> int:
        """Give the anchor position in the text."""
        return self.position


@dataclasses.dataclass
class _InsertSuffix:
    text: str
    position: int

    @property
    def end(self) -> int:
        """Give the anchor position in the text."""
        return self.position


@dataclasses.dataclass
class _Replace:
    text: str
    position: int
    end: int


_Action = Union[_InsertPrefix, _InsertSuffix, _Replace]


class Error:
    """Represent a parsing or a patching error with potential nested errors."""

    def __init__(
        self, message: str, underlying_errors: Optional[List["Error"]] = None
    ) -> None:
        self.message = message
        self.underlying_errors = underlying_errors

    def __str__(self) -> str:
        if self.underlying_errors is None or len(self.underlying_errors) == 0:
            return self.message

        blocks = [self.message]
        for sub_error in self.underlying_errors:
            sub_message = textwrap.indent(str(sub_error), "  ")
            sub_message = "*" + sub_message[1:]

            blocks.append(sub_message)

        return "\n".join(blocks)


def _check_replaces_do_not_overlap(actions: Sequence[_Action]) -> Optional[Error]:
    previous_replace: Optional[_Replace] = None
    for action in actions:
        if not isinstance(action, _Replace):
            continue

        if previous_replace is None:
            previous_replace = action
            continue

        if previous_replace.end >= action.position:
            return Error(
                f"The text to be replaced, {previous_replace}, "
                f"overlaps with another text to be replaced, {action}."
            )

        previous_replace = action

    return None


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    """
    Iterate pair-wise over the iterator.

    >>> list(pairwise("ABCDE"))
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def assert_never(value: NoReturn) -> NoReturn:
    """
    Signal to mypy to perform an exhaustive matching.

    Please see the following page for more details:
    https://hakibenita.com/python-mypy-exhaustive-checking
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"


@require(lambda actions: _check_replaces_do_not_overlap(actions) is None)
@require(lambda actions: actions == sorted(actions, key=lambda action: action.position))
@ensure(
    lambda result: all(
        previous_action.position < action.position
        for previous_action, action in pairwise(result)
    )
)
def _merge_actions(actions: Sequence[_Action]) -> List[_Action]:
    result: List[_Action] = []

    previous_prefix: Optional[_InsertPrefix] = None
    previous_suffix: Optional[_InsertSuffix] = None

    for action in actions:
        if isinstance(action, _InsertPrefix):
            if (
                previous_prefix is not None
                and previous_prefix.position == action.position
            ):
                previous_prefix.text = f"{action.text}{previous_prefix.text}"
            else:
                prefix_copy = dataclasses.replace(action)
                result.append(prefix_copy)
                previous_prefix = prefix_copy

        elif isinstance(action, _InsertSuffix):
            if (
                previous_suffix is not None
                and previous_suffix.position == action.position
            ):
                previous_suffix.text = f"{previous_suffix.text}{action.text}"
            else:
                suffix_copy = dataclasses.replace(action)
                result.append(suffix_copy)
                previous_suffix = suffix_copy

        elif isinstance(action, _Replace):
            result.append(dataclasses.replace(action))

        else:
            assert_never(action)

    return result


def apply_patches(patches: List[Patch], text: str) -> str:
    """Apply the patches by replacing the text correspond to a node with the new text."""
    if len(patches) == 0:
        return text

    actions: List[_Action] = []
    for patch in patches:
        if patch.prefix is not None:
            # noinspection PyUnresolvedReferences
            actions.append(
                _InsertPrefix(
                    text=patch.prefix, position=patch.range.first_token.startpos
                )
            )

        if patch.suffix is not None:
            # noinspection PyUnresolvedReferences
            actions.append(
                _InsertSuffix(text=patch.suffix, position=patch.range.last_token.endpos)
            )

        if patch.replacement is not None:
            # noinspection PyUnresolvedReferences
            actions.append(
                _Replace(
                    text=patch.replacement,
                    position=patch.range.first_token.startpos,
                    end=patch.range.last_token.endpos,
                )
            )

    actions = sorted(actions, key=lambda an_action: an_action.position)

    error = _check_replaces_do_not_overlap(actions)
    if error is not None:
        raise AssertionError(error)

    actions = _merge_actions(actions)

    parts: List[str] = []
    previous_action: Optional[_Action] = None

    for action in actions:
        if previous_action is None:
            parts.append(text[0 : action.position])
        else:
            parts.append(text[previous_action.end : action.position])
        parts.append(action.text)
        previous_action = action
    assert previous_action is not None
    parts.append(text[previous_action.end :])

    return "".join(parts)


@ensure(lambda result: (result[0] is None) ^ (result[1] is None))
def parse_file(
    path: pathlib.Path,
) -> Tuple[Optional[asttokens.ASTTokens], Optional[Error]]:
    """
    Parse the given python file and return the abstract syntax tree tokens of the module
    """
    source = path.read_text(encoding="utf-8")

    try:
        atok = asttokens.ASTTokens(source, parse=True)
    except Exception as exception:
        return None, Error(f"Failed to parse {path}: {exception}")

    assert atok is not None
    assert atok.tree is not None
    assert isinstance(atok.tree, ast.Module)

    return atok, None
