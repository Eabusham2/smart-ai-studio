"""Invertible geometric and palette transformations for ARC grid examples."""
import json


def grid(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if (not isinstance(value, list) or not value or not isinstance(value[0], list) or not value[0]
            or any(not isinstance(row, list) or len(row) != len(value[0])
                   or any(type(color) is not int or not 0 <= color <= 9 for color in row)
                   for row in value)):
        return None
    return value


def grid_text(value):
    return json.dumps(value, separators=(",", ":"))


def transform(value, variant, *, inverse=False):
    """Apply one of eight square symmetries and a cyclic relabeling fixing color 0."""
    turns, flip, shift = variant
    result = value
    if inverse:
        turns, shift = -turns, -shift
    if flip and not inverse:
        result = [row[::-1] for row in result]
    for _ in range(turns % 4):
        result = [list(column) for column in zip(*result[::-1])]
    if flip and inverse:
        result = [row[::-1] for row in result]
    return [[1 + (cell - 1 + shift) % 9 if cell else 0 for cell in row] for row in result]


# Interleave palette and geometry changes for small voting budgets.
VARIANTS = [(turns, flip, shift) for flip in (False, True)
            for turns in range(4) for shift in (0, 1)]
