"""Helpers for invoking sync or async user callbacks."""

from __future__ import annotations

import inspect
from typing import Any, Callable


async def invoke_maybe_awaitable(callback: Callable[..., Any], *args: Any) -> None:
    """Call ``callback(*args)`` and await the result when it is awaitable.

    True ``async def`` callbacks, sync wrappers that return a coroutine or
    other awaitable, and callable objects with ``async def __call__`` are
    all awaited inline. Plain sync callbacks that return ``None`` (or any
    non-awaitable) complete without scheduling.
    """
    result = callback(*args)
    if inspect.isawaitable(result):
        await result
