from collections import deque
from functools import wraps
from typing import Any, Callable, Generator, Sequence

from attrs import define, field

OperationCallback = tuple[Callable, Sequence, dict]


@define
class History:
    forward: deque["Operation"] = field(init=False, default=deque())
    backward: deque["Operation"] = field(init=False, default=deque())

    def add(self, operation: "Operation"):
        if len(self.forward) != 0:
            self.forward.clear()

        self.backward.append(operation)

    def clear(self):
        self.forward.clear()
        self.backward.clear()

    def undo_last(self):

        if not self.can_undo():
            return

        op = self.backward.pop()
        self.forward.appendleft(op)
        op.undo()

    def redo_last(self):
        if not self.can_redo():
            return

        op = self.forward.popleft()
        self.backward.append(op)
        op.do()

    def can_undo(self):
        return len(self.backward) > 0

    def can_redo(self):
        return len(self.forward) > 0


@define
class Operation:
    direct: OperationCallback
    inverse: OperationCallback

    def do(self):
        func, args, kwargs = self.direct
        func(*args, **kwargs)

    def undo(self):
        func, args, kwargs = self.inverse
        func(*args, **kwargs)


HISTORY = History()


def undoable[
    T, **P
](f: Callable[P, Generator[OperationCallback | None, Any, T]]) -> Callable[P, T]:
    @wraps(f)
    def _wrapper(*args, **kwargs):
        progress_generator = f(*args, **kwargs)

        direct = next(progress_generator)
        if direct is not None:
            inverse = next(progress_generator)
            if inverse is not None:
                HISTORY.add(Operation(direct=direct, inverse=inverse))
        try:
            next(progress_generator)
        except StopIteration as result:
            return result.value

    return _wrapper  # type:ignore
