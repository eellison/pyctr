"""
Currently the state of the AST being built is kept as a global variable here.
A solution to this can be to add an additional variable passed to `overload.call`,
`overload.assign` etc. that can be used to manage this state.
"""
from contextlib import contextmanager
_ast = []


def emit_node(n):
    global _ast
    _ast.append(n)


@contextmanager
def fresh_ast():
    global _ast
    _ast_checkpoint = _ast
    _ast = []
    yield _ast
    _ast = _ast_checkpoint


def flush_ast():
    global _ast
    ret = _ast
    _ast = []
    return ret
