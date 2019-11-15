from torch._C._jit_tree_views import (
    BinOp,
    Const,
    Expr,
    FalseLiteral,
    Subscript,
    TrueLiteral,
    TupleLiteral,
)

from pyctr.examples.torchscript.dmmy import dmmy_rng
from pyctr.overloads import py_defaults


def gen_bin_op(op):
    def bin_op(this, other):
        return Rep(BinOp(op, torch_expr(this), torch_expr(other)))

    return bin_op


class Rep:
    def __init__(self, node):
        self.node = node

    def __getitem__(self, item):
        return Rep(Subscript(self.node, [torch_expr(item)]))


_operators = {"__matmul__": "@", "__add__": "+", "__sub__": "-"}


for o in _operators:
    setattr(Rep, o, gen_bin_op(_operators[o]))


def torch_expr(e):
    ret = None

    # Constants
    if isinstance(e, bool):
        ret = TrueLiteral(dmmy_rng) if ret else FalseLiteral(dmmy_rng)
    elif isinstance(e, int) or isinstance(ret, float):
        ret = Const(dmmy_rng, str(e))
    elif isinstance(e, tuple):
        ret = TupleLiteral(dmmy_rng, list(map(lambda exp: torch_expr(exp), e)))
    elif isinstance(e, Rep):
        ret = e.node
    elif isinstance(e, list):
        ret = list(map(lambda exp: torch_expr(exp), e))
    elif isinstance(e, Expr):
        ret = e
    elif isinstance(e, py_defaults.Variable):
        ret = torch_expr(e.val)
    else:
        raise TypeError(f"Impossible to translate expression of type: {type(e)}")
    return ret
