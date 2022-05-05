from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
from pyctr.api import conversion
from pyctr.examples.z3py import z3py
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import logical_ops
from pyctr.transformers.virtualization import variables
from typing import Optional
import itertools

import z3
from typing import List

def bad_assert_shape_broadcast(lhs, rhs):
    r = []
    for x, y in itertools.zip_longest(
        reversed(lhs), reversed(rhs), fillvalue=1
    ):
        assert not (x != y and x != 1 and y != 1)
        if x == 1:
            val = y
        else:
            val = x
        r.append(val)
    return tuple(reversed(r))

def convert_fn(fn):
  cnvt = conversion.convert(fn, z3py, [variables, control_flow, logical_ops])
  def wrapper(*args, **kwargs):
    solve = z3.Solver()
    z3py.solver = solve
    out = cnvt(*args, **kwargs)
    return solve, out
  return wrapper

out = convert_fn(bad_assert_shape_broadcast)
solver, output = out([5, 4], [z3.Int("SS1")])
solver.check()
model = solver.model()
print("Input generated:", model[z3.Int("SS1")])
print(output)
# Input generated: 1
# 5, 4