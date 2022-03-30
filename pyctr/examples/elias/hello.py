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


import z3
from typing import List

def prove(f):
  s = z3.Solver()
  s.add(z3.Not(f))
  return s.check() == z3.unsat

def can_solve(f):
  s = z3.Solver()
  s.add(f)
  print(s)
  print(s.model())
  return s.check() == z3.sat

def demorgan(a, b):
  return (a and b) == (not (not a or not b))

def convert(f):
  return conversion.convert(f, z3py, [logical_ops])

# converted_demorgan = convert(demorgan)
# prove(converted_demorgan(True, z3.Bool('q')))

def chains(a, b, c):
  return (not (a and b and c)) == (not a or (not b) or (not c))

def test_fn(a, b, c):
  result = None
  if a:
    result = b
  else:
    result = c
  return result

def foo(hi):
  return hi + 4

def assert_val(x):
  if x == 2:
    return foo(x + 3)
  else:
    raise Exception("")


def infer_size_impl(shape: List[int], numel: int) -> List[int]:
    newsize = 1
    infer_dim = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize = newsize * shape[dim]
        else:
            raise AssertionError("invalid shape dimensions")
    if not (
        numel == newsize
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise AssertionError("invalid shape")
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out

def broadcast(a: List[int], b: List[int]):
    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes = []
    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1

        # TODO: handle exceptions ?
        assert sizeA != sizeB and sizeA != 1 and sizeB != 1
        # if sizeA != sizeB and sizeA != 1 and sizeB != 1:
        #     # TODO: only assertion error is bound in C++ compilation right now
        #     raise AssertionError(
        #         "The size of tensor a {} must match the size of tensor b ("
        #         "{}) at non-singleton dimension {}".format(sizeA, sizeB, i)
        #     )
        if sizeA == 1:
          out = sizeB
        else:
          out = sizeA
        expandedSizes.append(out)
    return expandedSizes

def fn(sizeA, sizeB):
  if sizeA != 1:
    raise Exception("")
  return sizeA
  # return sizeA, sizeB

# solve = z3.Solver()
# z3py.solver = solve
# converted_fn = conversion.convert(fn, z3py, [variables, control_flow, logical_ops])
# out = converted_fn(z3.Int("hi"), z3.Int("hello"))
# import pdb; pdb.set_trace()
# print(out, solve)


def multiply_integers(li: List[int]):
    out = 1
    for elem in li:
        out = out * elem
    return out

def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max)
    if dim < 0:
        dim += dim_post_expr
    return dim


def flatten(input, start_dim: int, end_dim: int):
    start_dim = maybe_wrap_dim(start_dim, len(input))
    end_dim = maybe_wrap_dim(end_dim, len(input))
    assert start_dim <= end_dim
    if len(input) == 0:
        return [1]
    if start_dim == end_dim:
        # TODO: return self
        return input
        # TODO: below causes bug - file why
        # out = []
        # for elem in input:
        #     out.append(elem)
        # return out
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel = slice_numel * input[i]
    # TODO: use slicing when slice optimization has landed
    # slice_numel = multiply_integers(input[start_dim:end_dim - start_dim + 1])
    shape = []
    for i in range(start_dim):
        shape.append(input[i])
    shape.append(slice_numel)
    for i in range(end_dim + 1, len(input)):
        shape.append(input[i])
    return shape


# solve = z3.Solver()
# z3py.solver = solve
# converted_fn = conversion.convert(flatten, z3py, [variables, control_flow, logical_ops])
# out = converted_fn([z3.Int("a"), z3.Int("b"), 5, z3.Int("c")], 1, 3)
# import pdb; pdb.set_trace()
# print(out)
def check_non_negative(array: List[int]) -> bool:
    # TODO: look into rewriting with early return and getting loop unrolling to fire
    non_negative = False
    for val in array:
        if val < 0:
            non_negative = True
    return non_negative

def check_shape_forward(
    input: List[int],
    weight_sizes: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    k = len(input)
    weight_dim = len(weight_sizes)
    # TODO: assertions could be expanded with the error messages
    assert not check_non_negative(padding)
    assert not check_non_negative(stride)
    assert weight_dim == k
    assert weight_sizes[0] >= groups
    assert (weight_sizes[0] % groups) == 0
    # only handling not transposed
    assert input[1] == weight_sizes[1] * groups
    assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])
    for i in range(2, k):
        assert (input[i] + 2 * padding[i - 2]) >= (
            dilation[i - 2] * (weight_sizes[i] - 1) + 1
        )
    # this is not handling transposed convolution yet


z3.ArithRef.__floordiv__ = z3.ArithRef.__div__

def conv_output_size(
    input_size: List[int],
    weight_size: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    k = len(input)
    weight_dim = len(weight_size)
    # TODO: assertions could be expanded with the error messages
    import pdb; pdb.set_trace()
    assert input[1] == weight_size[1] * groups
    assert bias is None or (len(bias) == 1 and bias[0] == weight_size[0])
    for i in range(2, k):
        assert (input[i] + 2 * padding[i - 2]) >= (
            dilation[i - 2] * (weight_size[i] - 1) + 1
        )
    has_dilation = len(dilation) > 0
    dim = len(input_size)
    output_size = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])
    output_size.append(weight_size[weight_output_channels_dim])
    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size.append(
            (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
        )
    return output_size

solve = z3.Solver()
input = [z3.Int(str(i) + "_var") for i in range(4)]
for int_ref in input:
  solve.add(int_ref >= 0)
weight = [33, 16, 3, 5]
bias = [33]
stride = [2, 1]
padding = [4, 2]
dilation = [3, 1]
groups = 1
z3py.solver = solve
converted_fn = conversion.convert(conv_output_size, z3py, [variables, control_flow, logical_ops])
out = converted_fn(input, weight, bias, stride, padding, dilation, groups)
import pdb; pdb.set_trace()
solve.check()
model = solve.model()
print([model[input[i]] for i in range(len(input))])
print(out)

# print(converted_fn([z3.Int("hi")], [z3.Int("hello2"), 1]))

# import pdb; pdb.set_trace()
# a = converted_fn(z3.Int("hi"))
# print(a)
# import pdb; pdb.set_trace()
# print(type(a))
# print(can_solve(a))
# # can_solve(converted_fn(z3.Int('hi')))
