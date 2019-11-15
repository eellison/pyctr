from torch.jit.frontend import SourceContext

# Sourcemap - unused
ctx = SourceContext(
    """def dense(w, x):
    ret = torch.add(w, x)
    return ret
""",
    "benchmark.py",
    0,
    0,
    False,
)
dmmy_rng = ctx.make_range(1, 19, 21)
