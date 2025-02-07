from __future__ import annotations

import os
import sys

from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *

# Hide output when running through exocc.
if __name__ != "__main__" and hasattr(os, "devnull"):
    sys.stdout = open(os.devnull, "w")

# Algorithm definition
@proc
def rank_k_reduce_6x16(
    K: size, A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM, C: f32[6, 16] @ DRAM
):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


print("\n============= original ==============")
print(rank_k_reduce_6x16)

print("\n============= reorder loops ==============")
neon = rename(rank_k_reduce_6x16, "rank_k_reduce_6x16_scheduled")
neon = reorder_loops(neon, "j k")
neon = reorder_loops(neon, "i k")
print(neon)


print("\n============= divide loop ==============")
# neon only supports vectors of width 4 for f32
# x86 supports either 4 or 8 wide
vec_reg_width = 4
neon = divide_loop(neon, "for j in _: _", vec_reg_width, ["jo", "ji"], perfect=True)
print(neon)

print("\n============= stage mem ==============")
# we want the computation to be "output stationary", which means,
# we want to preallocate all the output registers at the start.
# The staging of C will cause us to consume 12 out of the 16 vector registers
neon = stage_mem(neon, "for k in _:_", "C[0:6, 0:16]", "C_reg")
print(neon)
neon = simplify(neon)

print("\n============= reshape C_reg ==============")
# Reshape C_reg so we can map it into vector registers
neon = divide_dim(neon, "C_reg:_", 1, vec_reg_width)
print(neon)

print("\n============= divide loop ==============")
neon = repeat(divide_loop)(
    neon, "for i1 in _: _", vec_reg_width, ["i2", "i3"], perfect=True
)
neon = simplify(neon)
print(neon)

print("\n============= map C_reg ops ==============")
# Map C_reg operations to vector instructions
neon = set_memory(neon, "C_reg:_", Neon)
# this loads 8 items into the register but neon only loads 4
neon = replace_all(neon, neon_vld_4xf32)
neon = replace_all(neon, neon_vst_4xf32)
neon = simplify(neon)
print(neon)


# Now, the rest of the compute needs to work with the constraint that the
# we only have 4 more registers to work with here.

print("\n============= stage B_reg ==============")
# B is easy, it is just two vector loads
neon = stage_mem(neon, "for i in _:_", "B[k, 0:16]", "B_reg")
neon = simplify(neon)
print(neon)

print("\n============= block 1st B_reg load ==============")
neon = divide_loop(neon, "for i0 in _: _ #1", vec_reg_width, ["io", "ii"], perfect=True)
print(neon)

print("\n============= reshape B_reg ==============")
neon = divide_dim(neon, "B_reg:_", 0, vec_reg_width)
print(neon)

print("\n============= map B_reg ops ==============")
neon = set_memory(neon, "B_reg:_", Neon)
neon = simplify(neon)
neon = replace_all(neon, neon_vld_4xf32)
neon = simplify(neon)
print(neon)

# Now we've used up two more vector registers.
# The final part is staging A

print("\n============= stage A_reg ==============")
neon = bind_expr(neon, "A[i, k]", "A_reg")
neon = expand_dim(neon, "A_reg", vec_reg_width, "ji")
neon = lift_alloc(neon, "A_reg", n_lifts=2)
neon = fission(neon, neon.find("A_reg[ji] = _").after(), n_lifts=2)
neon = remove_loop(neon, "for jo in _: _")
neon = set_memory(neon, "A_reg:_", Neon)
neon = replace_all(neon, neon_broadcast_4xf32)
neon = simplify(neon)
print(neon)


# DO THE COMPUTE!!!
print("\n============= map mult add op ==============")
neon = replace_all(neon, neon_vfmadd_4xf32_4xf32)
neon = simplify(neon)
print(neon)

print("\n============= dnone! ==============")
