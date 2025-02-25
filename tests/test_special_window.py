from __future__ import annotations

import os
from pathlib import Path

from exo import (
    proc,
    instr,
    Procedure,
    DRAM,
    compile_procs_to_strings,
    MemWin,
    Memory,
    WindowStructCtx,
    SpecialWindow,
    SpecialWindowFromMemoryCtx,
    memwin_template,
)
from exo.libs.memories import MDRAM, MemGenError, StaticMemory, DRAM_STACK
from exo.libs.externs import *
from exo.stdlib.scheduling import *


class TestCudaGmem(Memory):
    @classmethod
    def window_definition(cls, ctx: WindowStructCtx):
        return ctx.generate_default("TestCudaGmem")


@memwin_template
def TestTensorMap(swizzle, *box):
    assert len(box) == 2
    smem_outer, smem_inner = box
    assert isinstance(smem_outer, int)
    assert isinstance(smem_inner, int)

    if swizzle == 0:
        cu_swizzle = "CU_TENSOR_MAP_SWIZZLE_NONE"
    else:
        assert swizzle in (32, 64, 128)
        cu_swizzle = f"CU_TENSOR_MAP_SWIZZLE_{swizzle}B"

    class Impl(SpecialWindow):
        @classmethod
        def global_(cls):
            return f"""\
#include <cuda.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>"""

        @classmethod
        def window_definition(cls, ctx: WindowStructCtx):
            assert ctx.type_shorthand() == "f32"
            cu_ctype_enum = "CU_TENSOR_MAP_DATA_TYPE_FLOAT32"
            sname = ctx.struct_name("CUtensorMap", cls.memwin_template_parameters)
            s_def = f"""\
struct {sname} {{
    unsigned outer_offset, inner_offset;
}};

struct {sname}_strides {{
    unsigned outer, inner;
}};

static inline CUtensorMap {sname}_encode_tensor_map(
        const void* globalAddress, // window dataptr
        struct {sname}_strides gmem_stride, // window layout
        unsigned gmem_outer, unsigned gmem_inner)
{{
    assert(gmem_stride.inner == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = {cu_swizzle};
    const uint32_t tensorRank = 2;
    const cuuint64_t globalDim[2] = {{gmem_inner, gmem_outer}};
    const cuuint64_t globalStrides[1] = {{sizeof({ctx.ctype()}) * gmem_stride.outer}};
    const cuuint32_t boxDim[2] = {{ {smem_inner}, {smem_outer} }};
    const cuuint32_t elementStrides[2] = {{1, 1}};
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            {cu_ctype_enum},
            tensorRank,
            (void*)globalAddress,
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {{
        fprintf(stderr, "cuTensorMapEncodeTiled: %i {{%u, %u}}\\n", (int)result, {smem_inner}, {smem_outer});
        assert(0);
    }}
    return tensorMap;
}}
"""
            return "CUtensorMap", s_def

        @classmethod
        def separate_dataptr(cls):
            return True

        @classmethod
        def window(cls, basetyp, in_expr, indices, strides, srcinfo):
            assert len(indices) == 2
            dataptr, in_layout = in_expr
            out_layout = f"{{ {in_layout}.outer_offset + {indices[0]}, {in_layout}.inner_offset + {indices[1]} }}"
            return dataptr, out_layout

        @classmethod
        def can_read(cls):
            return False

        @classmethod
        def source_memory_type(cls):
            return TestCudaGmem

        @classmethod
        def from_memory(cls, ctx: SpecialWindowFromMemoryCtx):
            sname = ctx.dst_struct_name()
            shape0, shape1 = ctx.shape_strs()
            clayout = f"(struct {sname}_strides){ctx.src_layout()}"
            d_def = f"{sname}_encode_tensor_map(&{ctx.src_data()}, {clayout}, {shape0}, {shape1})"

            # Offsets can be 0'd
            w_def = "{}"
            return d_def, w_def

    return Impl


def test_tensor_map():
    @proc
    def test_proc(
        tensor: f32[1024, 2048] @ TestCudaGmem,
        input_tensor_map: [f32][128, 128] @ TestTensorMap(0, 128, 128),
    ):
        basic_window = tensor[14, :]
        tensor_map_0 = tensor[:, :] @ TestTensorMap(0, 128, 128)
        tensor_map_1 = tensor_map_0[14:, :]
        tensor_map_C = tensor[14:, :] @ TestTensorMap(128, 196, 128)
        tensor_map_D = tensor_map_C[10:, 200:]

    c = test_proc.find("tensor_map_C = _")
    assert c.special_window() is TestTensorMap(128, 196, 128)
    assert c.special_window() is not TestTensorMap(0, 128, 128)

    cc, hh = compile_procs_to_strings([test_proc], "test.h")

    # This is just a placeholder test for now
    if False:
        HOME = os.environ["HOME"]
        open(f"{HOME}/junk/test.h", "w").write(hh)
        open(f"{HOME}/junk/test.c", "w").write(cc)

    # fmt: off

    # TestTensorMap(0, 128, 128) defs should have ended up in the header file.
    # There should be no const suffix.
    assert "struct exo_win_2f32_CUtensorMap_0_128_128 {" in hh
    assert "inline CUtensorMap exo_win_2f32_CUtensorMap_0_128_128_encode_tensor_map" in hh

    # test_proc definition should have separate tensormap, layout
    # inputs for input_tensor_map.
    assert "CUtensorMap exo_data_input_tensor_map, struct exo_win_2f32_CUtensorMap_0_128_128 input_tensor_map" in hh

    # TestTensorMap(128, 196, 128) defs should have ended up in the C file
    assert "struct exo_win_2f32_CUtensorMap_128_196_128 {" in cc
    assert "inline CUtensorMap exo_win_2f32_CUtensorMap_128_196_128_encode_tensor_map" in cc

    # Expected window code
    assert "struct exo_win_1f32c_TestCudaGmem basic_window = (struct exo_win_1f32c_TestCudaGmem){ &tensor[(14) * (2048)], { 1 } };" in cc
    assert "CUtensorMap exo_data_tensor_map_0 = exo_win_2f32_CUtensorMap_0_128_128_encode_tensor_map(&tensor[0], (struct exo_win_2f32_CUtensorMap_0_128_128_strides){ 2048, 1 }, 1024, 2048);" in cc
    assert "struct exo_win_2f32_CUtensorMap_0_128_128 tensor_map_0 = {};" in cc
    assert "CUtensorMap exo_data_tensor_map_1 = exo_data_tensor_map_0;" in cc
    assert "struct exo_win_2f32_CUtensorMap_0_128_128 tensor_map_1 = (struct exo_win_2f32_CUtensorMap_0_128_128) { tensor_map_0.outer_offset + 14, tensor_map_0.inner_offset + 0 };" in cc
    assert "CUtensorMap exo_data_tensor_map_C = exo_win_2f32_CUtensorMap_128_196_128_encode_tensor_map(&tensor[(14) * (2048)], (struct exo_win_2f32_CUtensorMap_128_196_128_strides){ 2048, 1 }, 1010, 2048);" in cc
    assert "struct exo_win_2f32_CUtensorMap_128_196_128 tensor_map_C = {};" in cc
    assert "CUtensorMap exo_data_tensor_map_D = exo_data_tensor_map_C;" in cc
    assert "struct exo_win_2f32_CUtensorMap_128_196_128 tensor_map_D = (struct exo_win_2f32_CUtensorMap_128_196_128) { tensor_map_C.outer_offset + 10, tensor_map_C.inner_offset + 200 };" in cc

    # fmt: on
