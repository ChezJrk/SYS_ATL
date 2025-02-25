from exo.core.memory import Memory, DRAM, StaticMemory, MemGenError, generate_offset


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


# ----------- DRAM using custom malloc ----------------


class MDRAM(DRAM):
    @classmethod
    def global_(cls):
        return '#include "custom_malloc.h"'

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        return (
            f"{prim_type} *{new_name} = "
            f"({prim_type}*) malloc_dram ({size_str} * sizeof({prim_type}));"
        )

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""

        return f"free_dram({new_name});"


# ----------- DRAM using static memory ----------------


class DRAM_STATIC(DRAM):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Error checking only
        for extent in shape:
            try:
                int(extent)
            except ValueError as e:
                raise MemGenError(
                    f"DRAM_STATIC requires constant shapes. Saw: {extent}"
                ) from e

        return f'static {prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""


# ----------- DRAM using stack memory ----------------
# This is necessary for parallelization, since the stack is thread-local whereas
# static is per-binary.


class DRAM_STACK(DRAM):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        for extent in shape:
            try:
                int(extent)
            except ValueError as e:
                raise MemGenError(
                    f"DRAM_STATIC requires constant shapes. Saw: {extent}"
                ) from e

        return f'{prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""


# ----------- GEMMINI scratchpad ----------------


class GEMM_SCRATCH(Memory):
    @classmethod
    def global_(cls):
        return '#include <include/gemmini.h>\n#include "gemm_malloc.h"'

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(
                f"{srcinfo}: "
                "Cannot allocate GEMMINI Scratchpad Memory "
                "unless innermost dimension is exactly 16.  "
                f"got {shape[-1]}"
            )
        return (
            f"{prim_type} *{new_name} = "
            f"({prim_type}*) ((uint64_t)gemm_malloc ({size_str} * sizeof({prim_type})));"
        )

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""

        return f"gemm_free((uint64_t)({new_name}));"

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        # assume that strides[-1] == 1
        #    and that strides[-2] == 16 (if there is a strides[-2])
        assert len(indices) == len(strides) and len(strides) >= 2
        prim_type = basetyp.basetype().ctype()
        offset = generate_offset(indices, strides)
        return (
            f"*({prim_type}*)((uint64_t)( "
            f"((uint32_t)((uint64_t){baseptr})) + "
            f"({offset})/16))"
        )


# ----------- GEMMINI accumulator scratchpad ----------------


class GEMM_ACCUM(Memory):
    @classmethod
    def global_(cls):
        return '#include <include/gemmini.h>\n#include "gemm_acc_malloc.h"'

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(
                f"{srcinfo}: "
                "Cannot allocate GEMMINI Accumulator Memory "
                "unless innermost dimension is exactly 16.  "
                f"got {shape[-1]}"
            )
        return (
            f"{prim_type} *{new_name} = "
            f"({prim_type}*) ((uint32_t)gemm_acc_malloc ({size_str} * sizeof({prim_type})));"
        )

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""
        return f"gemm_acc_free((uint32_t)({new_name}));"

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        # assume that strides[-1] == 1
        #    and that strides[-2] == 16 (if there is a strides[-2])
        assert len(indices) == len(strides) and len(strides) >= 2
        prim_type = basetyp.basetype().ctype()
        offset = generate_offset(indices, strides)
        return (
            f"*({prim_type}*)((uint64_t)( "
            f"((uint32_t)((uint64_t){baseptr})) + "
            f"({offset})/16))"
        )


# ----------- AVX2 registers ----------------


class AVX2(Memory):
    _vec_types = {
        "float": (8, "__m256"),
        "double": (4, "__m256d"),
        "uint16_t": (16, "__m256i"),
    }

    @classmethod
    def global_(cls):
        return "#include <immintrin.h>"

    @classmethod
    def window_definition(cls, ctx):
        if ctx.n_dims() != 1:
            raise MemGenError(
                f"{ctx.srcinfo()}: Only support windows to a single AVX vector (n_dims 1)"
            )
        _, c_vec = cls._vec_types[ctx.ctype()]
        return ctx.generate_default("AVX2", c_vec)

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        vec_types = cls._vec_types

        if not shape:
            raise MemGenError(f"{srcinfo}: AVX2 vectors are not scalar values")

        if not prim_type in vec_types.keys():
            raise MemGenError(
                f"{srcinfo}: AVX2 vectors must be f32/f64/ui16 (for now), got {prim_type}"
            )

        reg_width, C_reg_type_name = vec_types[prim_type]
        if not _is_const_size(shape[-1], reg_width):
            raise MemGenError(
                f"{srcinfo}: AVX2 vectors of type {prim_type} must be {reg_width}-wide, got {shape}"
            )
        shape = shape[:-1]
        if shape:
            result = f'{C_reg_type_name} {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"{C_reg_type_name} {new_name};"
        return result

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        if basetyp.is_win():
            return f"*{baseptr}.data"
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"


# ----------- AVX-512 registers ----------------


class AVX512(Memory):
    @classmethod
    def global_(cls):
        return "#include <immintrin.h>"

    @classmethod
    def window_definition(cls, ctx):
        if ctx.n_dims() != 1:
            raise MemGenError(
                f"{ctx.srcinfo()}: Only support windows to a single AVX vector (n_dims 1)"
            )
        return ctx.generate_default("AVX512", "__m512")

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: AVX512 vectors are not scalar values")
        if not prim_type == "float":
            raise MemGenError(f"{srcinfo}: AVX512 vectors must be f32 (for now)")
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f"{srcinfo}: AVX512 vectors must be 16-wide")
        shape = shape[:-1]
        if shape:
            result = f'__m512 {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"__m512 {new_name};"
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        if basetyp.is_win():
            return f"*{baseptr}.data"
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"


# ----------- AMX tile! ----------------


class AMX_TILE(StaticMemory):
    NUM_AMX_TILES = 8
    StaticMemory.init_state(NUM_AMX_TILES)
    tile_dict = {}

    # TODO: have a better way of doing this rather than manually
    # calling this after each test that fails to compile.
    @classmethod
    def reset_allocations(cls):
        cls.init_state(cls.NUM_AMX_TILES)
        cls.tile_dict = {}

    @classmethod
    def global_(cls):
        return "#include <immintrin.h>"

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not (shape[0].isdecimal() and int(shape[0]) <= 16):
            raise MemGenError("Number of tile rows must be a constant and <= 16.")

        ctype_size = {
            "float": 4,
            "double": 8,
            "int8_t": 1,
            "int32_t": 4,
            "int_fast32_t": 4,
        }

        if not (shape[1].isdecimal() and int(shape[1]) * ctype_size[prim_type] <= 64):
            raise MemGenError(
                f"Number of bytes per row must be a constant and <= 64, currently trying to allocate {int(shape[1]) * ctype_size[prim_type]} bytes per row."
            )

        tile_num = cls.find_free_chunk()
        cls.mark(tile_num)
        cls.tile_dict[new_name] = tile_num
        return f"#define {new_name} {tile_num}"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        tile_num = cls.tile_dict[new_name]
        del cls.tile_dict[new_name]
        cls.unmark(tile_num)
        return f"#undef {new_name}"
