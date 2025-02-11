import functools
import re
import textwrap
import warnings
from collections import ChainMap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..core.LoopIR import LoopIR, LoopIR_Do, get_writes_of_stmts, T, CIR
from ..core.configs import ConfigError
from .mem_analysis import MemoryAnalysis
from ..core.memory import MemGenError, Memory, DRAM, StaticMemory
from .parallel_analysis import ParallelAnalysis
from .prec_analysis import PrecisionAnalysis
from ..core.prelude import *
from .win_analysis import WindowAnalysis
from ..rewrite.range_analysis import IndexRangeEnvironment

from ..spork.async_config import BaseAsyncConfig
from ..spork.base_with_context import (
    BaseWithContext,
    is_if_holding_with,
    ExtWithContext,
)
from ..spork.collectives import SpecializeCollective
from ..spork.loop_modes import LoopMode, Seq, Par, _CodegenPar
from ..spork.spork_env import SporkEnv, KernelArgsScanner
from ..spork import actor_kinds

# XXX used for backdoor
from ..spork.cuda_memory import CudaRmem


def sanitize_str(s):
    return re.sub(r"\W", "_", s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

CacheDict = lambda: defaultdict(CacheDict)

op_prec = {
    "or": 10,
    #
    "and": 20,
    #
    "==": 30,
    #
    "<": 40,
    ">": 40,
    "<=": 40,
    ">=": 40,
    #
    "+": 50,
    "-": 50,
    #
    "*": 60,
    "/": 60,
    "%": 60,
    # unary minus
    "~": 70,
}


def lift_to_cir(e, range_env):
    assert e.type.is_indexable(), "why are you here?"

    is_non_neg = lambda e: range_env.check_expr_bound(0, IndexRangeEnvironment.leq, e)

    if isinstance(e, LoopIR.Read):
        return CIR.Read(e.name, is_non_neg(e))
    elif isinstance(e, LoopIR.Const):
        return CIR.Const(e.val)
    elif isinstance(e, LoopIR.BinOp):
        lhs = lift_to_cir(e.lhs, range_env)
        rhs = lift_to_cir(e.rhs, range_env)
        return CIR.BinOp(e.op, lhs, rhs, is_non_neg(e))
    elif isinstance(e, LoopIR.USub):
        arg = lift_to_cir(e.arg, range_env)
        return CIR.USub(arg, is_non_neg(e))
    else:
        assert False, "bad case!"


operations = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "%": lambda x, y: x % y,
}


def simplify_cir(e):
    if isinstance(e, (CIR.Read, CIR.Const, CIR.Stride)):
        return e

    elif isinstance(e, CIR.BinOp):
        lhs = simplify_cir(e.lhs)
        rhs = simplify_cir(e.rhs)

        if isinstance(lhs, CIR.Const) and isinstance(rhs, CIR.Const):
            return CIR.Const(operations[e.op](lhs.val, rhs.val))

        if isinstance(lhs, CIR.Const) and lhs.val == 0:
            if e.op == "+":
                return rhs
            elif e.op == "*" or e.op == "/":
                return CIR.Const(0)
            elif e.op == "-":
                pass  # cannot simplify
            else:
                assert False

        if isinstance(rhs, CIR.Const) and rhs.val == 0:
            if e.op == "+" or e.op == "-":
                return lhs
            elif e.op == "*":
                return CIR.Const(0)
            elif e.op == "/":
                assert False, "division by zero??"
            else:
                assert False, "bad case"

        if isinstance(lhs, CIR.Const) and lhs.val == 1 and e.op == "*":
            return rhs

        if isinstance(rhs, CIR.Const) and rhs.val == 1 and (e.op == "*" or e.op == "/"):
            return lhs

        return CIR.BinOp(e.op, lhs, rhs, e.is_non_neg)
    elif isinstance(e, CIR.USub):
        arg = simplify_cir(e.arg)
        if isinstance(arg, CIR.USub):
            return arg.arg
        if isinstance(arg, CIR.Const):
            return arg.update(val=-(arg.val))
        return e.update(arg=arg)
    else:
        assert False, "bad case!"


class LoopIR_SubProcs(LoopIR_Do):
    def __init__(self, proc):
        self._subprocs = set()
        if proc.instr is None:
            super().__init__(proc)

    def result(self):
        return self._subprocs

    # to improve efficiency
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Call):
            self._subprocs.add(s.f)
        else:
            super().do_s(s)


def find_all_subprocs(proc_list):
    all_procs = []
    seen = set()

    def walk(proc, visited):
        if proc in seen:
            return

        all_procs.append(proc)
        seen.add(proc)

        for sp in LoopIR_SubProcs(proc).result():
            if sp in visited:
                raise ValueError(f"found call cycle involving {sp.name}")
            walk(sp, visited | {proc})

    for proc in proc_list:
        walk(proc, set())

    # Reverse for C declaration order.
    return list(reversed(all_procs))


class LoopIR_FindMems(LoopIR_Do):
    def __init__(self, proc):
        self._mems = set()
        for a in proc.args:
            if a.mem:
                self._mems.add(a.mem)
        super().__init__(proc)

    def result(self):
        return self._mems

    # to improve efficiency
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.mem:
                self._mems.add(s.mem)
        else:
            super().do_s(s)

    def do_t(self, t):
        pass


class LoopIR_FindExterns(LoopIR_Do):
    def __init__(self, proc):
        self._externs = set()
        super().__init__(proc)

    def result(self):
        return self._externs

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.Extern):
            self._externs.add((e.f, e.type.basetype().ctype()))
        else:
            super().do_e(e)

    def do_t(self, t):
        pass


class LoopIR_FindConfigs(LoopIR_Do):
    def __init__(self, proc):
        self._configs = set()
        super().__init__(proc)

    def result(self):
        return self._configs

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.ReadConfig):
            self._configs.add(e.config)
        else:
            super().do_e(e)

    def do_s(self, s):
        if isinstance(s, LoopIR.WriteConfig):
            self._configs.add(s.config)
        super().do_s(s)

    def do_t(self, t):
        pass


def find_all_mems(proc_list):
    mems = set()
    for p in proc_list:
        mems.update(LoopIR_FindMems(p).result())

    return [m for m in mems]


def find_all_externs(proc_list):
    externs = set()
    for p in proc_list:
        externs.update(LoopIR_FindExterns(p).result())

    return externs


def find_all_configs(proc_list):
    configs = set()
    for p in proc_list:
        configs.update(LoopIR_FindConfigs(p).result())

    return list(configs)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowStruct:
    name: str
    definition: str


@functools.cache
def _window_struct(typename, ctype, n_dims, is_const) -> WindowStruct:
    const_kwd = "const " if is_const else ""
    const_suffix = "c" if is_const else ""

    sname = f"exo_win_{n_dims}{typename}{const_suffix}"
    sdef = (
        f"struct {sname}{{\n"
        f"    {const_kwd}{ctype} * const data;\n"
        f"    const int_fast32_t strides[{n_dims}];\n"
        f"}};"
    )

    sdef_guard = sname.upper()
    sdef = f"""#ifndef {sdef_guard}
#define {sdef_guard}
{sdef}
#endif"""

    return WindowStruct(sname, sdef)


def window_struct(base_type, n_dims, is_const) -> WindowStruct:
    assert n_dims >= 1

    _window_struct_shorthand = {
        T.f16: "f16",
        T.f32: "f32",
        T.f64: "f64",
        T.i8: "i8",
        T.ui8: "ui8",
        T.ui16: "ui16",
        T.i32: "i32",
    }

    return _window_struct(
        _window_struct_shorthand[base_type], base_type.ctype(), n_dims, is_const
    )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler Entry-points

# top level compiler function called by tests!


def run_compile(proc_list, file_stem: str):
    lib_name = sanitize_str(file_stem)
    fwd_decls, body, ext_lines = ext_compile_to_strings(lib_name, proc_list)

    def join_ext_lines(ext):
        if lines := ext_lines.get(ext):
            return "\n".join(["\n"] + lines + ["\n"])
        else:
            return ""

    source = f"""#include "{file_stem}.h"
{join_ext_lines("c")}
{body}"""

    header_guard = f"{lib_name}_H".upper()
    header = f"""
#pragma once
#ifndef {header_guard}
#define {header_guard}

#ifdef __cplusplus
extern "C" {{
#endif
{join_ext_lines("h")}
{fwd_decls}

#ifdef __cplusplus
}}
#endif
#endif  // {header_guard}
"""

    ext_snippets = {"c": source, "h": header}

    # Gather any non .c, .h files
    for ext, lines in ext_lines.items():
        if ext == "c" or ext == "h":
            continue
        elif ext == "cuh":
            text = f'#pragma once\n#include "{file_stem}.h"\n{join_ext_lines("cuh")}'
        elif ext == "cu":
            text = f'#include "{file_stem}.cuh"\n{join_ext_lines("cu")}'
        else:
            # A bit crappy we have per-file-extension logic here.
            assert "Add case for file extension"
        ext_snippets[ext] = text

    return ext_snippets


_static_helpers = {
    "exo_floor_div": textwrap.dedent(
        """
        static int exo_floor_div(int num, int quot) {
          int off = (num>=0)? 0 : quot-1;
          return (num-off)/quot;
        }
        """
    ),
}


def compile_to_strings(lib_name, proc_list):
    """Legacy wrapper, for procs that don't generate extension files"""
    header, body, ext = ext_compile_to_strings(lib_name, proc_list)
    assert not ext
    return header, body


def ext_compile_to_strings(lib_name, proc_list):
    # Get transitive closure of call-graph
    orig_procs = [id(p) for p in proc_list]

    def from_lines(x):
        return "\n".join(x)

    proc_list = list(sorted(find_all_subprocs(proc_list), key=lambda x: x.name))

    # Header contents
    ctxt_name, ctxt_def = _compile_context_struct(find_all_configs(proc_list), lib_name)
    struct_defns = set()
    public_fwd_decls = []

    # Body contents
    memory_code = _compile_memories(find_all_mems(proc_list))
    private_fwd_decls = []
    proc_bodies = []
    instrs_global = []
    analyzed_proc_list = []

    needed_helpers = set()

    # Compile proc bodies
    seen_procs = set()
    for p in proc_list:
        if p.name in seen_procs:
            raise TypeError(f"multiple procs named {p.name}")
        seen_procs.add(p.name)

        # don't compile instruction procedures, but add a comment.
        if p.instr is not None:
            argstr = ",".join([str(a.name) for a in p.args])
            proc_bodies.extend(
                [
                    "",
                    '/* relying on the following instruction..."',
                    f"{p.name}({argstr})",
                    p.instr.c_instr,
                    "*/",
                ]
            )
            if p.instr.c_global:
                instrs_global.append(p.instr.c_global)
        else:
            is_public_decl = id(p) in orig_procs

            p = ParallelAnalysis().run(p)
            p = PrecisionAnalysis().run(p)
            p = WindowAnalysis().apply_proc(p)
            p = MemoryAnalysis().run(p)

            comp = Compiler(p, ctxt_name, is_public_decl=is_public_decl)
            cpu_d, cpu_b = comp.comp_top()
            struct_defns |= comp.struct_defns()
            needed_helpers |= comp.needed_helpers()

            if is_public_decl:
                public_fwd_decls.append(cpu_d)
            else:
                private_fwd_decls.append(cpu_d)

            # TODO remove
            for gpu_d in comp.spork_decls:
                private_fwd_decls.append(gpu_d)

            proc_bodies.append(cpu_b)

            # TODO remove
            for gpu_b in comp.spork_defs:
                proc_bodies.append("namespace {")
                proc_bodies.append(gpu_b)
                proc_bodies.append("}")

            analyzed_proc_list.append(p)

    # Structs are just blobs of code... still sort them for output stability
    struct_defns = [x.definition for x in sorted(struct_defns, key=lambda x: x.name)]

    header_contents = f"""
#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \\
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif

{from_lines(ctxt_def)}
{from_lines(struct_defns)}
{from_lines(public_fwd_decls)}
"""

    extern_code = _compile_externs(find_all_externs(analyzed_proc_list))

    helper_code = [_static_helpers[v] for v in needed_helpers]
    body_contents = [
        helper_code,
        instrs_global,
        memory_code,
        extern_code,
        private_fwd_decls,
        proc_bodies,
    ]
    body_contents = list(filter(lambda x: x, body_contents))  # filter empty lines
    body_contents = map(from_lines, body_contents)
    body_contents = from_lines(body_contents)
    body_contents += "\n"  # New line at end of file
    return header_contents, body_contents, comp.ext_lines()


def _compile_externs(externs):
    extern_code = []
    for f, t in sorted(externs, key=lambda x: x[0].name() + x[1]):
        if glb := f.globl(t):
            extern_code.append(glb)
    return extern_code


def _compile_memories(mems):
    memory_code = []
    for m in sorted(mems, key=lambda x: x.name()):
        memory_code.append(m.global_())
    return memory_code


def _compile_context_struct(configs, lib_name):
    if not configs:
        return "void", []

    ctxt_name = f"{lib_name}_Context"
    ctxt_def = [f"typedef struct {ctxt_name} {{ ", f""]

    seen = set()
    for c in sorted(configs, key=lambda x: x.name()):
        name = c.name()

        if name in seen:
            raise TypeError(f"multiple configs named {name}")
        seen.add(name)

        if c.is_allow_rw():
            sdef_lines = c.c_struct_def()
            sdef_lines = [f"    {line}" for line in sdef_lines]
            ctxt_def += sdef_lines
            ctxt_def += [""]
        else:
            ctxt_def += [f"// config '{name}' not materialized", ""]

    ctxt_def += [f"}} {ctxt_name};"]
    return ctxt_name, ctxt_def


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler


class Compiler:
    def __init__(self, proc, ctxt_name, *, is_public_decl):
        assert isinstance(proc, LoopIR.proc)

        self.proc = proc
        self.ctxt_name = ctxt_name
        self.env = ChainMap()
        self.range_env = IndexRangeEnvironment(proc, fast=False)
        self.names = ChainMap()
        self.envtyp = dict()
        self.async_config_stack = []

        # TODO remove these
        self.spork = (
            None  # Set to SporkEnv only when compiling GPU kernel code, else None
        )
        self.spork_decls = []  # Fwd declaration of kernels compiled by spork
        self.spork_defs = []  # Definitions of kernels compiled by spork

        # Additional lines for each file extension
        # Since Exo was originally written for only .c and .h files,
        # we have a lot of special treatment for these files,
        # handled separately from this (see comp_top).
        self._ext_lines = {}

        self.mems = dict()
        self._tab = ""
        self._lines = []
        self._scalar_refs = set()
        self._needed_helpers = set()
        self.window_defns = set()
        self._known_strides = {}

        assert self.proc.name is not None, "expected names for compilation"
        name = self.proc.name
        arg_strs = []
        typ_comments = []

        # reserve the first "ctxt" argument
        self.new_varname(Sym("ctxt"), None)
        arg_strs.append(f"{ctxt_name} *ctxt")

        self.non_const = set(e for e, _ in get_writes_of_stmts(self.proc.body))

        for a in proc.args:
            typ = a.type
            mem = a.mem if typ.is_numeric() else None
            name_arg = self.new_varname(a.name, typ=typ, mem=mem)
            is_const = not (typ.is_numeric() and a.name in self.non_const)
            if typ.is_real_scalar():
                self._scalar_refs.add(a.name)

            mem_comment = f" @{mem.name()}" if mem else ""
            arg_strs.append(f"{self.format_fnarg_ctype(a, is_const)} {name_arg}")
            typ_comments.append(f"{name_arg} : {typ}{mem_comment}")

        for pred in proc.preds:
            if isinstance(pred, LoopIR.Const):
                # TODO: filter these out earlier?
                continue

            if (
                isinstance(pred, LoopIR.BinOp)
                and pred.op == "=="
                and isinstance(pred.lhs, LoopIR.StrideExpr)
                and isinstance(pred.rhs, LoopIR.Const)
            ):
                self._known_strides[(pred.lhs.name, pred.lhs.dim)] = CIR.Const(
                    pred.rhs.val
                )
                self.add_line(f"// assert {pred}")
            else:
                # Default to just informing the compiler about the constraint
                # on a best-effort basis
                self.add_line(f"EXO_ASSUME({self.comp_e(pred)});")

        if not self.static_memory_check(self.proc):
            raise MemGenError("Cannot generate static memory in non-leaf procs")

        self.comp_stmts(self.proc.body)

        static_kwd = "" if is_public_decl else "static "

        # Generate headers here?
        comment = (
            f"// {name}(\n" + ",\n".join(["//     " + s for s in typ_comments]) + "\n"
            "// )\n"
        )
        proc_decl = comment + f"{static_kwd}void {name}( {', '.join(arg_strs)} );\n"
        proc_def = (
            comment
            + f"{static_kwd}void {name}( {', '.join(arg_strs)} ) {{\n"
            + "\n".join(self._lines)
            + "\n"
            "}\n"
        )

        self.proc_decl = proc_decl
        self.proc_def = proc_def

    def format_fnarg_ctype(self, a, is_const):
        typ = a.type
        if typ in (T.size, T.index, T.bool, T.stride):
            return typ.ctype()
        # setup, arguments
        else:
            assert typ.is_numeric()
            assert typ.basetype() != T.R
            if typ.is_win():
                wintyp = self.get_window_type(a, is_const)
                return f"struct {wintyp}"
            else:
                const_kwd = "const " if is_const else ""
                ctyp = typ.basetype().ctype()
                return f"{const_kwd}{ctyp}*"

    def static_memory_check(self, proc):
        def allocates_static_memory(stmts):
            check = False
            for s in stmts:
                if isinstance(s, LoopIR.Alloc):
                    mem = s.mem
                    assert issubclass(mem, Memory)
                    check |= issubclass(mem, StaticMemory)
                elif isinstance(s, LoopIR.For):
                    check |= allocates_static_memory(s.body)
                elif isinstance(s, LoopIR.If):
                    check |= allocates_static_memory(s.body)
                    check |= allocates_static_memory(s.orelse)
            return check

        def is_leaf_proc(stmts):
            check = True
            for s in stmts:
                if isinstance(s, LoopIR.Call):
                    # Since intrinsics don't allocate memory, we can ignore
                    # them for leaf-node classification purposes. We want
                    # to avoid nested procs that both allocate static memory.
                    check &= s.f.instr is not None
                elif isinstance(s, LoopIR.For):
                    check &= is_leaf_proc(s.body)
                elif isinstance(s, LoopIR.If):
                    check &= is_leaf_proc(s.body)
                    check &= is_leaf_proc(s.orelse)
            return check

        return not allocates_static_memory(proc.body) or is_leaf_proc(proc.body)

    def add_line(self, line):
        if line:
            if self.spork:
                self.spork.add_line(self._tab + line)
            else:
                self._lines.append(self._tab + line)

    def comp_stmts(self, stmts):
        for b in stmts:
            self.comp_s(b)

    def comp_top(self):
        return self.proc_decl, self.proc_def

    def ext_lines(self):
        return self._ext_lines

    def struct_defns(self):
        return self.window_defns

    def needed_helpers(self):
        return self._needed_helpers

    def new_varname(self, symbol, typ, mem=None):
        strnm = str(symbol)
        if strnm not in self.names:
            pass
        else:
            s = self.names[strnm]
            while s in self.names:
                m = re.match(r"^(.*)_([0-9]*)$", s)
                if not m:
                    s = s + "_1"
                else:
                    s = f"{m[1]}_{int(m[2]) + 1}"
            self.names[strnm] = s
            strnm = s

        self.names[strnm] = strnm
        self.env[symbol] = strnm
        self.envtyp[symbol] = typ
        if mem is not None:
            self.mems[symbol] = mem
        else:
            self.mems[symbol] = DRAM
        return strnm

    def push(self, only=None):
        if only is None:
            self.env = self.env.new_child()
            self.range_env.enter_scope()
            self.names = self.names.new_child()
            self._tab = self._tab + "  "
        elif only == "env":
            self.env = self.env.new_child()
            self.range_env.enter_scope()
            self.names = self.names.new_child()
        elif only == "tab":
            self._tab = self._tab + "  "
        else:
            assert False, f"BAD only parameter {only}"

    def pop(self):
        self.env = self.env.parents
        self.range_env.exit_scope()
        self.names = self.names.parents
        self._tab = self._tab[:-2]

    def comp_cir(self, e, env, prec) -> str:
        if isinstance(e, CIR.Read):
            return env[e.name]

        elif isinstance(e, CIR.Const):
            return str(e.val)

        elif isinstance(e, CIR.BinOp):
            local_prec = op_prec[e.op]

            lhs = self.comp_cir(e.lhs, env, local_prec)
            rhs = self.comp_cir(e.rhs, env, local_prec)

            if isinstance(e.rhs, CIR.BinOp) and (e.op == "-" or e.op == "/"):
                rhs = f"({rhs})"

            if e.op == "/":
                if (isinstance(e.lhs, (CIR.Read, CIR.BinOp)) and e.lhs.is_non_neg) or (
                    isinstance(e.lhs, CIR.Const) and e.lhs.val > 0
                ):
                    return f"({lhs} / {rhs})"
                else:
                    return self._call_static_helper("exo_floor_div", lhs, rhs)

            s = f"{lhs} {e.op} {rhs}"
            if local_prec < prec:
                s = f"({s})"

            return s

        elif isinstance(e, CIR.Stride):
            return f"{e.name}.strides[{e.dim}]"
        elif isinstance(e, CIR.USub):
            return f'-{self.comp_cir(e.arg, env, op_prec["~"])}'
        else:
            assert False, "bad case!"

    def access_str(self, nm, idx_list) -> str:
        type = self.envtyp[nm]
        cirs = [lift_to_cir(i, self.range_env) for i in idx_list]
        idx_expr = self.get_idx_offset(nm, type, cirs)
        idx_expr_s = self.comp_cir(simplify_cir(idx_expr), self.env, prec=0)
        buf = self.env[nm]
        if not type.is_win():
            return f"{buf}[{idx_expr_s}]"
        else:
            return f"{buf}.data[{idx_expr_s}]"

    def shape_strs(self, shape, prec=100) -> str:
        comp_res = [
            self.comp_cir(simplify_cir(lift_to_cir(i, self.range_env)), self.env, prec)
            for i in shape
        ]
        return comp_res

    def tensor_strides(self, shape) -> CIR:
        szs = [lift_to_cir(i, self.range_env) for i in shape]
        assert len(szs) >= 1
        strides = [CIR.Const(1)]
        s = szs[-1]
        for sz in reversed(szs[:-1]):
            strides.append(s)
            s = CIR.BinOp("*", sz, s, True)
        strides = list(reversed(strides))

        return strides

    # works for any tensor or window type
    def get_strides(self, name: Sym, typ) -> CIR:
        if typ.is_win():
            res = []
            for i in range(len(typ.shape())):
                if stride := self._known_strides.get((name, i)):
                    res.append(stride)
                else:
                    res.append(CIR.Stride(name, i))

            return res
        else:
            return self.tensor_strides(typ.shape())

    def get_idx_offset(self, name: Sym, typ, idx) -> CIR:
        strides = self.get_strides(name, typ)
        assert len(strides) == len(idx)
        acc = CIR.BinOp("*", idx[0], strides[0], True)
        for i, s in zip(idx[1:], strides[1:]):
            new = CIR.BinOp("*", i, s, True)
            acc = CIR.BinOp("+", acc, new, True)

        return acc

    def get_window_type(self, typ, is_const=None):
        assert isinstance(typ, T.Window) or (
            isinstance(typ, LoopIR.fnarg) and typ.type.is_win()
        )

        if isinstance(typ, T.Window):
            base = typ.as_tensor.basetype()
            n_dims = len(typ.as_tensor.shape())
            if is_const is None:
                is_const = typ.src_buf not in self.non_const
        else:
            base = typ.type.basetype()
            n_dims = len(typ.type.shape())
            if is_const is None:
                is_const = typ.name not in self.non_const

        win = window_struct(base, n_dims, is_const)
        self.window_defns.add(win)
        return win.name

    def get_actor_kind(self):
        return (
            self.spork.get_async_config().get_actor_kind()
            if self.spork
            else actor_kinds.cpu
        )

    def comp_s(self, s):
        if isinstance(s, LoopIR.Pass):
            self.add_line("; // NO-OP")
        elif isinstance(s, LoopIR.SyncStmt):
            if s.codegen is None:
                raise TypeError(
                    f"{s.srcinfo}: SyncStmt not allowed here "
                    "(or internal compiler error -- missing codegen)"
                )
            self.add_line(s.codegen)
        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if s.name in self._scalar_refs:
                lhs = f"*{self.env[s.name]}"
            elif self.envtyp[s.name].is_real_scalar():
                lhs = self.env[s.name]
            else:
                lhs = self.access_str(s.name, s.idx)
            rhs = self.comp_e(s.rhs)

            # possibly cast!
            lbtyp = s.type.basetype()
            rbtyp = s.rhs.type.basetype()
            if lbtyp != rbtyp:
                assert s.type.is_real_scalar()
                assert s.rhs.type.is_real_scalar()

                rhs = f"({lbtyp.ctype()})({rhs})"

            mem: Memory = self.mems[s.name]
            if isinstance(s, LoopIR.Assign):
                self.add_line(mem.write(s, lhs, rhs))
            else:
                self.add_line(mem.reduce(s, lhs, rhs))

        elif isinstance(s, LoopIR.WriteConfig):
            if not s.config.is_allow_rw():
                raise ConfigError(
                    f"{s.srcinfo}: cannot write to config '{s.config.name()}'"
                )

            nm = s.config.name()
            rhs = self.comp_e(s.rhs)

            # possibly cast!
            ltyp = s.config.lookup_type(s.field)
            rtyp = s.rhs.type
            if ltyp != rtyp and not ltyp.is_indexable():
                assert ltyp.is_real_scalar()
                assert rtyp.is_real_scalar()

                rhs = f"({ltyp.ctype()})({rhs})"

            self.add_line(f"ctxt->{nm}.{s.field} = {rhs};")

        elif isinstance(s, LoopIR.WindowStmt):
            win_struct = self.get_window_type(s.rhs.type)
            rhs = self.comp_e(s.rhs)
            assert isinstance(s.rhs, LoopIR.WindowExpr)
            mem = self.mems[s.rhs.name]
            name = self.new_varname(s.name, typ=s.rhs.type, mem=mem)
            self.add_line(f"struct {win_struct} {name} = {rhs};")

        elif is_if_holding_with(s, LoopIR):  # must be before .If case
            ctx = s.cond.val
            if isinstance(ctx, ExtWithContext):
                # Reset indentation and direct text lines for compiled subtree
                # to new location (per-file-extension lines dict).
                old_lines = self._lines
                old_tab = self._tab
                self._lines = self._ext_lines.setdefault(ctx.body_ext, [])
                self._tab = ""

                # Add code snippets
                for ext, snippet in ctx.ext_snippets.items():
                    self._ext_lines.setdefault(ext, []).append(snippet)

                # Compile body, with prefix and suffix.
                # Note ordering after snippets are added, as promised in ExtWithContext.
                self.add_line(ctx.body_prefix)  # Might not really be just 1 line...
                self._tab += "  "
                self.comp_stmts(s.body)
                self._tab = ""
                self.add_line(ctx.body_suffix)

                # Restore old lines list and indentation
                self._tab = old_tab
                self._lines = old_lines

                # Add kernel launch syntax
                self.add_line(ctx.launch)

            elif isinstance(ctx, BaseAsyncConfig):
                # TODO change all of this!!!
                starting_kernel = not self.spork

                # Check async block is valid here
                expected_async_type = ctx.parent_async_type()
                parent_async_config = (
                    self.spork.get_async_config() if self.spork else "<no async block>"
                )
                expected_str = (
                    expected_async_type.__name__
                    if expected_async_type
                    else "<no async block>"
                )

                if (
                    self.spork
                    and expected_async_type
                    and isinstance(parent_async_config, expected_async_type)
                ):
                    pass
                elif not self.spork and not expected_async_type:
                    pass
                else:
                    raise TypeError(
                        f"Async block {ctx} must be nested in {expected_str}, not {parent_async_config}"
                    )

                if starting_kernel:
                    old_tabs = self._tab
                    self._tab = "  "
                    device_name = ctx.get_device_name()
                    assert (
                        device_name == "cuda"
                    ), "Future: subtypes of SporkEnv for non-cuda?"
                    self.spork = SporkEnv(
                        f"{self.proc.name}_{s.srcinfo.lineno:04d}_EXO", s
                    )
                else:
                    self.spork.push_async(ctx)

                self.push()
                self.comp_stmts(s.body)
                self.pop()

                if starting_kernel:
                    # Divert gpu code into separate lists
                    # Insert cpu code for kernel launch
                    proto, launch = self.spork.get_kernel_prototype_launch(
                        s, self.env, self.envtyp, self.format_fnarg_ctype
                    )
                    kernel_body = self.spork.get_kernel_body()

                    # Must clear SporkEnv now so add_line works and restore tabs
                    self.spork = None
                    self._tab = old_tabs

                    self.spork_decls.append("namespace{" + proto + ";}")
                    self.spork_defs.append(proto + "\n" + kernel_body)
                    self.add_line(launch)
                else:
                    self.spork.pop_async()

            elif isinstance(ctx, SpecializeCollective):
                if not self.spork:
                    raise ValueError(
                        f"{s.srcinfo}: SpecializeCollective outside async block"
                    )
                cond = self.spork.push_specialize_collective(s)

                self.add_line(f"if ({cond}) {{")
                self.push()
                self.comp_stmts(s.body)
                self.pop()
                assert len(s.orelse) == 0
                self.add_line("}")

                self.spork.pop_specialize_collective(s)
            else:
                raise TypeError(f"Unknown with stmt context type {type(ctx)}")

        # If statement that is not disguising a with statement
        # (remove note when this hack is fixed)
        elif isinstance(s, LoopIR.If):
            cond = self.comp_e(s.cond)
            self.add_line(f"if ({cond}) {{")
            self.push()
            self.comp_stmts(s.body)
            self.pop()
            if len(s.orelse) > 0:
                self.add_line("} else {")
                self.push()
                self.comp_stmts(s.orelse)
                self.pop()
            self.add_line("}")

        elif isinstance(s, LoopIR.For):
            lo = self.comp_e(s.lo)
            hi = self.comp_e(s.hi)
            self.push(only="env")
            itr = self.new_varname(s.iter, typ=T.index)  # allocate a new string
            sym_range = self.range_env.add_loop_iter(
                s.iter,
                s.lo,
                s.hi,
            )

            loop_mode = s.loop_mode
            emit_loop = True

            if isinstance(loop_mode, Par):
                self.add_line(f"#pragma omp parallel for")
            elif isinstance(loop_mode, Seq):
                pass  # common case
            elif isinstance(loop_mode, _CodegenPar):
                # This is not valid C; if we add non-cuda backends we may have
                # to add config options to _CodegenPar to tweak lowering syntax.
                conds = []
                if bdd := loop_mode.static_bounds[0] is not None:
                    conds.append(f"{itr} >= {bdd}")
                if bdd := loop_mode.static_bounds[1] is not None:
                    conds.append(f"{itr} < {bdd}")
                cond = "1" if not conds else " && ".join(conds)
                self.add_line(
                    f"if ([[mabye_unused]] int {itr} = ({loop_mode.c_index}); {cond}) {{"
                )
                emit_loop = False
            else:
                raise TypeError(
                    f"{s.srcinfo}: unexpected loop mode {loop_mode.loop_mode_name()}"
                )

            if emit_loop:
                self.add_line(
                    f"for (int_fast32_t {itr} = {lo}; {itr} < {hi}; {itr}++) {{"
                )

            self.push(only="tab")
            self.comp_stmts(s.body)

            self.pop()
            self.add_line("}")

        elif isinstance(s, LoopIR.Alloc):
            name = self.new_varname(s.name, typ=s.type, mem=s.mem)
            if isinstance(s.type, T.Barrier):
                self.add_line(f"// Scope of named barrier {s.name}")
            else:
                assert s.type.basetype().is_real_scalar()
                assert s.type.basetype() != T.R
                ctype = s.type.basetype().ctype()
                mem = s.mem or DRAM
                line = mem.alloc(
                    name, ctype, self.shape_strs(s.type.shape()), s.srcinfo
                )
                self.add_line(line)
        elif isinstance(s, LoopIR.Free):
            name = self.env[s.name]
            if isinstance(s.type, T.Barrier):
                pass
            else:
                assert s.type.basetype().is_real_scalar()
                ctype = s.type.basetype().ctype()
                mem = s.mem or DRAM
                line = mem.free(name, ctype, self.shape_strs(s.type.shape()), s.srcinfo)
                self.add_line(line)
        elif isinstance(s, LoopIR.Call):
            assert all(
                a.type.is_win() == fna.type.is_win() for a, fna in zip(s.args, s.f.args)
            )
            args = [self.comp_fnarg(e, s.f, i) for i, e in enumerate(s.args)]
            if s.f.instr is not None:
                d = dict()
                assert len(s.f.args) == len(args)
                for i in range(len(args)):
                    arg_name = str(s.f.args[i].name)
                    d[arg_name] = f"({args[i]})"
                    arg_type = s.args[i].type
                    if arg_type.is_win():
                        assert isinstance(s.args[i], LoopIR.WindowExpr)
                        data, _ = self.window_struct_fields(s.args[i])
                        d[f"{arg_name}_data"] = data
                        d[f"{arg_name}_int"] = self.env[s.args[i].name]
                    else:
                        d[f"{arg_name}_data"] = f"({args[i]})"

                self.add_line(f"{s.f.instr.c_instr.format(**d)}")
            else:
                fname = s.f.name
                args = ["ctxt"] + args
                self.add_line(f"{fname}({','.join(args)});")
        else:
            assert False, "bad case"

    def comp_fnarg(self, e, fn, i, *, prec=0):
        if isinstance(e, LoopIR.Read):
            assert not e.idx
            rtyp = self.envtyp[e.name]
            if rtyp.is_indexable():
                return self.env[e.name]
            elif rtyp is T.bool:
                return self.env[e.name]
            elif rtyp is T.stride:
                return self.env[e.name]
            elif e.name in self._scalar_refs:
                return self.env[e.name]
            elif rtyp.is_tensor_or_window():
                return self.env[e.name]
            else:
                assert rtyp.is_real_scalar()
                return f"&{self.env[e.name]}"
        elif isinstance(e, LoopIR.WindowExpr):
            if isinstance(fn, LoopIR.proc):
                callee_buf = fn.args[i].name
                is_const = callee_buf not in set(
                    x for x, _ in get_writes_of_stmts(fn.body)
                )
            else:
                raise NotImplementedError("Passing windows to externs")
            win_struct = self.get_window_type(e.type, is_const)
            data, strides = self.window_struct_fields(e)
            return f"(struct {win_struct}){{ &{data}, {{ {strides} }} }}"
        else:
            return self.comp_e(e, prec)

    def comp_e(self, e, prec=0):
        if isinstance(e, LoopIR.Read):
            rtyp = self.envtyp[e.name]
            if rtyp.is_indexable() or rtyp is T.bool or rtyp == T.stride:
                return self.env[e.name]

            mem: Memory = self.mems[e.name]

            if mem is CudaRmem:  # BACKDOOR
                warnings.warn("Backdoor for reading CudaRmem")
                return self.env[e.name]

            if not mem.can_read():
                raise MemGenError(
                    f"{e.srcinfo}: cannot read from buffer "
                    f"'{e.name}' in memory '{mem.name()}'"
                )

            if e.name in self._scalar_refs:
                return f"*{self.env[e.name]}"
            elif not rtyp.is_tensor_or_window():
                return self.env[e.name]
            else:
                return self.access_str(e.name, e.idx)

        elif isinstance(e, LoopIR.WindowExpr):
            win_struct = self.get_window_type(e.type)
            data, strides = self.window_struct_fields(e)
            return f"(struct {win_struct}){{ &{data}, {{ {strides} }} }}"

        elif isinstance(e, LoopIR.Const):
            if isinstance(e.val, bool):
                return "true" if e.val else "false"
            elif e.type.is_indexable():
                return f"{int(e.val)}"
            elif e.type == T.f64:
                return f"{float(e.val)}"
            elif e.type == T.f32:
                return f"{float(e.val)}f"
            elif e.type == T.with_context:
                assert False, "should be handled when compiling LoopIR.If"
            else:
                return f"(({e.type.ctype()}) {str(e.val)})"

        elif isinstance(e, LoopIR.BinOp):
            local_prec = op_prec[e.op]
            int_div = e.op == "/" and not e.type.is_numeric()
            if int_div:
                local_prec = 0
            op = e.op
            if op == "and":
                op = "&&"
            elif op == "or":
                op = "||"

            lhs = self.comp_e(e.lhs, local_prec)
            rhs = self.comp_e(e.rhs, local_prec + 1)

            if int_div:
                if self.range_env.check_expr_bound(0, IndexRangeEnvironment.leq, e):
                    # TODO: too many parens?
                    return f"(({lhs}) / ({rhs}))"
                return self._call_static_helper("exo_floor_div", lhs, rhs)

            s = f"{lhs} {op} {rhs}"
            if local_prec < prec:
                s = f"({s})"

            return s
        elif isinstance(e, LoopIR.USub):
            return f'-{self.comp_e(e.arg, op_prec["~"])}'

        elif isinstance(e, LoopIR.Extern):
            args = [self.comp_e(a) for a in e.args]
            return e.f.compile(args, e.type.basetype().ctype())

        elif isinstance(e, LoopIR.StrideExpr):
            basetyp = self.envtyp[e.name]
            stride = self.get_strides(e.name, basetyp)[e.dim]
            return self.comp_cir(simplify_cir(stride), self.env, prec=0)

        elif isinstance(e, LoopIR.ReadConfig):
            if not e.config.is_allow_rw():
                raise ConfigError(
                    f"{e.srcinfo}: cannot read from config '{e.config.name()}'"
                )
            return f"ctxt->{e.config.name()}.{e.field}"

        else:
            assert False, "bad case"

    def _call_static_helper(self, helper, *args):
        self._needed_helpers.add(helper)
        return f'{helper}({", ".join(map(str, args))})'

    def window_struct_fields(self, e):
        base = self.env[e.name]
        basetyp = self.envtyp[e.name]
        mem: Memory = self.mems[e.name]

        # compute offset to new data pointer
        def w_lo(w):
            return w.lo if isinstance(w, LoopIR.Interval) else w.pt

        cirs = [lift_to_cir(w_lo(w), self.range_env) for w in e.idx]
        idxs = [self.comp_cir(simplify_cir(i), self.env, prec=0) for i in cirs]

        # compute new window strides
        all_strides = self.get_strides(e.name, basetyp)
        all_strides_s = [
            self.comp_cir(simplify_cir(i), self.env, prec=0) for i in all_strides
        ]
        assert 0 < len(all_strides_s) == len(e.idx)
        dataptr = mem.window(basetyp, base, idxs, all_strides_s, e.srcinfo)
        strides = ", ".join(
            s for s, w in zip(all_strides_s, e.idx) if isinstance(w, LoopIR.Interval)
        )
        return dataptr, strides
