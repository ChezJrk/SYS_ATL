# import ast as pyast
import functools
import inspect
import re

# import types
from dataclasses import dataclass
from typing import Any, List, Tuple

from .API import Procedure
import exo.API_cursors as PC
from .core.LoopIR import LoopIR, T
import exo.rewrite.LoopIR_scheduling as scheduling
from .API_types import ExoType

from .rewrite.LoopIR_unification import DoReplace, UnificationError
from .core.configs import Config
from .core.memory import Memory
from .frontend.parse_fragment import parse_fragment
from .core.prelude import *
from .core import internal_cursors as ic


def is_subclass_obj(x, cls):
    return isinstance(x, type) and issubclass(x, cls)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Generic Definitions: Atomic Scheduling Operations and Argument Processing


@dataclass
class ArgumentProcessor:
    i: int
    arg_name: str
    f_name: str

    def __init__(self):
        # see setdata below for setting of the above fields
        pass

    def err(self, message, Error=TypeError):
        raise Error(f"argument {self.i}, '{self.arg_name}' to {self.f_name}: {message}")

    def setdata(self, i, arg_name, f_name):
        self.i = i
        self.arg_name = arg_name
        self.f_name = f_name

    def __call__(self, arg, all_args):
        raise NotImplementedError("Must Sub-class and redefine __call__")


class CursorArgumentProcessor(ArgumentProcessor):
    def __call__(self, cur, all_args):
        p = all_args["proc"]
        if isinstance(cur, PC.Cursor):
            cur = p.forward(cur)
        elif isinstance(cur, list):
            for i in range(len(cur)):
                cur[i] = p.forward(cur[i])
        return self._cursor_call(cur, all_args)

    def _cursor_call(self, cur, all_args):
        raise NotImplementedError("abstract method")


@dataclass
class AtomicSchedulingOp:
    sig: inspect.Signature
    arg_procs: List[ArgumentProcessor]
    func: Any

    def __str__(self):
        return f"<AtomicSchedulingOp-{self.__name__}>"

    def __call__(self, *args, **kwargs):
        # capture the arguments according to the provided signature
        bound_args = self.sig.bind(*args, **kwargs)

        # potentially need to patch in default values...
        bargs = bound_args.arguments
        if len(self.arg_procs) != len(bargs):
            for nm in self.sig.parameters:
                if nm not in bargs:
                    default_val = self.sig.parameters[nm].default
                    assert default_val != inspect.Parameter.empty
                    kwargs[nm] = default_val
            # now re-bind the arguments with the defaults having been added
            bound_args = self.sig.bind(*args, **kwargs)
            bargs = bound_args.arguments

        # convert the arguments using the provided argument processors
        assert len(self.arg_procs) == len(bargs)
        for nm, argp in zip(bargs, self.arg_procs):
            bargs[nm] = argp(bargs[nm], bargs)

        # invoke the scheduling function with the modified arguments
        return self.func(*bound_args.args, **bound_args.kwargs)


# decorator for building Atomic Scheduling Operations in the
# remainder of this file
def sched_op(arg_procs):
    def check_ArgP(argp):
        if is_subclass_obj(argp, ArgumentProcessor):
            return argp()
        else:
            assert isinstance(argp, ArgumentProcessor)
            return argp

    # note pre-pending of ProcA
    arg_procs = [check_ArgP(argp) for argp in ([ProcA] + arg_procs)]

    def build_sched_op(func):
        f_name = func.__name__
        sig = inspect.signature(func)
        assert len(arg_procs) == len(sig.parameters)

        # record extra implicit information in the argument processors
        for i, (param, arg_p) in enumerate(zip(sig.parameters, arg_procs)):
            arg_p.setdata(i, param, f_name)

        atomic_op = AtomicSchedulingOp(sig, arg_procs, func)
        return functools.wraps(func)(atomic_op)

    return build_sched_op


def is_atomic_scheduling_op(x):
    return isinstance(x, AtomicSchedulingOp)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Argument Processing


class IdA(ArgumentProcessor):
    def __call__(self, arg, all_args):
        return arg


class ProcA(ArgumentProcessor):
    def __call__(self, proc, all_args):
        if not isinstance(proc, Procedure):
            self.err("expected a Procedure object")
        return proc


class MemoryA(ArgumentProcessor):
    def __call__(self, mem, all_args):
        if not is_subclass_obj(mem, Memory):
            self.err("expected a Memory subclass")
        return mem


class ConfigA(ArgumentProcessor):
    def __call__(self, config, all_args):
        if not isinstance(config, Config):
            self.err("expected a Config object")
        return config


class ConfigFieldA(ArgumentProcessor):
    def __init__(self, config_arg_name="config"):
        self.cfg_arg = config_arg_name

    def __call__(self, field, all_args):
        config = all_args[self.cfg_arg]
        if not is_valid_name(field):
            self.err("expected a valid name string")
        elif not config.has_field(field):
            self.err(
                f"expected '{field}' to be a field of config '{config.name()}'",
                ValueError,
            )
        return field


class NameA(ArgumentProcessor):
    def __call__(self, name, all_args):
        if not is_valid_name(name):
            self.err("expected a valid name")
        return name


class PosIntA(ArgumentProcessor):
    def __call__(self, val, all_args):
        if not is_pos_int(val):
            self.err("expected a positive integer")
        return val


class IntA(ArgumentProcessor):
    def __call__(self, val, all_args):
        if not isinstance(val, int):
            self.err("expected an integer")
        return val


class BoolA(ArgumentProcessor):
    def __call__(self, bval, all_args):
        if not isinstance(bval, bool):
            self.err("expected a bool")
        return bval


class OptionalA(ArgumentProcessor):
    def __init__(self, arg_proc):
        if is_subclass_obj(arg_proc, ArgumentProcessor):
            arg_proc = arg_proc()
        self.arg_proc = arg_proc

    def setdata(self, i, arg_name, f_name):
        super().setdata(i, arg_name, f_name)
        self.arg_proc.setdata(i, arg_name, f_name)

    def __call__(self, opt_arg, all_args):
        if opt_arg is None:
            return opt_arg
        else:
            return self.arg_proc(opt_arg, all_args)


class DictA(ArgumentProcessor):
    def __call__(self, d, all_args):
        if not isinstance(d, dict):
            self.err("expected a dict")
        return d


class ListA(ArgumentProcessor):
    def __init__(self, elem_arg_proc, list_only=False, length=None):
        if is_subclass_obj(elem_arg_proc, ArgumentProcessor):
            elem_arg_proc = elem_arg_proc()
        self.elem_arg_proc = elem_arg_proc
        self.list_only = list_only
        self.fixed_length = length

    def setdata(self, i, arg_name, f_name):
        super().setdata(i, arg_name, f_name)
        self.elem_arg_proc.setdata(i, arg_name, f_name)

    def __call__(self, xs, all_args):
        if self.list_only:
            if not isinstance(xs, list):
                self.err("expected a list")
        else:
            if not isinstance(xs, (list, tuple)):
                self.err("expected a list or tuple")
        if self.fixed_length:
            if len(xs) != self.fixed_length:
                self.err(f"expected a list of length {self.fixed_length}")
        # otherwise, check the entries
        xs = [self.elem_arg_proc(x, all_args) for x in xs]
        return xs


class ListOrElemA(ListA):
    def __call__(self, xs, all_args):
        arg_typ = list if self.list_only else (list, tuple)
        if isinstance(xs, arg_typ):
            return super().__call__(xs, all_args)
        else:
            return [self.elem_arg_proc(xs, all_args)]


class InstrStrA(ArgumentProcessor):
    def __call__(self, instr, all_args):
        if not isinstance(instr, str):
            self.err("expected an instruction macro " "(i.e. a string with {} escapes)")
        return instr


_name_count_re = r"^([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$"


class NameCountA(ArgumentProcessor):
    def __call__(self, name_count, all_args):
        if not isinstance(name_count, str):
            self.err("expected a string")
        results = re.search(_name_count_re, name_count)
        if not results:
            self.err(
                "expected a name pattern of the form\n"
                "  <ident> [# <int>]?\n"
                "where <ident> is the name of a variable "
                "and <int> specifies which occurrence. "
                "(e.g. 'x #2' means 'the second occurence of x')",
                ValueError,
            )

        name = results[1]
        count = int(results[3]) if results[3] else None
        return name, count


class EnumA(ArgumentProcessor):
    def __init__(self, enum_vals):
        assert isinstance(enum_vals, list)
        self.enum_vals = enum_vals

    def __call__(self, arg, all_args):
        if arg not in self.enum_vals:
            vals_str = ", ".join([str(v) for v in self.enum_vals])
            self.err(f"expected one of the following values: {vals_str}", ValueError)
        return arg


class TypeAbbrevA(ArgumentProcessor):
    _shorthand = {
        "R": T.R,
        ExoType.R: T.R,
        "f16": T.f16,
        ExoType.F16: T.f16,
        "f32": T.f32,
        ExoType.F32: T.f32,
        "f64": T.f64,
        ExoType.F64: T.f64,
        "i8": T.int8,
        ExoType.I8: T.i8,
        "ui8": T.uint8,
        ExoType.UI8: T.uint8,
        "ui16": T.uint16,
        ExoType.UI16: T.ui16,
        "i32": T.int32,
        ExoType.I32: T.i32,
    }

    def __call__(self, typ, all_args):
        if not isinstance(typ, (str, ExoType)):
            self.err(
                f"expected an instance of {ExoType} or {str} specifying the precision",
                TypeError,
            )
        assert not isinstance(typ, ExoType) or typ in TypeAbbrevA._shorthand
        if typ in TypeAbbrevA._shorthand:
            return TypeAbbrevA._shorthand[typ]
        else:
            precisions = ", ".join(
                [t for t in TypeAbbrevA._shorthand if type(t) is str]
            )
            self.err(
                f"expected an instance of {ExoType} or one of the following strings specifying "
                f"precision: {precisions}",
                ValueError,
            )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cursor Argument Processing


class ExprCursorA(CursorArgumentProcessor):
    def __init__(self, many=False):
        self.match_many = many

    def _cursor_call(self, expr_pattern, all_args):
        if self.match_many:
            if isinstance(expr_pattern, list):
                if all(isinstance(ec, PC.ExprCursor) for ec in expr_pattern):
                    return expr_pattern
                else:
                    for ec in expr_pattern:
                        if not isinstance(ec, PC.ExprCursor):
                            self.err(
                                f"expected a list of ExprCursor, "
                                f"not {type(expr_pattern)}"
                            )
            elif not isinstance(expr_pattern, str):
                self.err("expected an ExprCursor or pattern string")
        else:
            if isinstance(expr_pattern, PC.ExprCursor):
                return expr_pattern
            elif isinstance(expr_pattern, PC.Cursor):
                self.err(f"expected an ExprCursor, not {type(expr_pattern)}")
            elif not isinstance(expr_pattern, str):
                self.err("expected an ExprCursor or pattern string")

        proc = all_args["proc"]
        matches = proc.find(expr_pattern, many=self.match_many, call_depth=1)

        if self.match_many:
            for m in matches:
                if not isinstance(m, PC.ExprCursor):
                    self.err(
                        f"expected pattern to match only ExprCursors, not {type(m)}"
                    )
            return matches
        else:
            match = matches
            if not isinstance(match, PC.ExprCursor):
                self.err(f"expected pattern to match an ExprCursor, not {type(match)}")
            return match


class StmtCursorA(CursorArgumentProcessor):
    def __init__(self, many=False):
        self.match_many = many

    def _cursor_call(self, stmt_pattern, all_args):
        if isinstance(stmt_pattern, PC.StmtCursor):
            return stmt_pattern
        elif isinstance(stmt_pattern, PC.Cursor):
            self.err(f"expected an StmtCursor, not {type(stmt_pattern)}")
        elif not isinstance(stmt_pattern, str):
            self.err("expected a StmtCursor or pattern string")

        proc = all_args["proc"]
        matches = proc.find(stmt_pattern, many=self.match_many, call_depth=1)

        match = matches[0] if self.match_many else matches
        if not isinstance(match, PC.StmtCursor):
            self.err(f"expected pattern to match a StmtCursor, not {type(match)}")

        return match


class BlockCursorA(CursorArgumentProcessor):
    def __init__(self, many=False, block_size=None):
        self.match_many = many
        self.block_size = block_size

    def _cursor_call(self, block_pattern, all_args):
        if isinstance(block_pattern, PC.BlockCursor):
            cursor = block_pattern
        elif isinstance(block_pattern, PC.StmtCursor):
            cursor = block_pattern.as_block()
        else:
            if isinstance(block_pattern, PC.Cursor):
                self.err(
                    f"expected a StmtCursor or BlockCursor, "
                    f"not {type(block_pattern)}"
                )
            elif not isinstance(block_pattern, str):
                self.err("expected a Cursor or pattern string")

            proc = all_args["proc"]
            matches = proc.find(block_pattern, many=self.match_many, call_depth=1)

            match = matches[0] if self.match_many else matches
            if isinstance(match, PC.StmtCursor):
                match = match.as_block()
            elif not isinstance(match, PC.BlockCursor):
                self.err(f"expected pattern to match a BlockCursor, not {type(match)}")
            cursor = match

        # regardless, check block size
        if self.block_size:
            if len(cursor) != self.block_size:
                self.err(
                    f"expected a block of size {self.block_size}, "
                    f"but got a block of size {len(cursor)}",
                    ValueError,
                )

        return cursor


class GapCursorA(CursorArgumentProcessor):
    def _cursor_call(self, gap_cursor, all_args):
        if not isinstance(gap_cursor, PC.GapCursor):
            self.err("expected a GapCursor")
        return gap_cursor


class AllocCursorA(StmtCursorA):
    def _cursor_call(self, alloc_pattern, all_args):
        try:
            name, count = NameCountA()(alloc_pattern, all_args)
            count = f" #{count}" if count is not None else ""
            alloc_pattern = f"{name} : _{count}"
        except:
            pass

        cursor = super()._cursor_call(alloc_pattern, all_args)
        if not isinstance(cursor, PC.AllocCursor):
            self.err(f"expected an AllocCursor, not {type(cursor)}")
        return cursor


class WindowStmtCursorA(StmtCursorA):
    def _cursor_call(self, alloc_pattern, all_args):
        cursor = super()._cursor_call(alloc_pattern, all_args)
        if not isinstance(cursor, PC.WindowStmtCursor):
            self.err(f"expected a WindowStmtCursor, not {type(cursor)}")
        return cursor


class ForOrIfCursorA(StmtCursorA):
    def _cursor_call(self, cursor_pat, all_args):
        # TODO: eliminate this redundancy with the ForCursorA code
        # allow for a special pattern short-hand, but otherwise
        # handle as expected for a normal statement cursor
        try:
            name, count = NameCountA()(cursor_pat, all_args)
            count = f"#{count}" if count is not None else ""
            cursor_pat = f"for {name} in _: _{count}"
        except:
            pass

        cursor = super()._cursor_call(cursor_pat, all_args)
        if not isinstance(cursor, (PC.ForCursor, PC.IfCursor)):
            self.err(f"expected a ForCursor or IfCursor, not {type(cursor)}")
        return cursor


class ArgCursorA(CursorArgumentProcessor):
    def _cursor_call(self, arg_pattern, all_args):
        if isinstance(arg_pattern, PC.ArgCursor):
            return arg_pattern
        elif isinstance(arg_pattern, str):
            name = arg_pattern
            proc = all_args["proc"]
            for arg in proc.args():
                if arg.name() == name:
                    return arg

            self.err(f"no argument {name} found")
        else:
            self.err("expected an ArgCursor or a string")


class ArgOrAllocCursorA(CursorArgumentProcessor):
    def _cursor_call(self, alloc_pattern, all_args):
        try:
            name, count = NameCountA()(alloc_pattern, all_args)
            count = f" #{count}" if count is not None else ""
            alloc_pattern = f"{name} : _{count}"
        except:
            pass

        cursor = alloc_pattern
        if not isinstance(cursor, (PC.AllocCursor, PC.ArgCursor)):
            proc = all_args["proc"]
            try:
                cursor = proc.find(alloc_pattern, call_depth=1)
            except:
                for arg in proc.args():
                    if arg.name() == name:
                        return arg
                self.err(
                    f"could not find a cursor matching {alloc_pattern}, nor an arg cursor with name {name}"
                )

        if not isinstance(cursor, (PC.AllocCursor, PC.ArgCursor)):
            self.err(
                f"expected either an AllocCursor or an ArgCursor, not {type(cursor)}"
            )
        return cursor


class ForCursorA(StmtCursorA):
    def _cursor_call(self, loop_pattern, all_args):
        # allow for a special pattern short-hand, but otherwise
        # handle as expected for a normal statement cursor
        try:
            name, count = NameCountA()(loop_pattern, all_args)
            count = f" #{count}" if count is not None else ""
            loop_pattern = f"for {name} in _: _{count}"
        except:
            pass

        cursor = super()._cursor_call(loop_pattern, all_args)
        if not isinstance(cursor, PC.ForCursor):
            self.err(f"expected a ForCursor, not {type(cursor)}")
        return cursor


class IfCursorA(StmtCursorA):
    def _cursor_call(self, if_pattern, all_args):
        cursor = super()._cursor_call(if_pattern, all_args)
        if not isinstance(cursor, PC.IfCursor):
            self.err(f"expected an IfCursor, not {type(cursor)}")
        return cursor


_name_name_count_re = r"^([a-zA-Z_]\w*)\s*([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$"


class NestedForCursorA(StmtCursorA):
    def _cursor_call(self, loops_pattern, all_args):
        if isinstance(loops_pattern, PC.ForCursor):
            cursor = loops_pattern
        elif isinstance(loops_pattern, PC.Cursor):
            self.err(f"expected a ForCursor, not {type(loops_pattern)}")
        elif isinstance(loops_pattern, str) and (
            match_result := re.search(_name_name_count_re, loops_pattern)
        ):
            out_name = match_result[1]
            in_name = match_result[2]
            count = f" #{match_result[3]}" if match_result[3] else ""
            pattern = f"for {out_name} in _:\n  for {in_name} in _: _{count}"
            cursor = super()._cursor_call(pattern, all_args)
        elif isinstance(loops_pattern, str):
            cursor = super()._cursor_call(loops_pattern, all_args)
            if not isinstance(cursor, PC.ForCursor):
                self.err(f"expected a ForCursor, not {type(cursor)}")
        else:
            self.err(
                "expected a ForCursor, pattern match string, "
                "or 'outer_loop inner_loop' shorthand"
            )

        if len(cursor.body()) != 1 or not isinstance(cursor.body()[0], PC.ForCursor):
            self.err(
                f"expected the body of the outer loop "
                f"to be a single loop, but it was a "
                f"{cursor.body()[0]}",
                ValueError,
            )

        return cursor


class AssignCursorA(StmtCursorA):
    def _cursor_call(self, stmt_pattern, all_args):
        cursor = super()._cursor_call(stmt_pattern, all_args)
        if not isinstance(cursor, PC.AssignCursor):
            self.err(f"expected an AssignCursor, not {type(cursor)}")
        return cursor


class AssignOrReduceCursorA(StmtCursorA):
    def _cursor_call(self, stmt_pattern, all_args):
        cursor = super()._cursor_call(stmt_pattern, all_args)
        if not isinstance(cursor, (PC.AssignCursor, PC.ReduceCursor)):
            self.err(f"expected an AssignCursor or ReduceCursor, not {type(cursor)}")
        return cursor


class CallCursorA(StmtCursorA):
    def _cursor_call(self, call_pattern, all_args):
        # allow for special pattern short-hands, but otherwise
        # handle as expected for a normal statement cursor
        if isinstance(call_pattern, Procedure):
            call_pattern = f"{call_pattern.name()}(_)"
        try:
            name, count = NameCountA()(call_pattern, all_args)
            count = f" #{count}" if count is not None else ""
            call_pattern = f"{name}(_){count}"
        except:
            pass

        cursor = super()._cursor_call(call_pattern, all_args)
        if not isinstance(cursor, PC.CallCursor):
            self.err(f"expected a CallCursor, not {type(cursor)}")
        return cursor


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# New Code Fragment Argument Processing


@dataclass
class FormattedExprStr:
    """
    Allows the user to provide a string with holes in it along with
    `ExprCursor`s to fill the holes. The object is designed as a wrapper to
    allow the user to give those inputs as an argument to scheduling
    operations. The object does not evaluate the expression, but merely
    holds the string and AST nodes the cursors point to until they are
    passed to the scheduling operation where they are extracted
    and evaluated to a new expression.
    """

    _expr_str: str
    _expr_holes: Tuple[LoopIR.expr]

    def __init__(self, expr_str: str, *expr_holes) -> None:
        if not isinstance(expr_str, str):
            raise TypeError("expr_str must be a string")
        self._expr_str = expr_str
        for cursor in expr_holes:
            if not isinstance(cursor, PC.ExprCursor):
                raise TypeError("Cursor provided to fill a hole must be a ExprCursor")
        self._expr_holes = tuple(cursor._impl._node for cursor in expr_holes)


class NewExprA(ArgumentProcessor):
    def __init__(self, cursor_arg, before=True):
        self.cursor_arg = cursor_arg
        self.before = before

    def _get_ctxt_stmt(self, all_args):
        cursor = all_args[self.cursor_arg]
        while isinstance(cursor, PC.ExprCursor):
            cursor = cursor.parent()

        # if we don't have a gap cursor, convert to a gap cursor
        if not isinstance(cursor, PC.GapCursor):
            cursor = cursor.before() if self.before else cursor.after()

        # TODO: improve parse_fragment to just take gaps
        return cursor.anchor()._impl._node

    def __call__(self, expr_str, all_args):
        expr_holes = None
        if isinstance(expr_str, int):
            return LoopIR.Const(expr_str, T.int, null_srcinfo())
        elif isinstance(expr_str, float):
            return LoopIR.Const(expr_str, T.R, null_srcinfo())
        elif isinstance(expr_str, bool):
            return LoopIR.Const(expr_str, T.bool, null_srcinfo())
        elif isinstance(expr_str, FormattedExprStr):
            expr_str, expr_holes = expr_str._expr_str, expr_str._expr_holes
        elif not isinstance(expr_str, str):
            self.err("expected a string")

        proc = all_args["proc"]
        ctxt_stmt = self._get_ctxt_stmt(all_args)

        expr = parse_fragment(
            proc._loopir_proc, expr_str, ctxt_stmt, expr_holes=expr_holes
        )
        return expr


# This is implemented as a workaround because the
# current PAST parser and PAST IR don't support windowing
# expressions.
class CustomWindowExprA(NewExprA):
    def __call__(self, expr_str, all_args):
        proc = all_args["proc"]
        ctxt_stmt = self._get_ctxt_stmt(all_args)

        # degenerate case of a scalar value
        if is_valid_name(expr_str):
            return expr_str, []

        # otherwise, we have multiple dimensions
        match = re.match(r"(\w+)\[([^\]]+)\]", expr_str)
        if not match:
            raise ValueError(
                f"expected windowing string of the form "
                f"'name[args]', but got '{expr_str}'"
            )
        buf_name, args = match.groups()
        if not is_valid_name(buf_name):
            raise ValueError(f"'{buf_name}' is not a valid name")

        loopir = proc._loopir_proc

        def parse_arg(a):
            match = re.match(r"\s*([^:]+)\s*:\s*([^:]+)\s*", a)
            if not match:
                # a.strip() to remove whitespace
                pt = parse_fragment(loopir, a.strip(), ctxt_stmt)
                return pt
            else:
                lo, hi = match.groups()
                lo = parse_fragment(loopir, lo, ctxt_stmt)
                hi = parse_fragment(loopir, hi, ctxt_stmt)
                return (lo, hi)

        args = [parse_arg(a) for a in args.split(",")]

        return buf_name, args


class NewExprOrCustomWindowExprA(NewExprA):
    def __call__(self, expr_str, all_args):
        try:
            return NewExprA(self.cursor_arg, self.before)(expr_str, all_args)
        except:
            return CustomWindowExprA(self.cursor_arg, self.before)(expr_str, all_args)


# --------------------------------------------------------------------------- #
#  - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#                       Atomic Scheduling Operations
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Basic Operations


@sched_op([])
def simplify(proc):
    """
    Simplify the code in the procedure body. Tries to reduce expressions
    to constants and eliminate dead branches and loops. Uses branch
    conditions to simplify expressions inside the branches.
    """
    # TODO: remove provenance handling from simplifier implementation
    return scheduling.DoSimplify(proc).result()


@sched_op([NameA])
def rename(proc, name):
    """
    Rename the procedure. Affects generated symbol names.

    args:
        name    - string
    """
    ir = proc._loopir_proc
    ir = ir.update(name=name)
    return Procedure(
        ir, _provenance_eq_Procedure=proc, _forward=ic.forward_identity(ir)
    )


@sched_op([InstrStrA, InstrStrA])
def make_instr(proc, c_instr, c_global=""):
    """
    Turn this procedure into an "instruction" using the provided macro-string

    args:
        c_instr  - string representing an instruction macro
        c_global - string representing global C code necessary for this instruction e.g. includes
    """
    ir = proc._loopir_proc
    instr = LoopIR.instr(c_instr=c_instr, c_global=c_global)
    ir = ir.update(instr=instr)
    return Procedure(
        ir,
        _provenance_eq_Procedure=proc,
        _forward=ic.forward_identity(ir),
    )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# General Statement and Expression Operations


@sched_op([GapCursorA])
def insert_pass(proc, gap_cursor):
    """
    Insert a `pass` statement at the indicated position.

    args:
        gap_cursor  - where to insert the new `pass` statement

    rewrite:
        `s1 ; s2` <--- gap_cursor pointed at the semi-colon
        -->
        `s1 ; pass ; s2`
    """
    ir, fwd = scheduling.DoInsertPass(gap_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([GapCursorA, ProcA, ListA(NewExprOrCustomWindowExprA("gap_cursor"))])
def insert_noop_call(proc, gap_cursor, instr, args):
    if len(args) != len(instr._loopir_proc.args):
        raise TypeError("Function argument count mismatch")

    ir, fwd = scheduling.DoInsertNoopCall(gap_cursor._impl, instr._loopir_proc, args)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([])
def delete_pass(proc):
    """
    DEPRECATED (to be replaced by a more general operation)

    Delete all `pass` statements in the procedure.
    """
    ir, fwd = scheduling.DoDeletePass(proc)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA(block_size=2)])
def reorder_stmts(proc, block_cursor):
    """
    swap the order of two statements within a block.

    args:
        block_cursor    - a cursor to a two statement block to reorder

    rewrite:
        `s1 ; s2`  <-- block_cursor
        -->
        `s2 ; s1`
    """
    s1 = block_cursor[0]._impl
    s2 = block_cursor[1]._impl

    ir, fwd = scheduling.DoReorderStmt(s1, s2)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForCursorA])
def parallelize_loop(proc, loop_cursor):
    loop = loop_cursor._impl

    ir, fwd = scheduling.DoParallelizeLoop(loop)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ExprCursorA(many=True)])
def commute_expr(proc, expr_cursors):
    """
    commute the binary operation of '+' and '*'.

    args:
        expr_cursors - a list of cursors to the binary operation

    rewrite:
        `a * b` <-- expr_cursor
        -->
        `b * a`

        or

        `a + b` <-- expr_cursor
        -->
        `b + a`
    """

    exprs = [ec._impl for ec in expr_cursors]
    for e in exprs:
        if not isinstance(e._node, LoopIR.BinOp) or (
            e._node.op != "+" and e._node.op != "*"
        ):
            raise TypeError(f"only '+' or '*' can commute, got {e._node.op}")
    if any(not e._node.type.is_numeric() for e in exprs):
        raise TypeError(
            "only numeric (not index or size) expressions "
            "can commute by commute_expr()"
        )

    ir, fwd = scheduling.DoCommuteExpr(exprs)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ExprCursorA])
def left_reassociate_expr(proc, expr):
    """
    Reassociate the binary operations of '+' and '*'.

    args:
        expr - the expression to reassociate

    rewrite:
        a + (b + c)
            ->
        (a + b) + c
    """
    expr = expr._impl
    if not isinstance(expr._node, LoopIR.BinOp) or (
        expr._node.op != "+" and expr._node.op != "*"
    ):
        raise TypeError(f"Only '+' or '*' can be reassociated, got {expr._node.op}")
    if (
        not isinstance(expr._node.rhs, LoopIR.BinOp)
        or expr._node.rhs.op != expr._node.op
    ):
        raise TypeError(
            f"The rhs of the expression must be the same binary operation as the expression ({expr._node.op})"
        )
    ir, fwd = scheduling.DoLeftReassociateExpr(expr)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ExprCursorA, NewExprA("expr_cursor")])
def rewrite_expr(proc, expr_cursor, new_expr):
    """
    Replaces [expr_cursor] with [new_expr] if the two are equivalent
    in the context.

    rewrite:
        `s`
        ->
        `s[ expr_cursor -> new_expr]`
    """
    ir, fwd = scheduling.DoRewriteExpr(expr_cursor._impl, new_expr)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ListOrElemA(ExprCursorA), NameA])
def bind_expr(proc, expr_cursors, new_name):
    """
    Bind some numeric/data-value type expression(s) into a new intermediate,
    scalar-sized buffer. It will fail if not all of the provided expressions
    can be bound safely. The precision of the new allocation is that of the
    bound expression.

    args:
        expr_cursors    - a list of cursors to multiple instances of the
                          same expression
        new_name        - a string to name the new buffer

    rewrite:
        bind_expr(..., '32.0 * x[i]', 'b')
        `a = 32.0 * x[i] + 4.0`
        -->
        `b : R`
        `b = 32.0 * x[i]`
        `a = b + 4.0`
    """
    exprs = [ec._impl for ec in expr_cursors]
    if any(not e._node.type.is_numeric() for e in exprs):
        raise TypeError(
            "only numeric (not index or size) expressions "
            "can be bound by bind_expr()"
        )

    ir, fwd = scheduling.DoBindExpr(new_name, exprs)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Sub-procedure Operations


@sched_op([BlockCursorA, NameA, BoolA])
def extract_subproc(proc, block, subproc_name, include_asserts=True):
    """
    Extract a block as a subprocedure with the name `subproc_name`.

    args:
        block           - the block to extract as a subprocedure.
        subproc_name    - the name of the new subprocedure.
        include_asserts - whether to include asserts about the parameters
                          that can be inferred from the parent.

    returns:
        a tuple (proc, subproc).

    rewrite:
        extract_subproc(..., "sub_foo", "for i in _:_")
        ```
        def foo(N: size, M: size, K: size, x: R[N, K + M]):
            assert N >= 8
            for i in seq(0, 8):
                x[i, 0] += 2.0
        ```
        -->
        ```
        def foo(N: size, M: size, K: size, x: R[N, K + M]):
            assert N >= 8
            sub_foo(N, M, K, x)
        def sub_foo(N: size, M: size, K: size, x: R[N, K + M]):
            assert N >= 8
            for i in seq(0, 8):
                x[i, 0] += 2.0
        ```

    """

    ir, fwd, subproc_ir = scheduling.DoExtractSubproc(
        block._impl, subproc_name, include_asserts
    )
    proc = Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)
    subproc = Procedure(subproc_ir)
    return proc, subproc


@sched_op([CallCursorA])
def inline(proc, call_cursor):
    """
    Inline a sub-procedure call.

    args:
        call_cursor     - Cursor or pattern pointing to a Call statement
                          whose body we want to inline
    """
    ir, fwd = scheduling.DoInline(call_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA, ProcA, BoolA])
def replace(proc, block_cursor, subproc, quiet=False):
    """
    Attempt to match the supplied `subproc` against the supplied
    statement block.  If the two can be unified, then replace the block
    of statements with a call to `subproc`.

    args:
        block_cursor    - Cursor or pattern pointing to block of statements
        subproc         - Procedure object to replace this block with a
                          call to
        quiet           - (bool) control how much this operation prints
                          out debug info
    """
    try:
        ir, fwd = DoReplace(subproc._loopir_proc, block_cursor._impl)
        return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)
    except UnificationError:
        if quiet:
            raise
        print(f"Failed to unify the following:\nSubproc:\n{subproc}\nStatements:")
        [print(sc._impl._node) for sc in block_cursor]
        raise


@sched_op([CallCursorA, ProcA])
def call_eqv(proc, call_cursor, eqv_proc):
    """
    Swap out the indicated call with a call to `eqv_proc` instead.
    This operation can only be performed if the current procedures being
    called and `eqv_proc` are equivalent due to being scheduled
    from the same procedure (or one scheduled from the other).

    args:
        call_cursor     - Cursor or pattern pointing to a Call statement
        eqv_proc        - Procedure object for the procedure to be
                          substituted in

    rewrite:
        `orig_proc(...)`    ->    `eqv_proc(...)`
    """
    call_stmt = call_cursor._impl
    new_loopir = eqv_proc._loopir_proc

    ir, fwd, cfg = scheduling.DoCallSwap(call_stmt, new_loopir)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd, _mod_config=cfg)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Precision, Memory and Window Setting Operations


@sched_op([ArgOrAllocCursorA, TypeAbbrevA])
def set_precision(proc, cursor, typ):
    """
    Set the precision annotation on a given buffer to the provided
    base-type precision.

    args:
        name    - string w/ optional count, e.g. "x" or "x #3"
        typ     - string representing base data type

    rewrite:
        `name : _[...]    ->    name : typ[...]`
    """
    ir, fwd = scheduling.DoSetTypAndMem(cursor._impl, basetyp=typ)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ArgCursorA, BoolA])
def set_window(proc, cursor, is_window=True):
    """
    Set the annotation on a given buffer to indicate that it should be
    a window (True) or should not be a window (False)

    args:
        name        - string w/ optional count, e.g. "x" or "x #3"
        is_window   - boolean representing whether a buffer is a window

    rewrite when is_window = True:
        `name : R[...]    ->    name : [R][...]`
    """
    ir, fwd = scheduling.DoSetTypAndMem(cursor._impl, win=is_window)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


# TODO support SpecialWindow for arg cursor (but not alloc)
@sched_op([ArgOrAllocCursorA, MemoryA])
def set_memory(proc, cursor, memory_type):
    """
    Set the memory annotation on a given buffer to the provided memory.

    args:
        name    - string w/ optional count, e.g. "x" or "x #3"
        mem     - new Memory object

    rewrite:
        `name : _ @ _    ->    name : _ @ mem`
    """
    ir, fwd = scheduling.DoSetTypAndMem(cursor._impl, mem=memory_type)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Configuration Modifying Operations


@sched_op([ExprCursorA, ConfigA, ConfigFieldA])
def bind_config(proc, var_cursor, config, field):
    """
    extract a control-value expression and write it into some
    designated field of a config

    args:
        var_cursor  - cursor or pattern pointing at the expression to
                      be bound
        config      - config object to be written into
        field       - (string) the field of `config` to be written to

    rewrite:
        Let `s[ e ]` mean a statement with control expression `e` occurring
        within it.  Then,
        `s[ e ]    ->    config.field = e ; s[ config.field ]`
    """
    e = var_cursor._impl._node
    cfg_f_type = config.lookup_type(field)

    if not isinstance(e, LoopIR.Read):
        raise TypeError("expected a cursor to a single variable Read")

    if not (e.type.is_real_scalar() and len(e.idx) == 0) and not e.type.is_bool():
        raise TypeError(
            f"cannot bind non-real-scalar non-boolean value {e} to configuration states, since index and size expressions may depend on loop iteration"
        )

    if e.type != cfg_f_type:
        raise TypeError(
            f"expected type of expression to bind ({e.type}) "
            f"to match type of Config variable ({cfg_f_type})"
        )

    ir, fwd, cfg = scheduling.DoBindConfig(config, field, var_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd, _mod_config=cfg)


@sched_op([StmtCursorA])
def delete_config(proc, stmt_cursor):
    """
    delete a statement that writes to some config.field

    args:
        stmt_cursor - cursor or pattern pointing at the statement to
                      be deleted

    rewrite:
        `s1 ; config.field = _ ; s3    ->    s1 ; s3`
    """
    ir, fwd, cfg = scheduling.DoDeleteConfig(proc._root(), stmt_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd, _mod_config=cfg)


@sched_op([GapCursorA, ConfigA, ConfigFieldA, NewExprA("gap_cursor")])
def write_config(proc, gap_cursor, config, field, rhs):
    """
    insert a statement that writes a desired value to some config.field

    args:
        gap_cursor  - cursor pointing to where the new write statement
                      should be inserted
        config      - config object to be written into
        field       - (string) the field of `config` to be written to
        rhs         - (string) the expression to write into the field

    rewrite:
        `s1 ; s3    ->    s1 ; config.field = new_expr ; s3`
    """

    # TODO: just have scheduling pass take a gap cursor directly
    stmtc = gap_cursor.anchor()
    before = gap_cursor.type() == ic.GapType.Before

    if not isinstance(rhs, (LoopIR.Read, LoopIR.StrideExpr, LoopIR.Const)):
        raise TypeError("expected the rhs to be read, stride expression, or constant")

    if isinstance(rhs, LoopIR.Read):
        if (
            not (rhs.type.is_real_scalar() and len(rhs.idx) == 0)
            and not rhs.type.is_bool()
        ):
            raise TypeError(
                f"cannot write non-real-scalar non-boolean value {rhs} to configuration states, since index and size expressions may depend on loop iteration"
            )

    stmt = stmtc._impl
    ir, fwd, cfg = scheduling.DoConfigWrite(stmt, config, field, rhs, before=before)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd, _mod_config=cfg)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory and Windowing-oriented Operations


@sched_op([AllocCursorA, IntA, NewExprA("buf_cursor"), NewExprA("buf_cursor"), BoolA])
def resize_dim(proc, buf_cursor, dim_idx, size, offset, fold: bool = False):
    """
    Resizes the [dim_idx]-th dimension of buffer [buf_cursor] to [size]. The [offset]
    specifies how to adjust the indices relative to the old buffer.

    If [fold] is True, we will try to perform a circular buffer optimization, which
    ignores the offset argument.

    Fails if there are any accesses to the [dim_idx]-th dimension outside of the
    (offset, offset + size) range.

    args:
        buf_cursor      - cursor pointing to the Alloc
        dim_idx         - which dimension to shrink
        size            - new size as a positive expression
        offset          - offset for adjusting the buffer access

    rewrite:
        `x : T[n, ...] ; s`
          ->
        `x : T[size, ...] ; s[ x[idx, ...] -> x[idx - offset, ...] ]`

    rewrite (if fold = True):
        `x : T ; s`
          ->
        `x : T ; s[ x[i] -> x[i % size] ]`

    checks:
        The provided dimension size is checked for positivity and the
        provided indexing expression is checked to make sure it is in-bounds
    """
    stmt_c = buf_cursor._impl
    assert dim_idx >= 0, "Dimension index must be non-negative"

    if fold:
        # Circular buffer folding
        assert isinstance(size, LoopIR.Const) and size.val > 0
        size = size.val
        buf_s = buf_cursor._impl
        ir, fwd = scheduling.DoFoldBuffer(buf_s, dim_idx, size)
        return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)
    else:
        # Normal resize operation
        ir, fwd = scheduling.DoResizeDim(stmt_c, dim_idx, size, offset)
        return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, NewExprA("buf_cursor"), NewExprA("buf_cursor")])
def expand_dim(proc, buf_cursor, alloc_dim, indexing_expr):
    """
    TODO: rename this...expand_dim sounds like its increasing the size
    of a dimension. It should be more like add_dim.

    expand the number of dimensions of a buffer variable (`buf_cursor`).
    After expansion, the existing code will initially only use particular
    entries of the new dimension, chosen by the provided `indexing_expr`

    args:
        buf_cursor      - cursor pointing to the Alloc to expand
        alloc_dim       - (string) an expression for the size
                          of the new buffer dimension.
        indexing_expr   - (string) an expression to index the newly
                          created dimension with.

    rewrite:
        `x : T[...] ; s`
          ->
        `x : T[alloc_dim, ...] ; s[ x[...] -> x[indexing_expr, ...] ]`
    checks:
        The provided dimension size is checked for positivity and the
        provided indexing expression is checked to make sure it is in-bounds
    """
    stmt_c = buf_cursor._impl
    ir, fwd = scheduling.DoExpandDim(stmt_c, alloc_dim, indexing_expr)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, ListA(IntA)])
def rearrange_dim(proc, buf_cursor, permute_vector):
    """
    Rearranges the dimensions of the indicated buffer allocation according
    to the supplied permutation (`permute_vector`).

    args:
        buf_cursor      - cursor pointing to an Alloc statement
                          for an N-dimensional array
        permute_vector  - a permutation of the integers (0,1,...,N-1)

    rewrite:
        (with permute_vector = [2,0,1])
        `x : T[N,M,K]` -> `x : T[K,N,M]`
    """
    stmt = buf_cursor._impl

    N = len(stmt._node.type.hi)
    if list(range(0, N)) != sorted(permute_vector):
        raise ValueError(
            f"permute_vector argument ({permute_vector}) "
            f"was not a permutation of {set(range(0, N))}"
        )

    ir, fwd = scheduling.DoRearrangeDim(stmt, permute_vector)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, IntA, PosIntA])
def divide_dim(proc, alloc_cursor, dim_idx, quotient):
    """
    Divide the `dim_idx`-th buffer dimension into a higher-order
    and lower-order dimensions, where the lower-order dimension is given
    by the constant integer `quotient`.

    args:
        alloc_cursor    - cursor to the allocation to divide a dimension of
        dim_idx         - the index of the dimension to divide
        quotient        - (positive int) the factor to divide by

    rewrite:
        divide_dim(..., 1, 4)
        `x : R[n, 12, m]`
        `x[i, j, k] = ...`
        ->
        `x : R[n, 3, 4, m]`
        `x[i, j / 4, j % 4, k] = ...`
    """
    stmt = alloc_cursor._impl
    if not (0 <= dim_idx < len(stmt._node.type.shape())):
        raise ValueError(f"Cannot divide out-of-bounds dimension index {dim_idx}")

    ir, fwd = scheduling.DoDivideDim(stmt, dim_idx, quotient)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, IntA, IntA])
def mult_dim(proc, alloc_cursor, hi_dim_idx, lo_dim_idx):
    """
    Mutiply the `hi_dim_idx`-th buffer dimension by the `low_dim_idx`-th
    buffer dimension to create a single buffer dimension.  This operation
    is only permitted when the `lo_dim_idx`-th dimension is a constant
    integer value.

    args:
        alloc_cursor    - cursor to the allocation to divide a dimension of
        hi_dim_idx      - the index of the higher order dimension to multiply
        lo_dim_idx      - the index of the lower order dimension to multiply

    rewrite:
        mult_dim(..., 0, 2)
        `x : R[n, m, 4]`
        `x[i, j, k] = ...`
        ->
        `x : R[4*n, m]`
        `x[4*i + k, j] = ...`
    """
    stmt = alloc_cursor._impl
    for dim_idx in [hi_dim_idx, lo_dim_idx]:
        if not (0 <= dim_idx < len(stmt._node.type.shape())):
            raise ValueError(f"Cannot multiply out-of-bounds dimension index {dim_idx}")
    if hi_dim_idx == lo_dim_idx:
        raise ValueError(f"Cannot multiply dimension {hi_dim_idx} by itself")

    ir, fwd = scheduling.DoMultiplyDim(stmt, hi_dim_idx, lo_dim_idx)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, IntA])
def unroll_buffer(proc, alloc_cursor, dimension):
    """
    Unroll the buffer allocation with constant dimension.

    args:
        alloc_cursor  - cursor to the buffer with constant dimension
        dimension     - dimension to unroll

    rewrite:
        `buf : T[2]` <- alloc_cursor
        `...`
        ->
        `buf_0 : T`
        `buf_1 : T`
        `...`
    """

    ir, fwd = scheduling.DoUnrollBuffer(alloc_cursor._impl, dimension)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, PosIntA])
def lift_alloc(proc, alloc_cursor, n_lifts=1):
    """
    Lift a buffer allocation up and out of various Loops / If-statements.

    args:
        alloc_cursor    - cursor to the allocation to lift up
        n_lifts         - number of times to try to move the allocation up

    rewrite:
        `for i in _:`
        `    buf : T` <- alloc_cursor
        `    ...`
        ->
        `buf : T`
        `for i in _:`
        `    ...`
    """
    stmt = alloc_cursor._impl

    ir, fwd = scheduling.DoLiftAllocSimple(stmt, n_lifts)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA])
def sink_alloc(proc, alloc_cursor):
    """
    Sinks a buffer allocation into a scope (for loop/if statement). This scope
    must come immediately after the alloc statemenet. Requires that the
    alloc_cursor occurs right before the scope_cursor

    args:
        alloc_cursor    - cursor to the allocation to sink up

    rewrite:
        `buf : T`       <- alloc_cursor
        `for i in _:`
            `    ...`
        ->
        `for i in _:`
        `    buf : T`
        `    ...`
    """

    scope_cursor = alloc_cursor.next()
    if not isinstance(scope_cursor._impl._node, (LoopIR.If, LoopIR.For)):
        raise ValueError(
            f"Cannot sink alloc because the statement after the allocation is not a loop or if statement, it is {scope_cursor._impl._node}"
        )

    ir, fwd = scheduling.DoSinkAlloc(alloc_cursor._impl, scope_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, PosIntA, EnumA(["row", "col"]), OptionalA(PosIntA), BoolA])
def autolift_alloc(
    proc, alloc_cursor, n_lifts=1, mode="row", size=None, keep_dims=False
):
    """
    Lift a buffer allocation up and out of various Loops / If-statements.

    Has some additional special legacy behavior.  Use lift_alloc instead for
    all new code.

    args:
        alloc_cursor    - cursor to the allocation to lift up
        n_lifts         - number of times to try to move the allocation up
        mode            - whether to expand the buffer's dimensions
                          on the inner or outer position
        size            - dimension extents to expand to?
        keep_dims       - ???

    rewrite:
        `for i in _:`
        `    buf : T` <- alloc_cursor
        `    ...`
        ->
        `buf : T`
        `for i in _:`
        `    ...`
    """
    stmt = alloc_cursor._impl

    return scheduling.DoLiftAlloc(proc, stmt, n_lifts, mode, size, keep_dims).result()


@sched_op([AllocCursorA])
def delete_buffer(proc, buf_cursor):
    """
    Deletes [buf_cursor] if it is unused.
    """
    buf_s = buf_cursor._impl
    ir, fwd = scheduling.DoDeleteBuffer(buf_s)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AllocCursorA, AllocCursorA])
def reuse_buffer(proc, buf_cursor, replace_cursor):
    """
    reuse existing buffer (`buf_cursor`) instead of
    allocating a new buffer (`replace_cursor`).

    args:
        buf_cursor      - cursor pointing to the Alloc to reuse
        replace_cursor  - cursor pointing to the Alloc to eliminate

    rewrite:
        `x : T ; ... ; y : T ; s`
          ->
        `x : T ; ... ; s[ y -> x ]`
    checks:
        Can only be performed if the variable `x` is dead at the statement
        `y : T`.
    """
    buf_s = buf_cursor._impl
    rep_s = replace_cursor._impl
    ir, fwd = scheduling.DoReuseBuffer(buf_s, rep_s)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([WindowStmtCursorA])
def inline_window(proc, winstmt_cursor):
    """
    Eliminate use of a window by inlining its definition and expanding
    it at all use-sites

    args:
        winstmt_cursor  - cursor pointing to the WindowStmt to inline

    rewrite:
        `y = x[...] ; s` -> `s[ y -> x[...] ]`
    """
    stmt = winstmt_cursor._impl
    ir, fwd = scheduling.DoInlineWindow(stmt)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA, CustomWindowExprA("block_cursor"), NameA, BoolA])
def stage_mem(proc, block_cursor, win_expr, new_buf_name, accum=False):
    """
    Stage the window of memory specified by `win_expr` into a new buffer
    before the indicated code block and move the memory back after the
    indicated code block.  If code analysis allows one to omit either
    the load or store between the original buffer and staging buffer, then
    the load/store loops/statements will be omitted.

    If code analysis determines determines that `win_expr` accesses
    out-of-bounds locations of the buffer, it will generate loop nests
    for the load/store stages corresponding to that window, but will add
    guards within the inner loop to ensure that all accesses to the buffer
    are within the buffer's bounds.

    In the event that the indicated block of code strictly reduces into
    the specified window, then the optional argument `accum` can be set
    to initialize the staging memory to zero, accumulate into it, and
    then accumulate that result back to the original buffer, rather than
    loading and storing.  This is especially valuable when one's target
    platform can more easily zero out memory and thereby either
    reduce memory traffic outright, or at least improve locality of access.

    args:
        block_cursor    - the block of statements to stage around
        win_expr        - (string) of the form `name[pt_or_slice*]`
                          e.g. 'x[32, i:i+4]'
                          In this case `x` should be accessed in the
                          block, but only at locations
                          (32, i), (32, i+1), (32, i+2), or (32, i+3)
        new_buf_name    - the name of the newly created staging buffer
        accum           - (optional, bool) see above

    rewrite:
        stage_mem(..., 'x[0:n,j-1:j]', 'xtmp')
        `for i in seq(0,n-1):`
        `    x[i,j] = 2 * x[i+1,j-1]`
        -->
        `for k0 in seq(0,n):`
        `    for k1 in seq(0,2):`
        `        xtmp[k0,k1] = x[k0,j-1+k1]`
        `for i in seq(0,n-1):`
        `    xtmp[i,j-(j-1)] = 2 * xtmp[i+1,(j-1)-(j-1)]`
        `for k0 in seq(0,n):`
        `    for k1 in seq(0,2):`
        `        x[k0,j-1+k1] = xtmp[k0,k1]`

    """
    buf_name, w_exprs = win_expr
    ir, fwd = scheduling.DoStageMem(
        block_cursor._impl, buf_name, w_exprs, new_buf_name, use_accum_zero=accum
    )
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop and Guard Rewriting


@sched_op([ForCursorA, NewExprA("loop_cursor"), PosIntA, ListA(NameA, length=2)])
def divide_with_recompute(proc, loop_cursor, outer_hi, outer_stride, new_iters):
    """
    Divides a loop into the provided [outer_hi] by [outer_stride] dimensions,
    and then adds extra compute so that the inner loop will fully cover the
    original loop's range.

    rewrite:
        `for i in seq(0, hi):`
        `    s`
            ->
        `for io in seq(0, outer_hi):`
        `    for ii in seq(0, outer_stride + (hi - outer_hi * outer_stride)):`
        `        s[ i -> outer_stride * io + ii ]`
    """
    ir, fwd = scheduling.DoDivideWithRecompute(
        loop_cursor._impl, outer_hi, outer_stride, new_iters[0], new_iters[1]
    )
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op(
    [
        ForCursorA,
        PosIntA,
        ListA(NameA, length=2),
        EnumA(["cut", "guard", "cut_and_guard"]),
        BoolA,
    ]
)
def divide_loop(proc, loop_cursor, div_const, new_iters, tail="guard", perfect=False):
    """
    Divide a loop into an outer and inner loop, where the inner loop
    iterates over the range 0 to `div_const`.

    Old Name: In Halide and TVM, this was called "split"

    args:
        loop_cursor     - cursor pointing to the loop to split ;
                          can also be specified using the special shorthands
                          pattern: <loop-iterator-name>
                               or: <loop-iterator-name> #<int>
        div_const       - integer > 1 specifying what to "divide by"
        new_iters       - list or tuple of two strings specifying the new
                          outer and inner iteration variable names
        tail (opt)      - specifies the strategy for handling the "remainder"
                          of the loop division (called the tail of the loop).
                          value can be "cut", "guard", or "cut_and_guard".
                          Default value: "guard"
        perfect (opt)   - Boolean (default False) that can be set to true
                          to assert that you know the remainder will always
                          be zero (i.e. there is no tail).  You will get an
                          error if the compiler cannot verify this fact itself.

    rewrite:
        divide(..., div_const=q, new_iters=['hi','lo'], tail='cut')
        `for i in seq(0,e):`
        `    s`
            ->
        `for hi in seq(0,e / q):`
        `    for lo in seq(0, q):`
        `        s[ i -> q*hi + lo ]`
        `for lo in seq(0,e - q * (e / q)):`
        `    s[ i -> q * (e / q) + lo ]
    """

    stmt = loop_cursor._impl

    ir, fwd = scheduling.DoDivideLoop(
        stmt,
        quot=div_const,
        outer_iter=new_iters[0],
        inner_iter=new_iters[1],
        tail=tail,
        perfect=perfect,
    )
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([NestedForCursorA, NameA])
def mult_loops(proc, nested_loops, new_iter_name):
    """
    Perform the inverse operation to `divide_loop`.  Take two loops,
    the innermost of which has a literal bound. (e.g. 5, 8, etc.) and
    replace them by a single loop that iterates over the product of their
    iteration spaces (e.g. 5*n, 8*n, etc.)

    args:
        nested_loops    - cursor pointing to a loop whose body is also a loop
        new_iter_name   - string with name of the new iteration variable

    rewrite:
        `for i in seq(0,e):`
        `    for j in seq(0,c):`    # c is a literal integer
        `        s`
        ->
        `for k in seq(0,e*c):`      # k is new_iter_name
        `    s[ i -> k/c, j -> k%c ]`
    """
    ir, fwd = scheduling.DoProductLoop(nested_loops._impl, new_iter_name)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForCursorA, ForCursorA])
def join_loops(proc, loop1_cursor, loop2_cursor):
    """
    Joins two loops with identical bodies and consecutive iteration spaces
    into one loop.

    args:
        loop1_cursor     - cursor pointing to the first loop
        loop2_cursor     - cursor pointing to the second loop

    rewrite:
        `for i in seq(lo, mid):`
        `    s`
        `for i in seq(mid, hi):`
        `    s`
        ->
        `for i in seq(lo, hi):`
        `    s`
    """
    ir, fwd = scheduling.DoJoinLoops(loop1_cursor._impl, loop2_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForCursorA, NewExprA("loop_cursor")])
def cut_loop(proc, loop_cursor, cut_point):
    """
    Cut a loop into two loops.

    First loop iterates from `lo` to `cut_point` and
    the second iterating from `cut_point` to `hi`.

    We must have:
        `lo` <= `cut_point` <= `hi`

    args:
        loop_cursor     - cursor pointing to the loop to split
        cut_point       - expression representing iteration to cut at

    rewrite:
        `for i in seq(0,n):`
        `    s`
        ->
        `for i in seq(0,cut):`
        `    s`
        `for i in seq(cut, n):`
        `    s`
    """
    ir, fwd = scheduling.DoCutLoop(loop_cursor._impl, cut_point)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForCursorA, NewExprA("loop_cursor")])
def shift_loop(proc, loop_cursor, new_lo):
    """
    Shift a loop iterations so that now it starts at `new_lo`

    We must have:
        0 <= `new_lo`

    args:
        loop_cursor     - cursor pointing to the loop to shift
        new_lo          - expression representing new loop lo

    rewrite:
        `for i in seq(m,n):`
        `    s(i)`
        ->
        `for i in seq(new_lo, new_lo + n - m):`
        `    s(i + (m - new_lo))`
    """
    ir, fwd = scheduling.DoShiftLoop(loop_cursor._impl, new_lo)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([NestedForCursorA])
def reorder_loops(proc, nested_loops):
    """
    Reorder two loops that are directly nested with each other.
    This is the primitive loop reordering operation, out of which
    other reordering operations can be built.

    args:
        nested_loops    - cursor pointing to the outer loop of the
                          two loops to reorder; a pattern to find said
                          cursor with; or a 'name name' shorthand where
                          the first name is the iteration variable of the
                          outer loop and the second name is the iteration
                          variable of the inner loop.  An optional '#int'
                          can be added to the end of this shorthand to
                          specify which match you want,

    rewrite:
        `for outer in _:`
        `    for inner in _:`
        `        s`
            ->
        `for inner in _:`
        `    for outer in _:`
        `        s`
    """

    stmt_c = nested_loops._impl
    if len(stmt_c.body()) != 1 or not isinstance(stmt_c.body()[0]._node, LoopIR.For):
        raise ValueError(f"expected loop directly inside of {stmt_c._node.iter} loop")

    ir, fwd = scheduling.DoLiftScope(stmt_c.body()[0])
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA(block_size=2)])
def merge_writes(proc, block_cursor):
    """
    Merge consecutive assign and reduce statement into a single statement.
    Handles all 4 cases of (assign, reduce) x (reduce, assign).

    args:
        block_cursor          - cursor pointing to the block of two consecutive
                                assign/reduce statement.

    rewrite:
        `a = b`
        `a = c`
            ->
        `a = c`
        ----------------------
        `a += b`
        `a = c`
            ->
        `a = c`
        ----------------------
        `a = b`
        `a += c`
            ->
        `a = b + c`
        ----------------------
        `a += b`
        `a += c`
            ->
        `a += b + c`
        ----------------------

    """
    stmt1 = block_cursor[0]._impl._node
    stmt2 = block_cursor[1]._impl._node

    # TODO: We should seriously consider how to improve Scheduling errors in general
    if not isinstance(stmt1, (LoopIR.Assign, LoopIR.Reduce)) or not isinstance(
        stmt2, (LoopIR.Assign, LoopIR.Reduce)
    ):
        raise ValueError(
            f"expected two consecutive assign/reduce statements, "
            f"got {type(stmt1)} and {type(stmt2)} instead."
        )
    if stmt1.name != stmt2.name or stmt1.type != stmt2.type:
        raise ValueError(
            "expected the two statements' left hand sides to have the same name & type"
        )
    if not stmt1.rhs.type.is_numeric() or not stmt2.rhs.type.is_numeric():
        raise ValueError(
            "expected the two statements' right hand sides to have numeric types."
        )

    ir, fwd = scheduling.DoMergeWrites(block_cursor[0]._impl, block_cursor[1]._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AssignOrReduceCursorA])
def split_write(proc, stmt):
    """
    Split a reduce or assign statement with an addition on the RHS into two
    writes.

    This operation is the opposite of the last two cases of `merge_writes`.

    args:
        stmt    - cursor pointing to the assign/reduce statement.

    rewrite:
        `a = b + c`
            ->
        `a = b`
        `a += c`
        ----------------------
        `a += b + c`
            ->
        `a += b`
        `a += c`
        ----------------------

    forwarding:
        - cursors to the statement and any cursors within the statement gets invalidated.
        - blocks containing the statement will forward to a new block containing the resulting block.
    """
    ir, fwd = scheduling.DoSplitWrite(stmt._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AssignCursorA])
def fold_into_reduce(proc, assign):
    """
    Fold an assignment into a reduction if the rhs is an addition
    whose lhs is equal to the lhs of the assignment.

    args:
        assign: a cursor pointing to the assignment to fold.

    rewrite:
        a = a + (expr)
            ->
        a += expr
    """
    ir, fwd = scheduling.DoFoldIntoReduce(assign._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([AssignCursorA])
def inline_assign(proc, alloc_cursor):
    """
    Inlines [alloc_cursor] into any statements where it is used after this assignment.

    rewrite:
        `x = y`
        `s`
        ->
        `s[ x -> y ]`
    """
    s = alloc_cursor._impl

    if not isinstance(s._node, LoopIR.Assign):
        raise ValueError(
            f"Expected the statement to be an assign, instead got {stmt1._node}"
        )

    ir, fwd = scheduling.DoInlineAssign(s)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA(block_size=2)])
def lift_reduce_constant(proc, block_cursor):
    """
    Lift a constant scaling factor out of a loop.

    args:
        block_cursor       - block of size 2 containing the zero assignment and the for loop to lift the constant out of

    rewrite:
        `x = 0.0`
        `for i in _:`
        `    x += c * y[i]`
        ->
        `x = 0.0`
        `for i in _:`
        `    x += y[i]`
        `x = c * x`
    """
    stmt_c = block_cursor[0]._impl
    loop_c = block_cursor[1]._impl

    ir, fwd = scheduling.DoLiftConstant(stmt_c, loop_c)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([GapCursorA, PosIntA, BoolA])
def fission(proc, gap_cursor, n_lifts=1, unsafe_disable_checks=False):
    """
    fission apart the For and If statements wrapped around
    this block of statements into two copies; the first containing all
    statements before the cursor, and the second all statements after the
    cursor.

    args:
        gap_cursor          - a cursor pointing to the point in the
                              statement block that we want to fission at.
        n_lifts (optional)  - number of levels to fission upwards (default=1)

    rewrite:
        `for i in _:`
        `    s1`
        `      ` <- gap
        `    s2`
            ->
        `for i in _:`
        `    s1`
        `for i in _:`
        `    s2`
    """

    if gap_cursor.type() == ic.GapType.Before:
        stmt = gap_cursor.anchor().prev()
    else:
        stmt = gap_cursor.anchor()

    if not stmt or not stmt.next():
        raise ValueError(
            "expected cursor to point to a gap between statements, not at an edge"
        )

    ir, fwd = scheduling.DoFissionAfterSimple(
        stmt._impl, n_lifts, unsafe_disable_checks
    )
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([GapCursorA, PosIntA])
def autofission(proc, gap_cursor, n_lifts=1):
    """
    Split the enclosing For and If statements wrapped around
    this block of statements at the indicated point.

    If doing so splits a loop, this version of fission attempts
    to remove those loops as well.

    args:
        gap_cursor          - a cursor pointing to the point in the
                              statement block that we want to fission at.
        n_lifts (optional)  - number of levels to fission upwards (default=1)

    rewrite:
        `for i in _:`
        `    s1`
        `      ` <- gap
        `    s2`
            ->
        `for i in _:`
        `    s1`
        `for i in _:`
        `    s2`
    """

    if gap_cursor.type() == ic.GapType.Before:
        stmt = gap_cursor.anchor().prev()
    else:
        stmt = gap_cursor.anchor()

    if not stmt or not stmt.next():
        raise ValueError(
            "expected cursor to point to a gap between statements, not at an edge"
        )

    return scheduling.DoFissionLoops(proc, stmt._impl, n_lifts).result()


# TODO: Debug scheduling error in fuse
@sched_op([ForOrIfCursorA, ForOrIfCursorA, BoolA])
def fuse(proc, stmt1, stmt2, unsafe_disable_check=False):
    """
    fuse together two loops or if-guards, provided that the loop bounds
    or guard conditions are compatible.

    args:
        stmt1, stmt2        - cursors to the two loops or if-statements
                              that are being fused

    rewrite:
        `for i in e:` <- stmt1
        `    s1`
        `for j in e:` <- stmt2
        `    s2`
            ->
        `for i in e:`
        `    s1`
        `    s2[ j -> i ]`
    or
        `if cond:` <- stmt1
        `    s1`
        `if cond:` <- stmt2
        `    s2`
            ->
        `if cond:`
        `    s1`
        `    s2`
    """
    if isinstance(stmt1, PC.IfCursor) != isinstance(stmt2, PC.IfCursor):
        raise ValueError(
            "expected the two argument cursors to either both "
            "point to loops or both point to if-guards"
        )
    s1 = stmt1._impl
    s2 = stmt2._impl
    if isinstance(stmt1, PC.IfCursor):
        ir, fwd = scheduling.DoFuseIf(s1, s2)
    else:
        ir, fwd = scheduling.DoFuseLoop(s1, s2, unsafe_disable_check)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForCursorA, BoolA])
def remove_loop(proc, loop_cursor, unsafe_disable_check=False):
    """
    Remove the loop around some block of statements.
    This operation is allowable when the block of statements in question
    can be proven to be idempotent.

    args:
        loop_cursor     - cursor pointing to the loop to remove

    rewrite:
        `for i in _:`
        `    s`
            ->
        `s`
    """
    ir, fwd = scheduling.DoRemoveLoop(loop_cursor._impl, unsafe_disable_check)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA, NameA, NewExprA("block_cursor"), BoolA, BoolA])
def add_loop(
    proc, block_cursor, iter_name, hi_expr, guard=False, unsafe_disable_check=False
):
    """
    Add a loop around some block of statements.
    This operation is allowable when the block of statements in question
    can be proven to be idempotent.

    args:
        block_cursor    - cursor pointing to the block to wrap in a loop
        iter_name       - string name for the new iteration variable
        hi_expr         - string to be parsed into the upper bound expression
                          for the new loop
        guard           - Boolean (default False) signaling whether to
                          wrap the block in a `if iter_name == 0: block`
                          condition; in which case idempotency need not
                          be proven.

    rewrite:
        `s`  <--- block_cursor
        ->
        `for iter_name in hi_expr:`
        `    s`
    """

    if len(block_cursor) != 1:
        raise NotImplementedError("TODO: support blocks of size > 1")

    stmt_c = block_cursor[0]._impl
    ir, fwd = scheduling.DoAddLoop(
        stmt_c, iter_name, hi_expr, guard, unsafe_disable_check
    )
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForCursorA])
def unroll_loop(proc, loop_cursor):
    """
    Unroll a loop with a constant, literal loop bound

    args:
        loop_cursor     - cursor pointing to the loop to unroll

    rewrite:
        `for i in seq(0,3):`
        `    s`
            ->
        `s[ i -> 0 ]`
        `s[ i -> 1 ]`
        `s[ i -> 2 ]`
    """
    ir, fwd = scheduling.DoUnroll(loop_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Guard Conditions


@sched_op([ForOrIfCursorA])
def lift_scope(proc, scope_cursor):
    """
    Lift the indicated For/If-statement upwards one scope.

    args:
        scope_cursor       - cursor to the inner scope statement to lift up

    rewrite: (one example)
        `for i in _:`
        `    if p:`
        `        s1`
        `    else:`
        `        s2`
        ->
        `if p:`
        `    for i in _:`
        `        s1`
        `else:`
        `    for i in _:`
        `        s2`
    """
    stmt_c = scope_cursor._impl

    ir, fwd = scheduling.DoLiftScope(stmt_c)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([ForOrIfCursorA])
def eliminate_dead_code(proc, stmt_cursor):
    """
    if statements: eliminate branch that is never reachable
    for statements: eliminate loop if its condition is always false

    args:
        stmt_cursor       - cursor to the if or for statement

    rewrite:
        `if p:`
        `    s1`
        `else:`
        `    s2`
        -> (assuming `p` is always True)
        `s1`
    """

    ir, fwd = scheduling.DoEliminateDeadCode(stmt_cursor._impl)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


@sched_op([BlockCursorA, ListOrElemA(NewExprA("block"))])
def specialize(proc, block, conds):
    """
    Duplicate a statement block multiple times, with the provided
    `cond`itions indicating when each copy should be invoked.
    Doing this allows one to then schedule differently the "specialized"
    variants of the blocks in different ways.

    If `n` conditions are given, then `n+1` specialized copies of the block
    are created (with the last copy as a "default" version).

    args:
        block           - cursor pointing to the block to duplicate/specialize
        conds           - list of strings or string to be parsed into
                          guard conditions for the

    rewrite:
        `B`
            ->
        `if cond_0:`
        `    B`
        `elif cond_1:`
        `    B`
        ...
        `else:`
        `    B`
    """

    ir, fwd = scheduling.DoSpecialize(block._impl, conds)
    return Procedure(ir, _provenance_eq_Procedure=proc, _forward=fwd)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Deprecated Operations


@sched_op([BlockCursorA, NewExprA("block_cursor")])
def add_unsafe_guard(proc, block_cursor, var_expr):
    """
    DEPRECATED
    This operation is deprecated, and will be removed soon.
    """
    stmt = block_cursor._impl[0]

    return scheduling.DoAddUnsafeGuard(proc, stmt, var_expr).result()
