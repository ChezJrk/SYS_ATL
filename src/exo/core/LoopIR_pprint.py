import re
from collections import ChainMap
from dataclasses import dataclass, field

# google python formatting project to save myself the trouble of being overly
# clever run the function FormatCode to transform one string into a formatted
# string
from yapf.yapflib.yapf_api import FormatCode

from .LoopIR import T
from .LoopIR import UAST, LoopIR
from .internal_cursors import Node, Gap, Block, Cursor, InvalidCursorError, GapType
from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Notes on Layout Schemes...

"""
  functions should return a list of strings, one for each line

  standard inputs
  tab     - holds a string of white-space
  prec    - the operator precedence of the surrounding text
            if this string contains a lower precedence operation then
            we must wrap it in parentheses.
"""

# We expect pprint to install functions on the IR rather than
# expose functions; therefore hide all variables as local
__all__ = []

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Operator Precedence

op_prec = {
    "or": 10,
    #
    "and": 20,
    #
    "<": 30,
    ">": 30,
    "<=": 30,
    ">=": 30,
    "==": 30,
    #
    "+": 40,
    "-": 40,
    #
    "*": 50,
    "/": 50,
    "%": 50,
    #
    # unary minus
    "~": 60,
}


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# UAST Pretty Printing


@extclass(UAST.proc)
@extclass(UAST.fnarg)
@extclass(UAST.stmt)
@extclass(UAST.expr)
@extclass(UAST.type)
@extclass(UAST.w_access)
def __str__(self):
    return UAST_PPrinter(self).str()


del __str__


class UAST_PPrinter:
    def __init__(self, node):
        self._node = node

        self.env = ChainMap()
        self._tab = ""
        self._lines = []

        if isinstance(node, UAST.proc):
            self.pproc(node)
        elif isinstance(node, UAST.fnarg):
            self.addline(self.pfnarg(node))
        elif isinstance(node, UAST.stmt):
            self.pstmts([node])
        elif isinstance(node, UAST.expr):
            self.addline(self.pexpr(node))
        elif isinstance(node, UAST.type):
            self.addline(self.ptype(node))
        elif isinstance(node, UAST.w_access):
            if isinstance(node, UAST.Interval):
                lo = self.pexpr(node.lo) if node.lo else "None"
                hi = self.pexpr(node.hi) if node.hi else "None"
                self.addline(f"Interval({lo},{hi})")
            elif isinstance(node, UAST.Point):
                self.addline(f"Point({self.pexpr(node.pt)})")
            else:
                assert False, "bad case"
        else:
            assert False, f"cannot print a {type(node)}"

    def str(self):
        if isinstance(self._node, (UAST.type, UAST.w_access)):
            assert len(self._lines) == 1
            return self._lines[0]

        fmtstr, linted = FormatCode("\n".join(self._lines))
        if isinstance(self._node, LoopIR.proc):
            assert linted, "generated unlinted code..."
        return fmtstr

    def push(self, only=None):
        if only is None:
            self.env = self.env.new_child()
            self._tab = self._tab + "  "
        elif only == "env":
            self.env = self.env.new_child()
        elif only == "tab":
            self._tab = self._tab + "  "
        else:
            assert False, f"BAD only parameter {only}"

    def pop(self):
        self.env = self.env.parents
        self._tab = self._tab[:-2]

    def addline(self, line):
        self._lines.append(f"{self._tab}{line}")

    def new_name(self, nm):
        strnm = str(nm)
        if strnm not in self.env:
            self.env[strnm] = strnm
            return strnm
        else:
            s = self.env[strnm]
            m = re.match("^(.*)_([0-9]*)$", s)
            # either post-pend a _1 or increment the post-pended counter
            if not m:
                s = s + "_1"
            else:
                s = f"{m[1]}_{int(m[2]) + 1}"
            self.env[strnm] = s
            return s

    def get_name(self, nm):
        strnm = str(nm)
        if strnm in self.env:
            return self.env[strnm]
        else:
            return repr(nm)

    def pproc(self, p):
        name = p.name or "_anon_"
        args = [self.pfnarg(a) for a in p.args]
        self.addline(f"def {name}({','.join(args)}):")

        self.push()
        if p.instr:
            instr_lines = p.instr.c_instr.split("\n")
            instr_lines = [f"# @instr {instr_lines[0]}"] + [
                f"#        {l}" for l in instr_lines[1:]
            ]
            for l in instr_lines:
                self.addline(l)
        for pred in p.preds:
            self.addline(f"assert {self.pexpr(pred)}")
        self.pstmts(p.body)
        self.pop()

    def pfnarg(self, a):
        mem = f" @{a.mem.name()}" if a.mem else ""
        return f"{self.new_name(a.name)} : {self.ptype(a.type)}{mem}"

    def pstmts(self, body):
        for stmt in body:
            if isinstance(stmt, UAST.Pass):
                self.addline("pass")
            elif isinstance(stmt, UAST.Assign) or isinstance(stmt, UAST.Reduce):
                op = "=" if isinstance(stmt, UAST.Assign) else "+="

                rhs = self.pexpr(stmt.rhs)

                if len(stmt.idx) > 0:
                    idx = [self.pexpr(e) for e in stmt.idx]
                    lhs = f"{self.get_name(stmt.name)}[{','.join(idx)}]"
                else:
                    lhs = self.get_name(stmt.name)

                self.addline(f"{lhs} {op} {rhs}")
            elif isinstance(stmt, LoopIR.WriteConfig):
                cname = stmt.config.name()
                rhs = self.pexpr(stmt.rhs)
                self.addline(f"{cname}.{stmt.field} = {rhs}")
            elif isinstance(stmt, UAST.FreshAssign):
                rhs = self.pexpr(stmt.rhs)
                self.addline(f"{self.new_name(stmt.name)} = {rhs}")
            elif isinstance(stmt, UAST.Alloc):
                mem = f" @{stmt.mem.name()}" if stmt.mem else ""
                self.addline(
                    f"{self.new_name(stmt.name)} : {self.ptype(stmt.type)}{mem}"
                )
            elif isinstance(stmt, UAST.Call):
                pname = stmt.f.name or "_anon_"
                args = [self.pexpr(a) for a in stmt.args]
                self.addline(f"{pname}({','.join(args)})")
            elif isinstance(stmt, UAST.If):
                cond = self.pexpr(stmt.cond)
                self.addline(f"if {cond}:")
                self.push()
                self.pstmts(stmt.body)
                self.pop()
                if len(stmt.orelse) > 0:
                    self.addline("else:")
                    self.push()
                    self.pstmts(stmt.orelse)
                    self.pop()
            elif isinstance(stmt, UAST.For):
                cond = self.pexpr(stmt.cond)
                self.push(only="env")
                self.addline(f"for {self.new_name(stmt.iter)} in {cond}:")
                self.push(only="tab")
                self.pstmts(stmt.body)
                self.pop()
            else:
                assert False, "unrecognized stmt type"

    def pexpr(self, e, prec=0):
        if isinstance(e, UAST.Read):
            if len(e.idx) > 0:
                idx = [self.pexpr(i) for i in e.idx]
                return f"{self.get_name(e.name)}[{','.join(idx)}]"
            else:
                return self.get_name(e.name)
        elif isinstance(e, UAST.Const):
            return str(e.val)
        elif isinstance(e, UAST.BinOp):
            local_prec = op_prec[e.op]
            # increment rhs by 1 to account for left-associativity
            lhs = self.pexpr(e.lhs, prec=local_prec)
            rhs = self.pexpr(e.rhs, prec=local_prec + 1)
            s = f"{lhs} {e.op} {rhs}"
            # if we have a lower precedence than the environment...
            if local_prec < prec:
                s = f"({s})"
            return s
        elif isinstance(e, UAST.USub):
            return f"-{self.pexpr(e.arg, prec=op_prec['~'])}"
        elif isinstance(e, UAST.ParRange):
            return f"par({self.pexpr(e.lo)},{self.pexpr(e.hi)})"
        elif isinstance(e, UAST.SeqRange):
            return f"seq({self.pexpr(e.lo)},{self.pexpr(e.hi)})"
        elif isinstance(e, UAST.WindowExpr):

            def pacc(w):
                if isinstance(w, UAST.Point):
                    return self.pexpr(w.pt)
                elif isinstance(w, UAST.Interval):
                    lo = self.pexpr(w.lo) if w.lo else ""
                    hi = self.pexpr(w.hi) if w.hi else ""
                    return f"{lo}:{hi}"
                else:
                    assert False, "bad case"

            return f"{self.get_name(e.name)}[{', '.join([pacc(w) for w in e.idx])}]"
        elif isinstance(e, UAST.StrideExpr):
            return f"stride({self.get_name(e.name)}, {e.dim})"
        elif isinstance(e, UAST.Extern):
            pname = e.f.name() or "_anon_"
            args = [self.pexpr(a) for a in e.args]
            return f"{pname}({','.join(args)})"
        elif isinstance(e, LoopIR.ReadConfig):
            cname = e.config.name()
            return f"{cname}.{e.field}"
        else:
            assert False, "unrecognized expr type"

    def ptype(self, t):
        if isinstance(t, UAST.Num):
            return "R"
        elif isinstance(t, UAST.F16):
            return "f16"
        elif isinstance(t, UAST.F32):
            return "f32"
        elif isinstance(t, UAST.F64):
            return "f64"
        elif isinstance(t, UAST.INT8):
            return "i8"
        elif isinstance(t, UAST.UINT8):
            return "ui8"
        elif isinstance(t, UAST.UINT16):
            return "ui16"
        elif isinstance(t, UAST.INT32):
            return "i32"
        elif isinstance(t, UAST.Bool):
            return "bool"
        elif isinstance(t, UAST.Int):
            return "int"
        elif isinstance(t, UAST.Index):
            return "index"
        elif isinstance(t, UAST.Size):
            return "size"
        elif isinstance(t, UAST.Tensor):
            base = str(t.basetype())
            if t.is_window:
                base = f"[{base}]"
            rngs = ",".join([self.pexpr(r) for r in t.shape()])
            return f"{base}[{rngs}]"
        else:
            assert False, "impossible type case"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# LoopIR Pretty Printing


def _format_code(code):
    return FormatCode(code)[0].rstrip("\n")


@extclass(LoopIR.proc)
def __str__(self):
    return _format_code("\n".join(_print_proc(self, PrintEnv(), "")))


@extclass(LoopIR.fnarg)
def __str__(self):
    return _format_code(_print_fnarg(self, PrintEnv()))


@extclass(LoopIR.stmt)
def __str__(self):
    return _format_code("\n".join(_print_stmt(self, PrintEnv(), "")))


@extclass(LoopIR.expr)
def __str__(self):
    return _format_code(_print_expr(self, PrintEnv()))


@extclass(LoopIR.type)
def __str__(self):
    return _format_code(_print_type(self, PrintEnv()))


del __str__


@dataclass
class PrintEnv:
    env: ChainMap[Sym, str] = field(default_factory=ChainMap)
    names: ChainMap[str, int] = field(default_factory=ChainMap)

    def push(self) -> "PrintEnv":
        return PrintEnv(self.env.new_child(), self.names.new_child())

    def get_name(self, nm):
        if resolved := self.env.get(nm):
            return resolved

        candidate = str(nm)
        num = self.names.get(candidate, 1)
        while candidate in self.names:
            candidate = f"{nm}_{num}"
            num += 1

        self.env[nm] = candidate
        self.names[str(nm)] = num
        return candidate


def _print_proc(p, env: PrintEnv, indent: str) -> list[str]:
    args = [_print_fnarg(a, env) for a in p.args]

    lines = [f"{indent}def {p.name}({', '.join(args)}):"]

    indent = indent + "  "

    if p.instr:
        for i, line in enumerate(p.instr.c_instr.split("\n")):
            if i == 0:
                lines.append(f"{indent}# @instr {line}")
            else:
                lines.append(f"{indent}#        {line}")

    for pred in p.preds:
        lines.append(f"{indent}assert {_print_expr(pred, env)}")

    lines.extend(_print_block(p.body, env, indent))

    return lines


def _print_block(blk, env: PrintEnv, indent: str) -> list[str]:
    lines = []
    for stmt in blk:
        lines.extend(_print_stmt(stmt, env, indent))
    return lines


def _print_stmt(stmt, env: PrintEnv, indent: str) -> list[str]:
    if isinstance(stmt, LoopIR.Pass):
        return [f"{indent}pass"]

    elif isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
        op = "=" if isinstance(stmt, LoopIR.Assign) else "+="

        lhs = env.get_name(stmt.name)

        idx = [_print_expr(e, env) for e in stmt.idx]
        idx = f"[{', '.join(idx)}]" if idx else ""

        rhs = _print_expr(stmt.rhs, env)

        return [f"{indent}{lhs}{idx} {op} {rhs}"]

    elif isinstance(stmt, LoopIR.WriteConfig):
        cname = stmt.config.name()
        rhs = _print_expr(stmt.rhs, env)
        return [f"{indent}{cname}.{stmt.field} = {rhs}"]

    elif isinstance(stmt, LoopIR.WindowStmt):
        rhs = _print_expr(stmt.rhs, env)
        if stmt.special_window is not None:
            rhs = f"{rhs} @ {stmt.special_window.name()}"
        return [f"{indent}{env.get_name(stmt.name)} = {rhs}"]

    elif isinstance(stmt, LoopIR.Alloc):
        mem = f" @{stmt.mem.name()}" if stmt.mem else ""
        ty = _print_type(stmt.type, env)
        return [f"{indent}{env.get_name(stmt.name)} : {ty}{mem}"]

    elif isinstance(stmt, LoopIR.Free):
        # mem = f"@{stmt.mem.name()}" if stmt.mem else ""
        return [f"{indent}free({env.get_name(stmt.name)})"]

    elif isinstance(stmt, LoopIR.Call):
        args = [_print_expr(a, env) for a in stmt.args]
        return [f"{indent}{stmt.f.name}({', '.join(args)})"]

    elif isinstance(stmt, LoopIR.If):
        cond = _print_expr(stmt.cond, env)
        lines = [f"{indent}if {cond}:"]
        lines.extend(_print_block(stmt.body, env.push(), indent + "  "))
        if stmt.orelse:
            lines.append(f"{indent}else:")
            lines.extend(_print_block(stmt.orelse, env.push(), indent + "  "))
        return lines

    elif isinstance(stmt, LoopIR.For):
        lo = _print_expr(stmt.lo, env)
        hi = _print_expr(stmt.hi, env)
        body_env = env.push()
        loop_type = "par" if isinstance(stmt.loop_mode, LoopIR.Par) else "seq"
        lines = [
            f"{indent}for {body_env.get_name(stmt.iter)} in {loop_type}({lo}, {hi}):"
        ]
        lines.extend(_print_block(stmt.body, body_env, indent + "  "))
        return lines

    assert False, f"unrecognized stmt: {type(stmt)}"


def _print_fnarg(a, env: PrintEnv) -> str:
    if a.type == T.size:
        return f"{env.get_name(a.name)} : size"
    elif a.type == T.index:
        return f"{env.get_name(a.name)} : index"
    else:
        ty = _print_type(a.type, env)
        mem = f" @{a.mem.name()}" if a.mem else ""
        return f"{env.get_name(a.name)} : {ty}{mem}"


def _print_expr(e, env: PrintEnv, prec: int = 0) -> str:
    if isinstance(e, LoopIR.Read):
        name = env.get_name(e.name)
        idx = f"[{', '.join(_print_expr(i, env) for i in e.idx)}]" if e.idx else ""
        return f"{name}{idx}"

    elif isinstance(e, LoopIR.Const):
        return str(e.val)

    elif isinstance(e, LoopIR.USub):
        return f'-{_print_expr(e.arg, env, prec=op_prec["~"])}'

    elif isinstance(e, LoopIR.BinOp):
        local_prec = op_prec[e.op]
        # increment rhs by 1 to account for left-associativity
        lhs = _print_expr(e.lhs, env, prec=local_prec)
        rhs = _print_expr(e.rhs, env, prec=local_prec + 1)
        s = f"{lhs} {e.op} {rhs}"
        # if we have a lower precedence than the environment...
        if local_prec < prec:
            s = f"({s})"
        return s

    elif isinstance(e, LoopIR.WindowExpr):
        name = env.get_name(e.name)
        return f"{name}[{', '.join([_print_w_access(w, env) for w in e.idx])}]"

    elif isinstance(e, LoopIR.StrideExpr):
        return f"stride({env.get_name(e.name)}, {e.dim})"

    elif isinstance(e, LoopIR.Extern):
        pname = e.f.name() or "_anon_"
        args = [_print_expr(a, env) for a in e.args]
        return f"{pname}({', '.join(args)})"

    elif isinstance(e, LoopIR.ReadConfig):
        cname = e.config.name()
        return f"{cname}.{e.field}"

    assert False, f"unrecognized expr: {type(e)}"


def _print_type(t, env: PrintEnv) -> str:
    if isinstance(t, T.Num):
        return "R"
    elif isinstance(t, T.F16):
        return "f16"
    elif isinstance(t, T.F32):
        return "f32"
    elif isinstance(t, T.F64):
        return "f64"
    elif isinstance(t, T.INT8):
        return "i8"
    elif isinstance(t, T.UINT8):
        return "ui8"
    elif isinstance(t, T.UINT16):
        return "ui16"
    elif isinstance(t, T.INT32):
        return "i32"
    elif isinstance(t, T.Bool):
        return "bool"
    elif isinstance(t, T.Int):
        return "int"
    elif isinstance(t, T.Index):
        return "index"
    elif isinstance(t, T.Size):
        return "size"
    elif isinstance(t, T.Error):
        return "err"

    elif isinstance(t, T.Tensor):
        base = _print_type(t.basetype(), env)
        if t.is_window:
            base = f"[{base}]"
        ranges = ", ".join([_print_expr(r, env) for r in t.shape()])
        return f"{base}[{ranges}]"

    elif isinstance(t, T.Window):
        # Below, we print idx='[x:y]' with single quotes because yapf can't
        # parse the colon in idx=[0:n] since it thinks its assignment.
        return (
            f"Window(src_type={t.src_type},as_tensor={t.as_tensor},"
            f"src_buf={t.src_buf},"
            f"idx='[{', '.join([_print_w_access(w, env) for w in t.idx])}]')"
        )
    elif isinstance(t, T.Stride):
        return "stride"

    assert False, f"impossible type {type(t)}"


def _print_w_access(node, env: PrintEnv) -> str:
    if isinstance(node, LoopIR.Interval):
        lo = _print_expr(node.lo, env)
        hi = _print_expr(node.hi, env)
        return f"{lo}:{hi}"

    elif isinstance(node, LoopIR.Point):
        return _print_expr(node.pt, env)

    assert False, "bad case"


def _print_cursor(cur):
    if cur == None:
        raise InvalidCursorError("Trying to print the Invalid Cursor!")

    if isinstance(cur, Node) and not isinstance(cur._node, (LoopIR.proc, LoopIR.stmt)):
        raise NotImplementedError(
            "Cursor printing is only implemented for procs and statements"
        )

    root_cur = cur.root()
    lines = _print_cursor_proc(root_cur, cur, PrintEnv(), "")
    code = _format_code("\n".join(lines))
    # need to use "..." for Python parsing, but unquoted ellipses are prettier
    code = code.replace('"..."', "...")
    return code


def _print_cursor_proc(
    cur: Node, target: Cursor, env: PrintEnv, indent: str
) -> list[str]:
    p = cur._node
    assert isinstance(p, LoopIR.proc)

    match_comment = "  # <-- NODE" if cur == target else ""

    args = [_print_fnarg(a, env) for a in p.args]
    lines = [f"{indent}def {p.name}({', '.join(args)}):{match_comment}"]

    indent = indent + "  "

    if cur == target:
        if p.instr:
            for i, line in enumerate(p.instr.c_instr.split("\n")):
                if i == 0:
                    lines.append(f"{indent}# @instr {line}")
                else:
                    lines.append(f"{indent}#        {line}")

        for pred in p.preds:
            lines.append(f"{indent}assert {_print_expr(pred, env)}")

    lines.extend(_print_cursor_block(cur.body(), target, env, indent))
    return lines


def _print_cursor_block(
    cur: Block, target: Cursor, env: PrintEnv, indent: str
) -> list[str]:
    def while_next(c):
        s = []
        while True:
            try:
                c = c.next()
                s.extend(local_stmt(c))
            except:
                return s

    def while_prev(c):
        s = []
        while True:
            try:
                c = c.prev()
                s.append(local_stmt(c))
            except:
                s.reverse()
                return [x for xs in s for x in xs]

    def local_stmt(c):
        return _print_cursor_stmt(c, target, env, indent)

    if isinstance(target, Gap) and target in cur:
        if target._type == GapType.Before:
            return [
                *while_prev(target.anchor()),
                f"{indent}[GAP - Before]",
                *local_stmt(target.anchor()),
                *while_next(target.anchor()),
            ]
        else:
            assert target._type == GapType.After
            return [
                *while_prev(target.anchor()),
                *local_stmt(target.anchor()),
                f"{indent}[GAP - After]",
                *while_next(target.anchor()),
            ]

    elif isinstance(target, Block) and target in cur:
        block = [f"{indent}# BLOCK START"]
        for stmt in target:
            block.extend(local_stmt(stmt))
        block.append(f"{indent}# BLOCK END")
        return [
            *while_prev(target[0]),
            *block,
            *while_next(target[-1]),
        ]

    else:
        block = []
        for stmt in cur:
            block.extend(local_stmt(stmt))
        return block


def _print_cursor_stmt(
    cur: Node, target: Cursor, env: PrintEnv, indent: str
) -> list[str]:
    stmt = cur._node

    if isinstance(stmt, LoopIR.If):
        cond = _print_expr(stmt.cond, env)
        lines = [f"{indent}if {cond}:"]
        lines.extend(_print_cursor_block(cur.body(), target, env.push(), indent + "  "))
        if stmt.orelse:
            lines.append(f"{indent}else:")
            lines.extend(
                _print_cursor_block(cur.orelse(), target, env.push(), indent + "  ")
            )

    elif isinstance(stmt, LoopIR.For):
        lo = _print_expr(stmt.lo, env)
        hi = _print_expr(stmt.hi, env)
        body_env = env.push()
        loop_type = "par" if isinstance(stmt.loop_mode, LoopIR.Par) else "seq"
        lines = [
            f"{indent}for {body_env.get_name(stmt.iter)} in {loop_type}({lo}, {hi}):",
            *_print_cursor_block(cur.body(), target, body_env, indent + "  "),
        ]

    else:
        lines = _print_stmt(stmt, env, indent)
    if cur == target:
        lines[0] = f"{lines[0]}  # <-- NODE"

    return lines
