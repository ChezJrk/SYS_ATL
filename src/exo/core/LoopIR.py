import re
from collections import ChainMap, defaultdict
from typing import List, Type

from asdl_adt import ADT, validators

from .extern import Extern
from .configs import Config
from .memory import MemWin, Memory, SpecialWindow
from .prelude import Sym, SrcInfo, extclass


# --------------------------------------------------------------------------- #
# Validated string subtypes
# --------------------------------------------------------------------------- #


class Identifier(str):
    _valid_re = re.compile(r"^(?:_\w|[a-zA-Z])\w*$")

    def __new__(cls, name):
        name = str(name)
        if Identifier._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f"invalid identifier: {name}")


class IdentifierOrHole(str):
    _valid_re = re.compile(r"^[a-zA-Z_]\w*$")

    def __new__(cls, name):
        name = str(name)
        if IdentifierOrHole._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f"invalid identifier: {name}")


comparision_ops = {"<", ">", "<=", ">=", "=="}
arithmetic_ops = {"+", "-", "*", "/", "%"}
logical_ops = {"and", "or"}

front_ops = comparision_ops | arithmetic_ops | logical_ops


class Operator(str):
    def __new__(cls, op):
        op = str(op)
        if op in front_ops:
            return super().__new__(cls, op)
        raise ValueError(f"invalid operator: {op}")


# --------------------------------------------------------------------------- #
# Loop IR
# --------------------------------------------------------------------------- #


LoopIR = ADT(
    """
module LoopIR {
    proc = ( name    name,
             fnarg*  args,
             expr*   preds,
             stmt*   body,
             instr?  instr,
             srcinfo srcinfo )

    instr  = ( string c_instr,
               string c_global )

    fnarg  = ( sym     name,
               type    type,
               memwin? mem,
               srcinfo srcinfo )

    stmt = Assign( sym name, type type, expr* idx, expr rhs )
         | Reduce( sym name, type type, expr* idx, expr rhs )
         | WriteConfig( config config, string field, expr rhs )
         | Pass()
         | If( expr cond, stmt* body, stmt* orelse )
         | For( sym iter, expr lo, expr hi, stmt* body, loop_mode loop_mode )
         | Alloc( sym name, type type, mem mem )
         | Free( sym name, type type, mem mem )
         | Call( proc f, expr* args )
         | WindowStmt( sym name, expr rhs, special_window? special_window )
         attributes( srcinfo srcinfo )

    loop_mode = Seq()
                | Par()

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | Extern( extern f, expr* args )
         | WindowExpr( sym name, w_access* idx )
         | StrideExpr( sym name, int dim )
         | ReadConfig( config config, string field )
         attributes( type type, srcinfo srcinfo )

    -- WindowExpr = (base : Sym, idx : [ Pt Expr | Interval Expr Expr ])
    w_access = Interval( expr lo, expr hi )
             | Point( expr pt )
             attributes( srcinfo srcinfo )

    type = Num()
         | F16()
         | F32()
         | F64()
         | INT8()
         | UINT8()
         | UINT16()
         | INT32()
         | Bool()
         | Int()
         | Index()
         | Size()
         | Stride()
         | Error()
         | Tensor( expr* hi, bool is_window, type type )
         -- src_type  - type of the Tensor from which the window was created
         -- as_tensor - tensor type as if this window were simply a tensor
         --             itself
         -- src_buf   - sym for the Tensor from which the window was created
         -- idx       - the expression that created this window
         -- NB: when creating a derived window from another derived window,
         -- we must "chain" the two window exprs so that src_type, src_buf
         -- still refer to the original Tensor
         | WindowType( type src_type, type as_tensor,
                       sym src_buf, w_access *idx )

    -- Dense tensor: Tensor(is_window = False)
    -- Window parameter (of proc): Tensor(is_window = True)
    -- Derived window (from WindowExpr): WindowType / T.Window

    -- First two are both "tensors" although imprecisely sometimes "tensor"
    -- refers only to "dense tensor" -- we should be more clear about that.
    -- Latter two are both "windows" (allows strides), but have separate
    -- types since derived windows (WindowType) requires aliasing reasoning
}""",
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "memwin": Type[MemWin],
        "mem": Type[Memory],
        "special_window": Type[SpecialWindow],
        "extern": Extern,
        "config": Config,
        "binop": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
    },
    memoize={
        "Num",
        "F16",
        "F32",
        "F64",
        "INT8",
        "UINT8",
        "UINT16",
        "INT32",
        "Bool",
        "Int",
        "Index",
        "Size",
        "Stride",
        "Error",
    },
)

# --------------------------------------------------------------------------- #
# Untyped AST
# --------------------------------------------------------------------------- #

UAST = ADT(
    """
module UAST {
    proc    = ( name?           name,
                fnarg*          args,
                expr*           preds,
                stmt*           body,
                instr?          instr,
                srcinfo         srcinfo )

    instr   = ( string          c_instr,
                string          c_global )

    fnarg   = ( sym             name,
                type            type,
                memwin?         mem,
                srcinfo         srcinfo )

    stmt    = Assign  ( sym name, expr* idx, expr rhs )
            | Reduce  ( sym name, expr* idx, expr rhs )
            | WriteConfig ( config config, string field, expr rhs )
            | FreshAssign( sym name, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | For     ( sym iter,  expr cond,   stmt* body )
            | Alloc   ( sym name, type type, mem? mem )
            | Call    ( loopir_proc f, expr* args )
            attributes( srcinfo srcinfo )

    expr    = Read    ( sym name, expr* idx )
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | Extern( extern f, expr* args )
            | WindowExpr( sym name, w_access* idx, special_window? special_window )
            | StrideExpr( sym name, int dim )
            | ParRange( expr lo, expr hi ) -- only use for loop cond
            | SeqRange( expr lo, expr hi ) -- only use for loop cond
            | ReadConfig( config config, string field )
            attributes( srcinfo srcinfo )

    w_access= Interval( expr? lo, expr? hi )
            | Point( expr pt )
            attributes( srcinfo srcinfo )

    type    = Num   ()
            | F16   ()
            | F32   ()
            | F64   ()
            | INT8  ()
            | UINT8  ()
            | UINT16 ()
            | INT32 ()
            | Bool  ()
            | Int   ()
            | Size  ()
            | Index ()
            | Stride()
            | Tensor( expr *hi, bool is_window, type type )
} """,
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "memwin": Type[MemWin],
        "mem": Type[Memory],
        "special_window": Type[SpecialWindow],
        "extern": Extern,
        "config": Config,
        "loopir_proc": LoopIR.proc,
        "op": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
    },
    memoize={
        "Num",
        "F16",
        "F32",
        "F64",
        "INT8",
        "UINT8",
        "UINT16",
        "INT32",
        "Bool",
        "Int",
        "Size",
        "Index",
        "Stride",
    },
)

# --------------------------------------------------------------------------- #
# Pattern AST
#   - used to specify pattern-matches
# --------------------------------------------------------------------------- #

PAST = ADT(
    """
module PAST {

    stmt    = Assign  ( name name, expr* idx, expr rhs )
            | Reduce  ( name name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body, stmt* orelse )
            | For     ( name iter, expr lo, expr hi, stmt* body )
            | Alloc   ( name name, expr* sizes ) -- may want to add mem back in?
            | Call    ( name f, expr* args )
            | WriteConfig ( name config, name field )
            | S_Hole  ()
            attributes( srcinfo srcinfo )

    expr    = Read    ( name name, expr* idx )
            | StrideExpr( name name, int? dim )
            | E_Hole  ()
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | Extern ( name f, expr* args )
            | ReadConfig( string config, string field )
            attributes( srcinfo srcinfo )

} """,
    ext_types={
        "name": validators.instance_of(IdentifierOrHole, convert=True),
        "op": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
    },
)


# --------------------------------------------------------------------------- #
# C Codegen AST
# --------------------------------------------------------------------------- #

CIR = ADT(
    """
module CIR {

    expr    = Read    ( sym name, bool is_non_neg )
            | Stride  ( sym name, int dim )
            | Const   ( object val )
            | BinOp   ( op op, expr lhs, expr rhs, bool is_non_neg )
            | USub    ( expr arg, bool is_non_neg )

} """,
    ext_types={
        "bool": bool,
        "int": int,
        "sym": Sym,
        "op": validators.instance_of(Operator, convert=True),
    },
)


# --------------------------------------------------------------------------- #
# Extension methods
# --------------------------------------------------------------------------- #


@extclass(UAST.Tensor)
@extclass(UAST.Num)
@extclass(UAST.F16)
@extclass(UAST.F32)
@extclass(UAST.F64)
@extclass(UAST.INT8)
@extclass(UAST.UINT8)
@extclass(UAST.UINT16)
@extclass(UAST.INT32)
def shape(t):
    shp = t.hi if isinstance(t, UAST.Tensor) else []
    return shp


del shape


@extclass(UAST.type)
def basetype(t):
    if isinstance(t, UAST.Tensor):
        t = t.type
    return t


del basetype


# make proc be a hashable object
@extclass(LoopIR.proc)
def __hash__(self):
    return id(self)


del __hash__


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types


class T:
    Num = LoopIR.Num
    F16 = LoopIR.F16
    F32 = LoopIR.F32
    F64 = LoopIR.F64
    INT8 = LoopIR.INT8
    UINT8 = LoopIR.UINT8
    UINT16 = LoopIR.UINT16
    INT32 = LoopIR.INT32
    Bool = LoopIR.Bool
    Int = LoopIR.Int
    Index = LoopIR.Index
    Size = LoopIR.Size
    Stride = LoopIR.Stride
    Error = LoopIR.Error
    Tensor = LoopIR.Tensor
    Window = LoopIR.WindowType
    type = LoopIR.type
    R = Num()
    f16 = F16()
    f32 = F32()
    int8 = INT8()
    uint8 = UINT8()
    uint16 = UINT16()
    i8 = INT8()
    ui8 = UINT8()
    ui16 = UINT16()
    int32 = INT32()
    i32 = INT32()
    f64 = F64()
    bool = Bool()  # note: accessed as T.bool outside this module
    int = Int()
    index = Index()
    size = Size()
    stride = Stride()
    err = Error()


# --------------------------------------------------------------------------- #
# type helper functions


@extclass(T.Tensor)
def as_tensor_type(t):
    return t


@extclass(T.Window)
def as_tensor_type(t):
    return t.as_tensor


del as_tensor_type


@extclass(T.Tensor)
@extclass(T.Window)
@extclass(T.Num)
@extclass(T.F16)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
@extclass(T.UINT8)
@extclass(T.UINT16)
@extclass(T.INT32)
def shape(t):
    if isinstance(t, T.Window):
        return t.as_tensor.shape()
    elif isinstance(t, T.Tensor):
        assert not isinstance(t.type, T.Tensor), "expect no nesting"
        return t.hi
    else:
        return []


del shape


@extclass(T.Num)
@extclass(T.F16)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
@extclass(T.UINT8)
@extclass(T.UINT16)
@extclass(T.INT32)
@extclass(T.Bool)
@extclass(T.Int)
@extclass(T.Index)
@extclass(T.Size)
@extclass(T.Stride)
def ctype(t):
    if isinstance(t, T.Num):
        assert False, "Don't ask for ctype of Num"
    elif isinstance(t, T.F16):
        return "_Float16"
    elif isinstance(t, T.F32):
        return "float"
    elif isinstance(t, T.F64):
        return "double"
    elif isinstance(t, T.INT8):
        return "int8_t"
    elif isinstance(t, T.UINT8):
        return "uint8_t"
    elif isinstance(t, T.UINT16):
        return "uint16_t"
    elif isinstance(t, T.INT32):
        return "int32_t"
    elif isinstance(t, T.Bool):
        return "bool"
    elif isinstance(t, (T.Int, T.Index, T.Size, T.Stride)):
        return "int_fast32_t"


del ctype


@extclass(LoopIR.type)
def is_real_scalar(t):
    return isinstance(
        t, (T.Num, T.F16, T.F32, T.F64, T.INT8, T.UINT8, T.UINT16, T.INT32)
    )


del is_real_scalar


@extclass(LoopIR.type)
def is_tensor_or_window(t):
    return isinstance(t, (T.Tensor, T.Window))


del is_tensor_or_window


@extclass(LoopIR.type)
def is_win(t):
    # T.Tensor and t.is_window: window parameter
    # T.Window: derived window
    return (isinstance(t, T.Tensor) and t.is_window) or isinstance(t, T.Window)


del is_win


@extclass(LoopIR.type)
def is_dense_tensor(t):
    return isinstance(t, T.Tensor) and not t.is_window


del is_dense_tensor


@extclass(LoopIR.type)
def is_numeric(t):
    return t.is_real_scalar() or isinstance(t, (T.Tensor, T.Window))


del is_numeric


@extclass(LoopIR.type)
def is_bool(t):
    return isinstance(t, (T.Bool))


del is_bool


@extclass(LoopIR.type)
def is_indexable(t):
    return isinstance(t, (T.Int, T.Index, T.Size))


del is_indexable


@extclass(LoopIR.type)
def is_stridable(t):
    return isinstance(t, (T.Int, T.Stride))


@extclass(LoopIR.type)
def basetype(t):
    if isinstance(t, T.Window):
        return t.as_tensor.basetype()
    elif isinstance(t, T.Tensor):
        assert not t.type.is_tensor_or_window()
        return t.type
    else:
        return t


del basetype


def chain_window_idx(idx0, idx1):
    """Given

    window_0 = tensor[idx0]
    window_1 = window_0[idx1]

    Return chained_idx such that window_1 = tensor[chained_idx]
    """

    def add_e(scalar_0, scalar_1):
        if isinstance(scalar_0, LoopIR.Const) and scalar_0.val == 0:
            return scalar_1
        if isinstance(scalar_1, LoopIR.Const) and scalar_1.val == 0:
            return scalar_0
        return LoopIR.BinOp("+", scalar_0, scalar_1, T.index, scalar_1.srcinfo)

    assert sum(isinstance(e0, LoopIR.Interval) for e0 in idx0) == len(idx1)
    chained_idx = [None] * len(idx0)
    i1 = 0
    for i0, e0 in enumerate(idx0):
        if isinstance(e0, LoopIR.Point):
            chained_idx[i0] = e0
        else:
            assert isinstance(e0, LoopIR.Interval)
            e1 = idx1[i1]
            i1 += 1
            srcinfo = e1.srcinfo  # newer srcinfo likely more relevant
            if isinstance(e1, LoopIR.Point):
                chained_idx[i0] = LoopIR.Point(add_e(e0.lo, e1.pt), srcinfo)
            else:
                # Note e0.hi unused ... not responsibility here to do
                # bounds checking.
                chained_idx[i0] = LoopIR.Interval(
                    add_e(e0.lo, e1.lo), add_e(e0.lo, e1.hi), srcinfo
                )
    return chained_idx


def build_window_shape(ws: List[LoopIR.w_access]):
    def subtract(hi, lo):
        if isinstance(lo, LoopIR.Const) and lo.val == 0:
            return hi
        else:
            return LoopIR.BinOp("-", hi, lo, T.index, hi.srcinfo)

    return [subtract(w.hi, w.lo) for w in ws if isinstance(w, LoopIR.Interval)]


def create_window_type(in_name: Sym, in_typ: LoopIR.type, idx):
    """Construct a derived window type from any tensor or window type"""
    assert isinstance(in_name, Sym)
    window_shape = build_window_shape(idx)
    as_tensor = T.Tensor(window_shape, True, in_typ.basetype())

    if isinstance(in_typ, T.Tensor):
        # in_typ is dense tensor or window parameter
        w_typ = T.Window(in_typ, as_tensor, in_name, idx)
    else:
        # in_typ is another derived window
        # we need to "inline" through to get the underlying Tensor
        assert isinstance(in_typ, T.Window)
        chained_idx = chain_window_idx(in_typ.idx, idx)
        w_typ = T.Window(in_typ.src_type, as_tensor, in_typ.src_buf, chained_idx)

    return w_typ


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Install string printing functions on LoopIR, UAST and T
# This must be imported after those objects are defined to
# prevent circular inclusion problems
# TODO: FIX THIS!!!
# noinspection PyUnresolvedReferences
from . import LoopIR_pprint


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def apply_proc(self, old):
        return self.map_proc(old) or old

    def apply_fnarg(self, old):
        return self.map_fnarg(old) or old

    def apply_stmts(self, old):
        if (new := self.map_stmts(old)) is not None:
            return new
        return old

    def apply_exprs(self, old):
        if (new := self.map_exprs(old)) is not None:
            return new
        return old

    def apply_s(self, old):
        if (new := self.map_s(old)) is not None:
            return new
        return [old]

    def apply_e(self, old):
        return self.map_e(old) or old

    def apply_w_access(self, old):
        return self.map_w_access(old) or old

    def apply_t(self, old):
        return self.map_t(old) or old

    def map_proc(self, p):
        new_args = self._map_list(self.map_fnarg, p.args)
        new_preds = self.map_exprs(p.preds)
        new_body = self.map_stmts(p.body)

        if any((new_args is not None, new_preds is not None, new_body is not None)):
            new_preds = new_preds or p.preds
            new_preds = [
                p for p in new_preds if not (isinstance(p, LoopIR.Const) and p.val)
            ]
            return p.update(
                args=new_args or p.args, preds=new_preds, body=new_body or p.body
            )

        return None

    def map_fnarg(self, a):
        if t := self.map_t(a.type):
            return a.update(type=t)

        return None

    def map_stmts(self, stmts):
        return self._map_list(self.map_s, stmts)

    def map_exprs(self, exprs):
        return self._map_list(self.map_e, exprs)

    def map_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            new_type = self.map_t(s.type)
            new_idx = self.map_exprs(s.idx)
            new_rhs = self.map_e(s.rhs)
            if any((new_type, new_idx is not None, new_rhs)):
                return [
                    s.update(
                        type=new_type or s.type,
                        idx=new_idx or s.idx,
                        rhs=new_rhs or s.rhs,
                    )
                ]
        elif isinstance(s, (LoopIR.WriteConfig, LoopIR.WindowStmt)):
            new_rhs = self.map_e(s.rhs)
            if new_rhs:
                return [s.update(rhs=new_rhs or s.rhs)]
        elif isinstance(s, LoopIR.If):
            new_cond = self.map_e(s.cond)
            new_body = self.map_stmts(s.body)
            new_orelse = self.map_stmts(s.orelse)
            if any((new_cond, new_body is not None, new_orelse is not None)):
                return [
                    s.update(
                        cond=new_cond or s.cond,
                        body=new_body or s.body,
                        orelse=new_orelse or s.orelse,
                    )
                ]
        elif isinstance(s, LoopIR.For):
            new_lo = self.map_e(s.lo)
            new_hi = self.map_e(s.hi)
            new_body = self.map_stmts(s.body)
            if any((new_lo, new_hi, new_body is not None)):
                return [
                    s.update(
                        lo=new_lo or s.lo, hi=new_hi or s.hi, body=new_body or s.body
                    )
                ]
        elif isinstance(s, LoopIR.Call):
            new_args = self.map_exprs(s.args)
            if new_args is not None:
                return [s.update(args=new_args or s.args)]
        elif isinstance(s, LoopIR.Alloc):
            new_type = self.map_t(s.type)
            if new_type:
                return [s.update(type=new_type or s.type)]
        elif isinstance(s, LoopIR.Pass):
            return None
        else:
            raise NotImplementedError(f"bad case {type(s)}")
        return None

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            new_type = self.map_t(e.type)
            new_idx = self.map_exprs(e.idx)
            if any((new_type, new_idx is not None)):
                return e.update(
                    idx=new_idx or e.idx,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.BinOp):
            new_lhs = self.map_e(e.lhs)
            new_rhs = self.map_e(e.rhs)
            new_type = self.map_t(e.type)
            if any((new_lhs, new_rhs, new_type)):
                return e.update(
                    lhs=new_lhs or e.lhs,
                    rhs=new_rhs or e.rhs,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.Extern):
            new_type = self.map_t(e.type)
            new_args = self.map_exprs(e.args)
            if any((new_type, new_args is not None)):
                return e.update(
                    args=new_args or e.args,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.USub):
            new_arg = self.map_e(e.arg)
            new_type = self.map_t(e.type)
            if any((new_arg, new_type)):
                return e.update(
                    arg=new_arg or e.arg,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.WindowExpr):
            new_idx = self._map_list(self.map_w_access, e.idx)
            new_type = self.map_t(e.type)
            if any((new_idx is not None, new_type)):
                return e.update(
                    idx=new_idx or e.idx,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.ReadConfig):
            if new_type := self.map_t(e.type):
                return e.update(type=new_type or e.type)
        elif isinstance(e, (LoopIR.Const, LoopIR.StrideExpr)):
            return None
        else:
            raise NotImplementedError(f"bad case {type(e)}")
        return None

    def map_w_access(self, w):
        if isinstance(w, LoopIR.Interval):
            new_lo = self.map_e(w.lo)
            new_hi = self.map_e(w.hi)
            if new_lo or new_hi:
                return w.update(
                    lo=new_lo or w.lo,
                    hi=new_hi or w.hi,
                )
        else:
            if new_pt := self.map_e(w.pt):
                return w.update(pt=new_pt or w.pt)
        return None

    def map_t(self, t):
        if isinstance(t, T.Tensor):
            new_hi = self.map_exprs(t.hi)
            new_type = self.map_t(t.type)
            if (new_hi is not None) or new_type:
                return t.update(hi=new_hi or t.hi, type=new_type or t.type)
        elif isinstance(t, T.Window):
            new_src_type = self.map_t(t.src_type)
            new_as_tensor = self.map_t(t.as_tensor)
            new_idx = self._map_list(self.map_w_access, t.idx)
            if new_src_type or new_as_tensor or (new_idx is not None):
                return t.update(
                    src_type=new_src_type or t.src_type,
                    as_tensor=new_as_tensor or t.as_tensor,
                    idx=new_idx or t.idx,
                )
        return None

    @staticmethod
    def _map_list(fn, nodes):
        new_stmts = []
        needs_update = False

        for s in nodes:
            s2 = fn(s)
            if s2 is None:
                new_stmts.append(s)
            else:
                needs_update = True
                if isinstance(s2, list):
                    new_stmts.extend(s2)
                else:
                    new_stmts.append(s2)

        if not needs_update:
            return None

        return new_stmts


class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc = proc

        for a in self.proc.args:
            self.do_t(a.type)
        for p in self.proc.preds:
            self.do_e(p)

        self.do_stmts(self.proc.body)

    def do_stmts(self, stmts):
        for s in stmts:
            self.do_s(s)

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
            self.do_t(s.type)
        elif styp is LoopIR.WriteConfig:
            self.do_e(s.rhs)
        elif styp is LoopIR.WindowStmt:
            self.do_e(s.rhs)
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
        elif styp is LoopIR.For:
            self.do_e(s.lo)
            self.do_e(s.hi)
            self.do_stmts(s.body)
        elif styp is LoopIR.Call:
            for e in s.args:
                self.do_e(e)
        elif styp is LoopIR.Alloc:
            self.do_t(s.type)
        else:
            pass

    def do_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            for e in e.idx:
                self.do_e(e)
        elif etyp is LoopIR.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        elif etyp is LoopIR.Extern:
            for a in e.args:
                self.do_e(a)
        elif etyp is LoopIR.USub:
            self.do_e(e.arg)
        elif etyp is LoopIR.WindowExpr:
            for w in e.idx:
                self.do_w_access(w)
        else:
            pass

        self.do_t(e.type)

    def do_w_access(self, w):
        if isinstance(w, LoopIR.Interval):
            self.do_e(w.lo)
            self.do_e(w.hi)
        elif isinstance(w, LoopIR.Point):
            self.do_e(w.pt)
        else:
            assert False, "bad case"

    def do_t(self, t):
        if isinstance(t, T.Tensor):
            for i in t.hi:
                self.do_e(i)
        elif isinstance(t, T.Window):
            self.do_t(t.src_type)
            self.do_t(t.as_tensor)
            for w in t.idx:
                self.do_w_access(w)
        else:
            pass


class LoopIR_Compare:
    def __init__(self):
        pass

    def match_stmts(self, stmts1, stmts2):
        return all(self.match_s(s1, s2) for s1, s2 in zip(stmts1, stmts2))

    def match_s(self, s1, s2):
        if type(s1) is not type(s2):
            return False

        if isinstance(s1, (LoopIR.Assign, LoopIR.Reduce)):
            return (
                self.match_name(s1.name, s2.name)
                and self.match_t(s1.type, s2.type)
                and all(self.match_e(i1, i2) for i1, i2 in zip(s1.idx, s2.idx))
                and self.match_e(s1.rhs, s2.rhs)
            )
        elif isinstance(s1, LoopIR.WriteConfig):
            # TODO: check config and field equality
            return (
                s1.config == s2.config
                and s1.field == s2.field
                and self.match_e(s1.rhs, s2.rhs)
            )
        elif isinstance(s1, LoopIR.Pass):
            return True
        elif isinstance(s1, LoopIR.If):
            return (
                self.match_e(s1.cond, s2.cond)
                and self.match_stmts(s1.body, s2.body)
                and self.match_stmts(s1.orelse, s2.orelse)
            )
        elif isinstance(s1, LoopIR.For):
            return (
                self.match_name(s1.iter, s2.iter)
                and self.match_e(s1.lo, s2.lo)
                and self.match_e(s1.hi, s2.hi)
                and self.match_stmts(s1.body, s2.body)
            )
        elif isinstance(s1, LoopIR.Alloc):
            return self.match_name(s1.name, s2.name) and self.match_t(s1.type, s2.type)
        elif isinstance(s1, LoopIR.Call):
            return s1.f == s2.f and all(
                self.match_e(a1, a2) for a1, a2 in zip(s1.args, s2.args)
            )
        elif isinstance(s1, LoopIR.WindowStmt):
            return self.match_name(s1.name, s2.name) and self.match_e(s1.rhs, s2.rhs)
        else:
            assert False, f"bad case: {type(s1)}"

    def match_e(self, e1, e2):
        if type(e1) is not type(e2):
            return False

        if isinstance(e1, LoopIR.Read):
            return self.match_name(e1.name, e2.name) and all(
                self.match_e(i1, i2) for i1, i2 in zip(e1.idx, e2.idx)
            )
        elif isinstance(e1, LoopIR.Const):
            return e1.val == e2.val
        elif isinstance(e1, LoopIR.USub):
            return self.match_e(e1.arg, e2.arg)
        elif isinstance(e1, LoopIR.BinOp):
            return (
                e1.op == e2.op
                and self.match_e(e1.lhs, e2.lhs)
                and self.match_e(e1.rhs, e2.rhs)
            )
        elif isinstance(e1, LoopIR.Extern):
            # TODO: check f equality
            return e1.f is e2.f and all(
                self.match_e(a1, a2) for a1, a2 in zip(e1.args, e2.args)
            )
        elif isinstance(e1, LoopIR.WindowExpr):
            return self.match_name(e1.name, e2.name) and all(
                self.match_w_access(w1, w2) for w1, w2 in zip(e1.idx, e2.idx)
            )
        elif isinstance(e1, LoopIR.StrideExpr):
            return self.match_name(e1.name, e2.name) and e1.dim == e2.dim
        elif isinstance(e1, LoopIR.ReadConfig):
            # TODO: check configfield equality
            return e1.config == e2.config and e1.field == e2.field
        else:
            assert False, "bad case"

    def match_name(self, n1, n2):
        # TODO: if its a free var, check for exact match using ID. This
        # doesn't matter for join_loops, but in general if we use this
        # anywhere else, we should reason about that.
        return n1.name() == n2.name()

    def match_w_access(self, w1, w2):
        if isinstance(w1, LoopIR.Interval):
            return self.match_e(w1.lo, w2.lo) and self.match_e(w1.hi, w2.hi)
        elif isinstance(w1, LoopIR.Point):
            return self.match_e(w1.pt, w2.pt)
        else:
            assert False, "bad case"

    def match_t(self, t1, t2):
        if isinstance(t1, LoopIR.Tensor):
            return (
                t1.is_window == t2.is_window
                and self.match_t(t1.type, t2.type)
                and all(self.match_e(i1, i2) for i1, i2 in zip(t1.hi, t2.hi))
            )
        else:  # scalar
            return type(t1) == type(t2)


class GetReads(LoopIR_Do):
    def __init__(self):
        self.reads = []

    def do_e(self, e):
        if hasattr(e, "name"):
            self.reads.append((e.name, e.type))
        super().do_e(e)


class GetReadConfigs(LoopIR_Do):
    def __init__(self):
        self.readconfigs = []

    def do_e(self, e):
        if isinstance(e, LoopIR.ReadConfig):
            self.readconfigs.append((e.config, e.field))
        super().do_e(e)


def get_reads_of_expr(e):
    gr = GetReads()
    gr.do_e(e)
    return gr.reads


def get_reads_of_stmts(stmts):
    gr = GetReads()
    for stmt in stmts:
        gr.do_s(stmt)
    return gr.reads


def get_readconfigs(stmts):
    gr = GetReadConfigs()
    for stmt in stmts:
        gr.do_s(stmt)
    return gr.readconfigs


class GetWrites(LoopIR_Do):
    def __init__(self):
        self.writes = []
        # Translates access through T.Window to underlying T.Tensor
        self.window_dict = {}

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            sym = s.name
            self.writes.append((self.window_dict.get(sym, sym), s.type))
        elif isinstance(s, LoopIR.Call):
            writes_in_subproc = [a for a, _ in get_writes_of_stmts(s.f.body)]
            for arg, call_arg in zip(s.args, s.f.args):
                if call_arg.name in writes_in_subproc:
                    if isinstance(
                        arg, (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)
                    ):
                        sym = arg.name
                        self.writes.append((self.window_dict.get(sym, sym), arg.type))
        elif isinstance(s, LoopIR.WindowStmt):
            w_sym, base_sym = s.name, s.rhs.name
            while base_sym in self.window_dict:
                base_sym = self.window_dict[base_sym]
            self.window_dict[w_sym] = base_sym

        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_writes_of_stmts(stmts):
    gw = GetWrites()
    gw.do_stmts(stmts)
    return gw.writes


class GetWriteConfigs(LoopIR_Do):
    def __init__(self):
        self.writeconfigs = []

    def do_s(self, s):
        if isinstance(s, LoopIR.WriteConfig):
            self.writeconfigs.append((s.config, s.field))
        elif isinstance(s, LoopIR.Call):
            self.writeconfigs += get_writeconfigs(s.f.body)

        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_writeconfigs(stmts):
    gw = GetWriteConfigs()
    gw.do_stmts(stmts)
    return gw.writeconfigs


class GetLoopIters(LoopIR_Do):
    def __init__(self):
        self.loop_iters = []

    def do_s(self, s):
        if isinstance(s, LoopIR.For):
            self.loop_iters.append(s.iter)
        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_loop_iters(stmts):
    gw = GetLoopIters()
    gw.do_stmts(stmts)
    return gw.loop_iters


def is_const_zero(e):
    return isinstance(e, LoopIR.Const) and e.val == 0


class FreeVars(LoopIR_Do):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env = ChainMap()
        self.fv = set()

        for n in node:
            if isinstance(n, LoopIR.stmt):
                self.do_s(n)
            elif isinstance(n, LoopIR.expr):
                self.do_e(n)
            else:
                assert False, "expected stmt or expr"

    def result(self):
        return self.fv

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name not in self.env:
                self.fv.add(s.name)
        elif styp is LoopIR.WindowStmt:
            self.env[s.name] = True
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.push()
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
            self.pop()
            return
        elif styp is LoopIR.For:
            self.do_e(s.lo)
            self.do_e(s.hi)
            self.push()
            self.env[s.iter] = True
            self.do_stmts(s.body)
            self.pop()
            return
        elif styp is LoopIR.Alloc:
            self.env[s.name] = True

        super().do_s(s)

    def do_e(self, e):
        etyp = type(e)
        if (
            etyp is LoopIR.Read
            or etyp is LoopIR.WindowExpr
            or etyp is LoopIR.StrideExpr
        ):
            if e.name not in self.env:
                self.fv.add(e.name)

        super().do_e(e)

    def do_t(self, t):
        if isinstance(t, T.Window):
            if t.src_buf not in self.env:
                self.fv.add(t.src_buf)

        super().do_t(t)


class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        self.env = ChainMap()
        self.node = []

        if isinstance(node, LoopIR.proc):
            self.node = self.apply_proc(node)
        else:
            assert isinstance(node, list)
            for n in node:
                if isinstance(n, LoopIR.stmt):
                    self.node += self.apply_s(n)
                elif isinstance(n, LoopIR.expr):
                    self.node += [self.apply_e(n)]
                else:
                    assert False, "expected stmt or expr"

    def result(self):
        return self.node

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def map_fnarg(self, fa):
        nm = fa.name.copy()
        self.env[fa.name] = nm
        return fa.update(name=nm, type=self.map_t(fa.type) or fa.type)

    def map_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            s2 = super().map_s(s)
            if new_name := self.env.get(s.name):
                return [((s2 and s2[0]) or s).update(name=new_name)]
            else:
                return s2
        elif isinstance(s, LoopIR.Alloc):
            s2 = super().map_s(s)
            assert s.name not in self.env
            new_name = s.name.copy()
            self.env[s.name] = new_name
            return [((s2 and s2[0]) or s).update(name=new_name)]
        elif isinstance(s, LoopIR.WindowStmt):
            rhs = self.map_e(s.rhs) or s.rhs
            name = s.name.copy()
            self.env[s.name] = name
            return [s.update(name=name, rhs=rhs)]
        elif isinstance(s, LoopIR.If):
            self.push()
            stmts = super().map_s(s)
            self.pop()
            return stmts
        elif isinstance(s, LoopIR.For):
            lo = self.map_e(s.lo) or s.lo
            hi = self.map_e(s.hi) or s.hi

            self.push()
            itr = s.iter.copy()
            self.env[s.iter] = itr
            body = self.map_stmts(s.body) or s.body
            self.pop()

            return [s.update(iter=itr, lo=lo, hi=hi, body=body)]

        return super().map_s(s)

    def map_e(self, e):
        if isinstance(e, (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)):
            e2 = super().map_e(e)
            if new_name := self.env.get(e.name):
                return (e2 or e).update(name=new_name)
            else:
                return e2

        return super().map_e(e)

    def map_t(self, t):
        t2 = super().map_t(t)

        if isinstance(t, T.Window):
            if src_buf := self.env.get(t.src_buf):
                return (t2 or t).update(src_buf=src_buf)

        return t2


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, nodes, binding):
        assert isinstance(nodes, list)
        assert isinstance(binding, dict)
        assert all(isinstance(v, LoopIR.expr) for v in binding.values())
        assert not any(isinstance(v, LoopIR.WindowExpr) for v in binding.values())
        self.env = binding
        self.nodes = []
        for n in nodes:
            if isinstance(n, LoopIR.stmt):
                self.nodes += self.apply_s(n)
            elif isinstance(n, LoopIR.expr):
                self.nodes += [self.apply_e(n)]
            else:
                assert False, "expected stmt or expr"

    def result(self):
        return self.nodes

    def map_s(self, s):
        s2 = super().map_s(s)
        s_new = s2[0] if s2 is not None else s

        # this substitution could refer to a read or a window expression
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if s.name in self.env:
                sym = self.env[s.name]
                assert isinstance(sym, LoopIR.Read) and len(sym.idx) == 0
                return [s_new.update(name=sym.name)]

        return s2

    def map_e(self, e):
        # this substitution could refer to a read or a window expression
        if isinstance(e, LoopIR.Read):
            if e.name in self.env:
                sub_e = self.env[e.name]

                if not e.idx:
                    return sub_e

                assert isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0
                return e.update(name=sub_e.name, idx=self.apply_exprs(e.idx))

        elif isinstance(e, LoopIR.WindowExpr):
            if e.name in self.env:
                sub_e = self.env[e.name]

                if not e.idx:
                    return sub_e

                assert isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0
                return (super().map_e(e) or e).update(name=sub_e.name)

        elif isinstance(e, LoopIR.StrideExpr):
            if e.name in self.env:
                return e.update(name=self.env[e.name].name)

        return super().map_e(e)

    def map_t(self, t):
        t2 = super().map_t(t)

        if isinstance(t, T.Window):
            if src_buf := self.env.get(t.src_buf):
                return (t2 or t).update(src_buf=src_buf.name)

        return t2


# Data-flow dependencies between variable names
# TODO: Refactor this using new AI based analysis

# So, what is dependency analysis?
# Or to put it another way, what extensional property(s)
# does dependency analysis guarantee?
#
# Let B be a block of statements,
#     s be a store, and
#     x, y, … be names/symbols.
# Let FV(B) be the set of names that are free in B
#
# Then, first observe that the "meaning" of B is
#
#   Exec[[B]] : (FV(B) -> Value) -> Store -> Store
#
# (note that (FV(B) -> Value) is a valuation/mapping specifying the values
#       of all free variables)
# (further note that Store = (Name -> Maybe Value) is a valuation/mapping
#       of variables that models the heap/store)
#
# Then, (not x DependsOn y in B) for some y in FV(B) implies that
#
#   (Exec[[B]] (env[ y := v1 ]) s)[x] =
#   (Exec[[B]] (env[ y := v2 ]) s)[x]
#
# for all v1, v2
#
# Or in other words, the meaning of B
# w.r.t. its effect on x
# is invariant to the value of y
# when x does not depend on y in B


class LoopIR_Dependencies(LoopIR_Do):
    def __init__(self, buf_sym, stmts):
        self._buf_sym = buf_sym
        self._lhs = None
        self._depends = defaultdict(set)
        self._alias = dict()

        # If `lhs` is not None, then `lhs` will become dependent
        # on anything read.
        self._lhs = None

        # variables that affect whether or not the
        # currently examined code is even running
        self._context = set()

        # If `control` is True, then anything read will be added
        # to `context`.
        self._control = False

        self.do_stmts(stmts)

    def result(self):
        depends = self._depends[self._buf_sym]
        new = list(depends)
        done = []
        while True:
            if len(new) == 0:
                break
            sym = new.pop()
            done.append(sym)
            d = self._depends[sym]
            depends.update(d)
            new.extend(s for s in d if s not in done)

        return depends

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            lhs = self._alias.get(s.name, s.name)
            self._lhs = lhs
            self._depends[lhs].add(lhs)
            self._depends[lhs].update(self._context)
            for i in s.idx:
                self.do_e(i)
            self.do_e(s.rhs)
            self._lhs = None
        elif isinstance(s, LoopIR.WriteConfig):
            lhs = (s.config, s.field)
            self._lhs = lhs
            self._depends[lhs].add(lhs)
            self._depends[lhs].update(self._context)
            self.do_e(s.rhs)
            self._lhs = None
        elif isinstance(s, LoopIR.WindowStmt):
            rhs_buf = self._alias.get(s.rhs.name, s.rhs.name)
            self._alias[s.name] = rhs_buf
            self._lhs = rhs_buf
            self._depends[rhs_buf].add(rhs_buf)
            self.do_e(s.rhs)
            self._lhs = None

        elif isinstance(s, LoopIR.If):
            old_context = self._context
            self._context = old_context.copy()

            self._control = True
            self.do_e(s.cond)
            self._control = False

            self.do_stmts(s.body)
            self.do_stmts(s.orelse)

            self._context = old_context

        elif isinstance(s, LoopIR.For):
            old_context = self._context
            self._context = old_context.copy()

            self._control = True
            self._lhs = s.iter
            self._depends[s.iter].add(s.iter)
            self.do_e(s.lo)
            self.do_e(s.hi)
            self._lhs = None
            self._control = False

            self.do_stmts(s.body)

            self._context = old_context

        elif isinstance(s, LoopIR.Call):

            def process_reads():
                # now handle dependencies on buffers that might
                # be read from in the sub-procedure
                # and dependencies on other arguments
                for faa, aa in zip(s.f.args, s.args):
                    if faa.type.is_numeric():
                        maybe_read = any(
                            t[0] == faa.name for t in get_reads_of_stmts(s.f.body)
                        )
                    else:
                        maybe_read = True

                    if maybe_read:
                        self.do_e(aa)

                # additionally, we need to handle dependencies
                # on configuration fields
                for name in get_readconfigs(s.f.body):
                    if self._lhs:
                        self._depends[self._lhs].add(name)

            # for every argument that represents a buffer being
            # written to
            for fa, a in zip(s.f.args, s.args):
                maybe_write = fa.type.is_numeric() and any(
                    t[0] == fa.name for t in get_writes_of_stmts(s.f.body)
                )
                if maybe_write:
                    name = self._alias.get(a.name, a.name)
                    self._lhs = name
                    self._depends[name].add(name)
                    self._depends[name].update(self._context)
                    process_reads()
                    self._lhs = None

            # secondly, for every configuration field being written to
            # by this sub-procedure, we need to determine dependencies
            for name in get_writeconfigs(s.f.body):
                self._lhs = name
                self._depends[name].add(name)
                self._depends[name].update(self._context)
                process_reads()
                self._lhs = None

        elif isinstance(s, (LoopIR.Pass, LoopIR.Alloc)):
            pass
        else:
            assert False, "bad case"

    def do_e(self, e):
        if isinstance(e, (LoopIR.Read, LoopIR.WindowExpr)):

            def visit_idx(e):
                if isinstance(e, LoopIR.Read):
                    for i in e.idx:
                        self.do_e(i)
                else:
                    for w in e.idx:
                        if isinstance(w, LoopIR.Interval):
                            self.do_e(w.lo)
                            self.do_e(w.hi)
                        else:
                            self.do_e(w.pt)

            name = self._alias.get(e.name, e.name)
            if self._lhs:
                self._depends[self._lhs].add(name)
            if self._control:
                self._context.add(name)

            visit_idx(e)

        elif isinstance(e, LoopIR.ReadConfig):
            name = (e.config, e.field)
            if self._lhs:
                self._depends[self._lhs].add(name)
            if self._control:
                self._context.add(name)

        else:
            super().do_e(e)

    def do_t(self, t):
        pass
