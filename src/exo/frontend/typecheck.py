from ..core.LoopIR import (
    T,
    UAST,
    LoopIR,
    LoopIR_Dependencies,
    get_writeconfigs,
    get_loop_iters,
    create_window_type,
)
from ..core.extern import Extern_Typecheck_Error
from ..core.memory import *


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# This pass converts from UAST to LoopIR
#   In the process, different kinds of expressions are identified and
#   checked for proper usage.
#
#   Non-Goals:
#     - The UAST already has symbol conversion sorted out, so we
#       don't need to do that or to capture additional Python values
#       due to metaprogramming etc.
#     - We're not going to check bounds correctness here
#     - We're not going to check for parallelism-induced race-conditions here
#

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helper functions

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The typechecker


def check_call_types(err_handler, args, call_args):
    for call_a, sig_a in zip(args, call_args):
        if call_a.type == T.err:
            pass
        elif sig_a.type is T.size or sig_a.type is T.index:
            if not call_a.type.is_indexable():
                err_handler(
                    call_a,
                    "expected size or index type "
                    "expression, "
                    f"but got type {call_a.type}",
                )

        elif sig_a.type is T.bool:
            if not call_a.type is T.bool:
                err_handler(
                    call_a,
                    "expected bool-type variable, " f"but got type {call_a.type}",
                )

        elif sig_a.type is T.stride:
            if not call_a.type.is_stridable():
                err_handler(
                    call_a,
                    "expected stride-type variable, " f"but got type {call_a.type}",
                )

        elif sig_a.type.is_numeric():
            if call_a.type.is_numeric():
                if len(call_a.type.shape()) != len(sig_a.type.shape()):
                    err_handler(
                        call_a,
                        f"expected argument of type '{sig_a.type}', "
                        f"but got type '{call_a.type}'",
                    )

                # ensure scalars are simply variable names
                elif (
                    call_a.type.is_real_scalar()
                    and not isinstance(call_a, LoopIR.ReadConfig)
                    and not (isinstance(call_a, LoopIR.Read) and len(call_a.idx) == 0)
                ):
                    err_handler(
                        call_a,
                        "expected scalar arguments "
                        "to be simply variable names "
                        "for now",
                    )
            else:
                err_handler(
                    call_a,
                    "expected numeric type expression, " f"but got type {call_a.type}",
                )
        else:
            assert False, "bad argument type case"


class TypeChecker:
    def __init__(self, proc):
        self.uast_proc = proc
        self.env = dict()
        self.errors = []

        args = []
        for a in proc.args:
            typ = self.check_t(a.type)
            self.env[a.name] = typ
            mem = a.mem
            if mem is None:
                mem = DRAM
            args.append(LoopIR.fnarg(a.name, typ, mem, a.srcinfo))

        preds = []
        for p in proc.preds:
            pred = self.check_e(p, is_index=True)
            if pred.type != T.err and pred.type != T.bool:
                self.err(pred, f"expected a bool expression")
            preds.append(pred)

        body = self.check_stmts(proc.body)

        if not proc.name:
            self.err(proc, "expected all procedures to be named")

        loop_iters = set(get_loop_iters(body))
        for name in get_writeconfigs(body):
            deps = LoopIR_Dependencies(name, body).result()
            if loop_iters & deps != set():
                self.err(
                    proc,
                    f"expected writes to configuration {name[0].name()}.{name[1]} does not depend on loop iterations",
                )

        instr = proc.instr
        if instr:
            instr = LoopIR.instr(c_instr=instr.c_instr, c_global=instr.c_global)

        self.loopir_proc = LoopIR.proc(
            name=proc.name or "anon",
            args=args,
            preds=preds,
            body=body,
            instr=instr,
            srcinfo=proc.srcinfo,
        )

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError(
                "Errors occurred during typechecking:\n" + "\n".join(self.errors)
            )

    def get_loopir(self):
        return self.loopir_proc

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    def check_stmts(self, body):
        assert len(body) > 0 or self.uast_proc.instr
        stmts = []
        for s in body:
            stmts += self.check_single_stmt(s)
        return stmts

    def check_access(self, node, nm, idx, lvalue=False):
        # check indexing
        idx = [self.check_e(i, is_index=True) for i in idx]
        for i in idx:
            if i.type != T.err and not i.type.is_indexable():
                self.err(i, f"cannot index with expression of type '{i.type}'")

        # check compatibility with buffer type
        typ = self.env[nm]
        if typ is T.err:
            pass
        elif typ.is_numeric():
            if len(idx) > len(typ.shape()):
                self.err(
                    node,
                    f"expected access of variable "
                    f"'{nm}' of type '{typ}' to have "
                    f"no more than {len(typ.shape())} "
                    f"indices, but {len(idx)} were found.",
                )
                typ = T.err
            elif lvalue and len(typ.shape()) != len(idx):
                self.err(
                    node,
                    f"expected lvalue access of variable "
                    f"'{nm}' of type '{typ}' to have "
                    f"exactly {len(typ.shape())} "
                    f"indices, but {len(idx)} were found.",
                )
                typ = T.err
            elif len(idx) > 0:
                assert len(idx) == len(typ.shape())
                typ = typ.basetype()
            # else if len(idx) == 0 then just fall through

        elif lvalue:
            self.err(
                node,
                f"cannot assign/reduce to '{nm}', "
                f"a non-numeric variable of type '{typ}'",
            )
            typ = T.err
        elif len(idx) > 0:
            self.err(node, f"cannot index a variable of type '{typ}'")

        return idx, typ

    def check_single_stmt(self, stmt):
        if isinstance(stmt, UAST.FreshAssign):
            rhs = self.check_e(stmt.rhs, allow_special_window=True)

            # We see a statement of the form
            #   nm = ...
            # we have a windowing statement
            #   nm = x[...]
            # It doesn't make sense to have a general freshassign,
            # when we don't know which memory to allocate.
            if rhs.type == T.err:
                self.env[stmt.name] = T.err
                return []
            elif isinstance(rhs.type, T.Window):
                assert isinstance(rhs, LoopIR.WindowExpr)
                self.env[stmt.name] = rhs.type
                return [
                    LoopIR.WindowStmt(
                        stmt.name, rhs, stmt.rhs.special_window, stmt.srcinfo
                    )
                ]
            else:
                self.err(
                    stmt,
                    f"unable to disambiguate assignment to "
                    f"undefined variable '{stmt.name}'",
                )
                self.env[stmt.name] = T.err
                return []

        if isinstance(stmt, (UAST.Assign, UAST.Reduce)):
            rhs = self.check_e(stmt.rhs)
            if rhs.type != T.err and not rhs.type.is_real_scalar():
                self.err(rhs, f"cannot assign/reduce a '{rhs.type}' type value")

            idx, typ = self.check_access(stmt, stmt.name, stmt.idx, lvalue=True)
            assert typ.is_real_scalar() or typ is T.err

            IRnode = LoopIR.Assign if isinstance(stmt, UAST.Assign) else LoopIR.Reduce
            return [IRnode(stmt.name, typ, idx, rhs, stmt.srcinfo)]

        elif isinstance(stmt, UAST.WriteConfig):
            # Check that field is in config
            if not stmt.config.has_field(stmt.field):
                self.err(
                    stmt.field,
                    f"expected '{stmt.field}'  to be a field "
                    f"in config '{stmt.config.name()}'",
                )

            ftyp = stmt.config.lookup_type(stmt.field)
            rhs = self.check_e(
                stmt.rhs,
                is_index=ftyp.is_indexable() or ftyp.is_stridable() or ftyp == T.bool,
            )

            if rhs.type != T.err:
                if ftyp.is_real_scalar():
                    if not rhs.type.is_real_scalar():
                        self.err(
                            rhs,
                            f"expected a real scalar value, but "
                            f"got an expression of type {rhs.type}",
                        )
                elif ftyp.is_indexable():
                    if not rhs.type.is_indexable():
                        self.err(
                            rhs,
                            f"expected an index or size type "
                            f"expression, but got type {rhs.type}",
                        )
                elif ftyp == T.bool:
                    if rhs.type != T.bool:
                        self.err(
                            rhs,
                            f"expected a bool expression, but got type {rhs.type}",
                        )
                elif ftyp.is_stridable():
                    if not rhs.type.is_stridable():
                        self.err(
                            rhs,
                            f"expected a stride type expression, "
                            f"but got type {rhs.type}",
                        )
                else:
                    assert False, "bad case"

            return [LoopIR.WriteConfig(stmt.config, stmt.field, rhs, stmt.srcinfo)]
        elif isinstance(stmt, UAST.Pass):
            return [LoopIR.Pass(stmt.srcinfo)]
        elif isinstance(stmt, UAST.If):
            cond = self.check_e(stmt.cond, is_index=True)
            if cond.type != T.err and cond.type != T.bool:
                self.err(cond, f"expected a bool expression")
            body = self.check_stmts(stmt.body)
            ebody = []
            if len(stmt.orelse) > 0:
                ebody = self.check_stmts(stmt.orelse)
            return [LoopIR.If(cond, body, ebody, stmt.srcinfo)]

        elif isinstance(stmt, UAST.For):
            self.env[stmt.iter] = T.index

            # handle standard ParRanges
            parerr = (
                "currently supporting for-loops of the form:\n"
                "  'for _ in par(affine_expression, affine_expression):' and "
                "'for _ in seq(affine_expression, affine_expression):'"
            )

            if not isinstance(stmt.cond, (UAST.ParRange, UAST.SeqRange)):
                self.err(stmt.cond, parerr)

            lo = self.check_e(stmt.cond.lo, is_index=True)
            if lo.type != T.err and not lo.type.is_indexable():
                self.err(lo, "expected loop bound to be indexable.")
            hi = self.check_e(stmt.cond.hi, is_index=True)
            if hi.type != T.err and not hi.type.is_indexable():
                self.err(hi, "expected loop bound to be indexable.")

            body = self.check_stmts(stmt.body)
            if isinstance(stmt.cond, UAST.SeqRange):
                return [LoopIR.For(stmt.iter, lo, hi, body, LoopIR.Seq(), stmt.srcinfo)]
            elif isinstance(stmt.cond, UAST.ParRange):
                return [LoopIR.For(stmt.iter, lo, hi, body, LoopIR.Par(), stmt.srcinfo)]
            else:
                assert False, "bad case"

        elif isinstance(stmt, UAST.Alloc):
            typ = self.check_t(stmt.type)
            self.env[stmt.name] = typ
            mem = stmt.mem
            if mem is None:
                mem = DRAM
            return [LoopIR.Alloc(stmt.name, typ, mem, stmt.srcinfo)]

        elif isinstance(stmt, UAST.Call):
            args = [
                self.check_e(
                    call_a, is_index=sig_a.type in (T.size, T.index, T.stride, T.bool)
                )
                for call_a, sig_a in zip(stmt.args, stmt.f.args)
            ]

            check_call_types(self.err, args, stmt.f.args)

            return [LoopIR.Call(stmt.f, args, stmt.srcinfo)]
        else:
            assert False, f"not a loopir in check_stmts {type(stmt)}"

    def check_w_access(self, e, orig_hi):
        if isinstance(e, UAST.Point):
            pt = self.check_e(e.pt, is_index=True)
            if pt.type != T.err and not pt.type.is_indexable():
                self.err(pt, f"cannot index with expression of type '{pt.type}'")
            return LoopIR.Point(pt, e.srcinfo)

        elif isinstance(e, UAST.Interval):
            if e.lo is None:
                lo = LoopIR.Const(0, T.int, e.srcinfo)
            else:
                lo = self.check_e(e.lo, is_index=True)
                if lo.type != T.err and not lo.type.is_indexable():
                    self.err(lo, f"cannot index with expression of type '{lo.type}'")

            if e.hi is None:
                hi = orig_hi
            else:
                hi = self.check_e(e.hi, is_index=True)
                if hi.type != T.err and not hi.type.is_indexable():
                    self.err(hi, f"cannot index with expression of type '{hi.type}'")

            return LoopIR.Interval(lo, hi, e.srcinfo)

    def check_e(self, e, is_index=False, allow_special_window=False):
        if isinstance(e, UAST.Read):
            typ = self.env[e.name]
            # if we only partially accessed the base tensor/window,
            # then this is sugar for a windowing expression
            if typ.is_tensor_or_window() and 0 < len(e.idx) < len(typ.shape()):
                # expand to a windowing expression
                # x[i] == x[i,:]

                idxs = [UAST.Point(i, i.srcinfo) for i in e.idx] + [
                    UAST.Interval(None, None, e.srcinfo)
                    for _ in range(0, len(typ.shape()) - len(e.idx))
                ]

                desugared = UAST.WindowExpr(e.name, idxs, None, e.srcinfo)
                return self.check_e(desugared)

            # otherwise, we have a normal access
            else:
                idx, typ = self.check_access(e, e.name, e.idx, lvalue=False)
                return LoopIR.Read(e.name, idx, typ, e.srcinfo)

        elif isinstance(e, UAST.WindowExpr):
            in_typ = self.env[e.name]
            if not in_typ.is_tensor_or_window():
                self.err(
                    e,
                    f"cannot perform windowing on non-tensor, "
                    f"non-window type {e.base}",
                )
                return LoopIR.WindowExpr(e.name, [], T.err, e.srcinfo)

            if e.special_window is not None:
                # UAST has the optional special window as part of WindowExpr as
                # that's how it parses, but LoopIR has the special window as
                # part of WindowStmt since that matches the usage pattern
                # (can't construct special windows just anywhere)
                if not allow_special_window:
                    self.err(
                        e,
                        f"Can only create SpecialWindow as part of "
                        f"WindowStmt (W = t[idx...] @ SpecialWindow)",
                    )
                elif not in_typ.is_dense_tensor():
                    self.err(
                        e,
                        "Can only create SpecialWindow from a dense "
                        "tensor, not another window",
                    )

            in_shape = in_typ.shape()
            if len(in_shape) != len(e.idx):
                self.err(
                    e,
                    f"expected {len(in_shape)} indices for window "
                    f"but got {len(e.idx)}",
                )

            idx = [self.check_w_access(w, t) for w, t in zip(e.idx, in_shape)]
            w_typ = create_window_type(e.name, in_typ, idx)
            return LoopIR.WindowExpr(e.name, idx, w_typ, e.srcinfo)

        elif isinstance(e, UAST.Const):
            ty = {float: T.R, bool: T.bool, int: T.int if is_index else T.R}.get(
                type(e.val)
            )
            if not ty:
                self.err(
                    e,
                    f"literal of unexpected type '{type(e.val)}' and "
                    f"value: {e.val}",
                )
                return LoopIR.Const(0, T.err, e.srcinfo)
            return LoopIR.Const(e.val, ty, e.srcinfo)

        elif isinstance(e, UAST.USub):
            arg = self.check_e(e.arg, is_index=is_index)
            if arg.type.is_real_scalar() or arg.type.is_indexable():
                if isinstance(arg, LoopIR.Const):
                    return LoopIR.Const(-arg.val, arg.type, e.srcinfo)
                else:
                    return LoopIR.USub(arg, arg.type, e.srcinfo)
            elif arg.type != T.err:
                self.err(e, f"cannot negate expression of type '{arg.type}'")
            return LoopIR.Const(0, T.err, e.srcinfo)

        elif isinstance(e, UAST.BinOp):
            lhs = self.check_e(e.lhs, is_index=is_index)
            rhs = self.check_e(e.rhs, is_index=is_index)
            typ = T.err
            if lhs.type == T.err or rhs.type == T.err:
                typ = T.err
            elif e.op in ("and", "or"):
                for operand in (lhs, rhs):
                    if operand.type is not T.bool:
                        self.err(operand, "expected 'bool' argument to logical op")
                typ = T.bool
            elif e.op == "==" and (
                (lhs.type == T.bool and rhs.type == T.bool)
                or (
                    lhs.type.is_stridable()
                    and rhs.type.is_stridable()
                    and not (lhs.type == T.int and rhs.type == T.int)
                )
            ):
                typ = T.bool
            elif e.op in ("<", "<=", "==", ">", ">="):
                for operand in (lhs, rhs):
                    if not operand.type.is_indexable():
                        self.err(
                            operand,
                            f"expected 'index' or 'size' argument to "
                            f"comparison op: {e.op}",
                        )
                typ = T.bool
            elif e.op in ("+", "-", "*", "/", "%"):
                if lhs.type.is_real_scalar():
                    if not rhs.type.is_real_scalar():
                        self.err(rhs, "expected scalar type")
                        typ = T.err
                    elif e.op == "%":
                        self.err(e, "cannot compute modulus of 'R' values")
                        typ = T.err
                    else:
                        typ = lhs.type
                elif rhs.type.is_real_scalar():
                    self.err(lhs, "expected scalar type")
                elif lhs.type == T.bool or rhs.type == T.bool:
                    node = lhs if lhs.type == T.bool else rhs
                    self.err(node, "cannot perform arithmetic on 'bool' values")
                    typ = T.err
                elif lhs.type == T.stride or rhs.type == T.stride:
                    node = lhs if lhs.type == T.bool else rhs
                    self.err(node, "cannot perform arithmetic on 'stride' values")
                    typ = T.err
                elif lhs.type.is_tensor_or_window() or rhs.type.is_tensor_or_window():
                    self.err(lhs, "cannot perform arithmetic on tensors")
                    typ = T.err
                else:
                    assert lhs.type.is_indexable()
                    assert rhs.type.is_indexable()
                    if e.op == "/" or e.op == "%":
                        if rhs.type != T.int or not isinstance(rhs, LoopIR.Const):
                            self.err(
                                rhs,
                                "cannot divide or modulo by a " "non-constant value",
                            )
                            typ = T.err
                        elif rhs.val <= 0:
                            self.err(
                                rhs,
                                "cannot divide or modulo by zero "
                                "or a negative value",
                            )
                            typ = T.err

                        typ = lhs.type
                    elif e.op == "*":
                        if lhs.type == T.int:
                            typ = rhs.type
                        elif rhs.type == T.int:
                            typ = lhs.type
                        else:
                            self.err(
                                e,
                                "cannot multiply two non-constant "
                                "indexing/sizing expressions, since "
                                "the result would be non-affine",
                            )
                            typ = T.err
                    else:  # + or -
                        if lhs.type == T.index or rhs.type == T.index:
                            typ = T.index
                        elif lhs.type == T.size or rhs.type == T.size:
                            typ = T.size
                        else:
                            typ = T.int

            else:
                assert False, f"bad op: '{e.op}'"

            return LoopIR.BinOp(e.op, lhs, rhs, typ, e.srcinfo)

        elif isinstance(e, UAST.Extern):

            args = [self.check_e(a) for a in e.args]

            try:
                typ = e.f.typecheck(args)
            except Extern_Typecheck_Error as err:
                typ = T.err
                self.err(e, str(err))

            return LoopIR.Extern(e.f, args, typ, e.srcinfo)

        elif isinstance(e, UAST.StrideExpr):
            idx, typ = self.check_access(e, e.name, [], lvalue=False)
            assert len(idx) == 0
            if typ == T.err:
                pass
            elif not typ.is_tensor_or_window():
                self.err(e, f"expected {e.name} to be a tensor or window")
            else:
                shape = typ.shape()
                if not (0 <= e.dim < len(shape)):
                    self.err(
                        e,
                        f"expected {e.dim} to be in-bounds "
                        f"(i.e. 0 <= {e.dim} < {len(shape)})",
                    )

            return LoopIR.StrideExpr(e.name, e.dim, T.stride, e.srcinfo)

        elif isinstance(e, UAST.ParRange):
            assert False, (
                "parser should not place ParRange anywhere "
                "outside of a for-loop condition"
            )
        elif isinstance(e, UAST.ReadConfig):
            if not e.config.has_field(e.field):
                self.err(
                    e.field,
                    f"'{e.field}' has to be a field in config '{e.config.name()}'",
                )

            ftyp = e.config.lookup_type(e.field)
            return LoopIR.ReadConfig(e.config, e.field, ftyp, e.srcinfo)
        else:
            assert False, "not a LoopIR in check_e"

    _typ_table = {
        UAST.Num: T.R,
        UAST.F16: T.f16,
        UAST.F32: T.f32,
        UAST.F64: T.f64,
        UAST.INT8: T.int8,
        UAST.UINT8: T.uint8,
        UAST.UINT16: T.uint16,
        UAST.INT32: T.int32,
        UAST.Bool: T.bool,
        UAST.Int: T.int,
        UAST.Size: T.size,
        UAST.Index: T.index,
        UAST.Stride: T.stride,
    }

    def check_t(self, typ):
        if type(typ) in TypeChecker._typ_table:
            return TypeChecker._typ_table[type(typ)]
        elif isinstance(typ, UAST.Tensor):
            hi = [self.check_e(h, is_index=True) for h in typ.hi]
            sub_typ = self.check_t(typ.type)
            for h in hi:
                if not h.type.is_indexable():
                    self.err(
                        h,
                        "expected array size expression "
                        "to have type 'size' or type 'index'",
                    )
            return T.Tensor(hi, typ.is_window, sub_typ)
        else:
            assert False, "bad case"
