from collections import ChainMap, defaultdict

import numpy as np

from ..core.LoopIR import LoopIR
from ..core.LoopIR import T
from ..core.prelude import *

from .parallel_analysis import ParallelAnalysis
from .prec_analysis import PrecisionAnalysis
from .win_analysis import WindowAnalysis
from .mem_analysis import MemoryAnalysis

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Interpreter


# method copied from Python ChainMap docs https://docs.python.org/3/library/collections.html#collections.ChainMap
# to delete items from parent maps
@extclass(ChainMap)
def __delitem__(self, key):
    for mapping in self.maps:
        if key in mapping:
            del mapping[key]
            return
    raise KeyError(key)


def _eshape(typ, env):
    return tuple(r if is_pos_int(r) else env[r] for r in typ.shape())


def run_interpreter(proc, kwargs):
    Interpreter(proc, kwargs)


class Interpreter:
    def __init__(self, proc, kwargs, use_randomization=False):
        if not isinstance(proc, LoopIR.proc):
            raise TypeError(f"Expected {proc.name} to be of type proc")

        self.env = ChainMap()
        self.use_randomization = use_randomization
        self.ctxt = defaultdict(dict)

        self.eval_proc(proc, kwargs)

    def _new_scope(self):
        self.env = self.env.new_child()

    def _del_scope(self):
        self.env = self.env.parents

    def typecheck_input_buffer(self, proc_arg, kwargs):
        nm = proc_arg.name
        if not proc_arg.type.is_numeric():
            raise TypeError(f"arg {nm} is expected to be numeric")

        basetype = proc_arg.type.basetype()
        buf = kwargs[str(proc_arg.name)]

        pre = f"bad argument '{nm}'"
        if not isinstance(buf, np.ndarray):
            raise TypeError(f"{pre}: expected numpy.ndarray")

        if isinstance(basetype, T.F32):
            if buf.dtype != np.float32:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if isinstance(basetype, T.F16):
            if buf.dtype != np.float16:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if isinstance(basetype, (T.F64, T.Num)):
            if buf.dtype != np.float64:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if isinstance(basetype, T.INT8):
            if buf.dtype != np.int8:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if isinstance(basetype, T.INT32):
            if buf.dtype != np.int32:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if isinstance(basetype, T.UINT8):
            if buf.dtype != np.uint8:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if isinstance(basetype, T.UINT16):
            if buf.dtype != np.uint16:
                raise TypeError(f"{pre}: received {buf.dtype} values")

        if proc_arg.type.is_real_scalar():
            if tuple(buf.shape) != (1,):
                raise TypeError(
                    f"{pre}: expected buffer of shape (1,), "
                    f"but got shape {tuple(buf.shape)}"
                )
        else:
            shape = self.eval_shape(proc_arg.type)
            if shape != tuple(buf.shape):
                raise TypeError(
                    f"{pre}: expected buffer of shape {shape}, "
                    f"but got shape {tuple(buf.shape)}"
                )

    def eval_proc(self, proc, kwargs):
        proc = ParallelAnalysis().run(proc)
        proc = PrecisionAnalysis().run(proc)  # TODO: need this?
        proc = WindowAnalysis().apply_proc(proc)
        proc = MemoryAnalysis().run(proc)  # TODO: need this?

        for a in proc.args:
            if not str(a.name) in kwargs:
                raise TypeError(f"expected argument '{a.name}' to be supplied")

            if a.type is T.size:
                if not is_pos_int(kwargs[str(a.name)]):
                    raise TypeError(
                        f"expected size '{a.name}' to have positive integer value"
                    )
                self.env[a.name] = kwargs[str(a.name)]
            elif a.type is T.index:
                if type(kwargs[str(a.name)]) is not int:
                    raise TypeError(
                        f"expected index variable '{a.name}' to be an integer"
                    )
                self.env[a.name] = kwargs[str(a.name)]
            elif a.type is T.bool:
                if type(kwargs[str(a.name)]) is not bool:
                    raise TypeError(f"expected bool variable '{a.name}' to be a bool")
                self.env[a.name] = kwargs[str(a.name)]
            elif a.type is T.stride:
                if type(kwargs[str(a.name)]) is not int:
                    raise TypeError(
                        f"expected stride variable '{a.name}' to be an integer"
                    )
                self.env[a.name] = kwargs[str(a.name)]
            else:
                self.typecheck_input_buffer(a, kwargs)
                self.env[a.name] = kwargs[str(a.name)]

        # evaluate preconditions
        for pred in proc.preds:
            if isinstance(pred, LoopIR.Const):
                continue
            else:
                assert self.eval_e(pred), "precondition not satisfied"

        # eval statements
        self.eval_stmts(proc.body)

    def eval_stmts(self, stmts):
        for s in stmts:
            self.eval_s(s)

    def eval_s(self, s):
        if isinstance(s, LoopIR.Pass):
            pass
        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            lbuf = self.env[s.name]
            if len(s.idx) == 0:
                # lbuf = rhs
                idx = (0,)
            else:
                # lbuf[a0,a1,...] = rhs
                idx = tuple(self.eval_e(a) for a in s.idx)
            rhs = self.eval_e(s.rhs)
            if isinstance(s, LoopIR.Assign):
                lbuf[idx] = rhs
            else:
                lbuf[idx] += rhs

        elif isinstance(s, LoopIR.WriteConfig):
            nm = s.config.name()
            rhs = self.eval_e(s.rhs)
            self.ctxt[nm][s.field] = rhs

        elif isinstance(s, LoopIR.WindowStmt):
            # nm = rbuf[...]
            assert s.name not in self.env, "WindowStmt should be a fresh assignment"
            assert isinstance(
                s.rhs, LoopIR.WindowExpr
            ), "WindowStmt rhs should be WindowExpr"
            self.env[s.name] = self.eval_e(s.rhs)

        elif isinstance(s, LoopIR.If):
            cond = self.eval_e(s.cond)
            if cond:
                self._new_scope()
                self.eval_stmts(s.body)
                self._del_scope()
            if s.orelse and not cond:
                self._new_scope()
                self.eval_stmts(s.orelse)
                self._del_scope()

        elif isinstance(s, LoopIR.For):
            # future TODO: handle loop_mode
            lo = self.eval_e(s.lo)
            hi = self.eval_e(s.hi)
            assert self.use_randomization is False, "TODO: Implement Rand"
            self._new_scope()
            for itr in range(lo, hi):
                self.env[s.iter] = itr
                self.eval_stmts(s.body)
            self._del_scope()

        elif isinstance(s, LoopIR.Alloc):
            if s.type.is_real_scalar():
                self.env[s.name] = np.empty([1])
            else:
                size = self.eval_shape(s.type)
                # TODO: Maybe randomize?
                self.env[s.name] = np.empty(size)

        elif isinstance(s, LoopIR.Free):
            # use extension to chain map from python docs
            del self.env[s.name]

        elif isinstance(s, LoopIR.Call):
            argvals = [self.eval_e(a, call_arg=True) for a in s.args]
            argnames = [str(a.name) for a in s.f.args]
            kwargs = {nm: val for nm, val in zip(argnames, argvals)}
            self._new_scope()
            self.eval_proc(s.f, kwargs)
            self._del_scope()

        else:
            assert False, "bad statement case"

    def eval_e(self, e, call_arg=False):

        if isinstance(e, LoopIR.Read):
            buf = self.env[e.name]
            if call_arg or isinstance(buf, (int, bool)):
                # read without indices
                return buf
            else:
                idx = (0,) if len(e.idx) == 0 else tuple(self.eval_e(a) for a in e.idx)
                return buf[idx]

        elif isinstance(e, LoopIR.WindowExpr):
            buf = self.env[e.name]

            def stringify_w_access(a):
                if isinstance(a, LoopIR.Interval):
                    return f"{self.eval_e(a.lo)}:{self.eval_e(a.hi)}"
                elif isinstance(a, LoopIR.Point):
                    return f"{self.eval_e(a.pt)}"
                else:
                    assert False, "bad w_access case"

            # hack to handle interval indexes: LoopIR.Interval returns a string representing the interval
            idx = (
                ("0",)
                if len(e.idx) == 0
                else tuple(stringify_w_access(a) for a in e.idx)
            )
            res = eval(f"buf[{','.join(idx)}]")
            return res

        elif isinstance(e, LoopIR.Const):
            return e.val

        elif isinstance(e, LoopIR.BinOp):
            lhs, rhs = self.eval_e(e.lhs), self.eval_e(e.rhs)
            if e.op == "+":
                return lhs + rhs
            elif e.op == "-":
                return lhs - rhs
            elif e.op == "*":
                return lhs * rhs
            elif e.op == "/":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    # this is what was here before and without the rhs check
                    # counter example of why this is wrong -3 / 2 == -1 in C and 0 in this impl
                    # return (lhs + rhs - 1) // rhs
                    return int(lhs / rhs)
                else:
                    return lhs / rhs
            elif e.op == "%":
                return lhs % rhs
            elif e.op == "==":
                return lhs == rhs
            elif e.op == "<":
                return lhs < rhs
            elif e.op == ">":
                return lhs > rhs
            elif e.op == "<=":
                return lhs <= rhs
            elif e.op == ">=":
                return lhs >= rhs
            elif e.op == "and":
                return lhs and rhs
            elif e.op == "or":
                return lhs or rhs

        elif isinstance(e, LoopIR.USub):
            return -self.eval_e(e.arg)

        # BuiltIns don't go to the interpreter, they are just called (via call) like a proc
        # TODO Discuss to make sure
        # elif isinstance(e, LoopIR.BuiltIn):
        #     assert False, "Not implemented"
        # args = [self.eval_e(a) for a in e.args]
        # return e.f.interpret(args)

        elif isinstance(e, LoopIR.StrideExpr):
            buf = self.env[e.name]
            assert e.dim < len(buf.strides), "invalid dim in stride expression"
            # grammar guarantees int (not an expression)
            return int(buf.strides[e.dim] / buf.dtype.itemsize)

        elif isinstance(e, LoopIR.ReadConfig):
            nm = e.config.name()
            return self.ctxt[nm][e.field]

        else:
            print(e)
            assert False, "bad expression case"

    def eval_shape(self, typ):
        return tuple(self.eval_e(s) for s in typ.shape())
