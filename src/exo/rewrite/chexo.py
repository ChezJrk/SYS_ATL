from typing import Optional
from ..core.LoopIR import LoopIR, T
from dataclasses import dataclass
from ..core.prelude import Sym, SrcInfo
from ..core.memory import DRAM
from ..backend.LoopIR_interpreter import run_interpreter
import numpy as np
from .new_eff import SchedulingError
from .constraint_solver import (
    ConjunctionConstraint,
    ConstraintTerm,
    GenericConstraint,
    Constraint,
    ConstraintMaker,
)


class LoopIRVisitor:
    def visit(self, node):
        self.visit_generic(node)

    def visit_generic(self, node):
        if (
            isinstance(node, LoopIR.proc)
            or isinstance(node, LoopIR.instr)
            or isinstance(node, LoopIR.fnarg)
            or isinstance(node, LoopIR.stmt)
            or isinstance(node, LoopIR.loop_mode)
            or isinstance(node, LoopIR.expr)
            or isinstance(node, LoopIR.w_access)
            or isinstance(node, LoopIR.type)
        ):
            for field_name in dir(node):
                if not field_name.startswith("_"):
                    field = getattr(node, field_name)
                    if isinstance(field, list):
                        for child in field:
                            self.visit(child)
                    else:
                        self.visit(field)


@dataclass
class TypeVisitor(LoopIRVisitor):
    type_map: dict[Sym, LoopIR.type]

    def visit(self, node):
        if isinstance(node, LoopIR.For):
            self.type_map[node.iter] = T.Index()
            self.visit_generic(node)
        elif isinstance(node, LoopIR.Alloc):
            self.type_map[node.name] = node.type
        elif isinstance(node, LoopIR.WindowStmt):
            self.type_map[node.name] = node.rhs.type
        elif isinstance(node, LoopIR.fnarg):
            self.type_map[node.name] = node.type
        else:
            self.visit_generic(node)


@dataclass
class UsedVariableVisitor(LoopIRVisitor):
    used_vars: set[Sym]

    def visit(self, node):
        if isinstance(node, Sym):
            self.used_vars.add(node)
        else:
            self.visit_generic(node)


def get_free_variables(type_map, fragment):
    fragment_type_visitor = TypeVisitor({})
    fragment_var_visitor = UsedVariableVisitor(set())
    for stmt in fragment:
        fragment_type_visitor.visit(stmt)
        fragment_var_visitor.visit(stmt)
    for var in fragment_var_visitor.used_vars - fragment_type_visitor.type_map.keys():
        fragment_var_visitor.visit(type_map[var])
    return {
        var: type_map[var]
        for var in fragment_var_visitor.used_vars
        - fragment_type_visitor.type_map.keys()
    }


def eval_tensor_dimension(dim_expr, control_values):
    if isinstance(dim_expr, LoopIR.Read):
        return control_values[dim_expr.name]
    elif isinstance(dim_expr, LoopIR.Const):
        return dim_expr.val
    elif isinstance(dim_expr, LoopIR.USub):
        return -eval_tensor_dimension(dim_expr.arg)
    elif isinstance(dim_expr, LoopIR.BinOp):
        lhs, rhs = eval_tensor_dimension(dim_expr.lhs), eval_tensor_dimension(
            dim_expr.rhs
        )
        if dim_expr.op == "+":
            return lhs + rhs
        elif dim_expr.op == "-":
            return lhs - rhs
        elif dim_expr.op == "*":
            return lhs * rhs
        elif dim_expr.op == "/":
            if isinstance(lhs, int) and isinstance(rhs, int):
                # this is what was here before and without the rhs check
                # counter example of why this is wrong -3 / 2 == -1 in C and 0 in this impl
                # return (lhs + rhs - 1) // rhs
                return int(lhs / rhs)
            else:
                return lhs / rhs
        elif dim_expr.op == "%":
            return lhs % rhs
        elif dim_expr.op == "==":
            return lhs == rhs
        elif dim_expr.op == "<":
            return lhs < rhs
        elif dim_expr.op == ">":
            return lhs > rhs
        elif dim_expr.op == "<=":
            return lhs <= rhs
        elif dim_expr.op == ">=":
            return lhs >= rhs
        elif dim_expr.op == "and":
            return lhs and rhs
        elif dim_expr.op == "or":
            return lhs or rhs
    else:
        assert False, "unexpected expression type in tensor dimension"


CONTROL_VAL_BOUND = 16
INT_BOUND = 128
FLOAT_BOUND = 32


def collect_path_constraints(cursor, cm: ConstraintMaker) -> GenericConstraint:
    cur = cursor
    result = Constraint(())
    while cur.depth() != 0:
        if isinstance(cur._node, LoopIR.For):
            result = ConjunctionConstraint(
                ConjunctionConstraint(
                    result,
                    Constraint(
                        (ConstraintTerm(1, (cur._node.iter,)),)
                        + tuple(
                            term.negate()
                            for term in cm.make_constraint_terms(cur._node.lo)
                        )
                    ),
                ),
                Constraint(
                    (ConstraintTerm(-1, (cur._node.iter,)),)
                    + cm.make_constraint_terms(cur._node.hi)
                    + (ConstraintTerm(-1, ()),)
                ),
            )
        elif isinstance(cur._node, LoopIR.If):
            result = ConjunctionConstraint(result, cm.make_constraint(cur._node.cond))
        cur = cur.parent()
    return result


def generate_args(args, constraint: Constraint, cm: ConstraintMaker):
    arg_values = {}
    control_values = {}
    assignments = cm.solve_constraint(constraint, CONTROL_VAL_BOUND)
    for arg in args:
        if not arg.type.is_numeric():
            if arg.name in assignments:
                val = assignments[arg.name]
            elif isinstance(arg.type, T.Bool):
                val = np.random.randint(0, CONTROL_VAL_BOUND) < CONTROL_VAL_BOUND / 2
            else:
                val = np.random.randint(0, CONTROL_VAL_BOUND)
            control_values[arg.name] = val
            arg_values[str(arg.name)] = val

    for arg in args:
        if arg.type.is_numeric():
            basetype = arg.type.basetype()
            if isinstance(basetype, (T.F32, T.Num)):
                dtype = np.float32
            elif isinstance(basetype, T.F16):
                dtype = np.float16
            elif isinstance(basetype, T.F64):
                dtype = np.float64
            elif isinstance(basetype, T.INT8):
                dtype = np.int8
            elif isinstance(basetype, T.INT32):
                dtype = np.int32
            elif isinstance(basetype, T.UINT8):
                dtype = np.uint8
            elif isinstance(basetype, T.UINT16):
                dtype = np.uint16

            if arg.type.is_real_scalar():
                shape = (1,)
            else:
                shape = tuple(
                    eval_tensor_dimension(dim_expr, control_values)
                    for dim_expr in arg.type.shape()
                )
            if dtype in [np.int8, np.int32]:
                arg_values[str(arg.name)] = np.random.randint(
                    -INT_BOUND, INT_BOUND, shape, dtype=dtype
                )
            elif dtype in [np.uint8, np.uint16]:
                arg_values[str(arg.name)] = np.random.randint(
                    0, INT_BOUND, shape, dtype=dtype
                )
            elif dtype in [np.float16, np.float32, np.float64]:
                arg_values[str(arg.name)] = (
                    np.random.rand(*shape) * FLOAT_BOUND
                ).astype(dtype)

    return arg_values


TEST_CASE_BOUND = 10


def fuzz_reorder_stmts(s1, s2):
    proc = s1.get_root()
    proc_type_visitor = TypeVisitor({})
    proc_type_visitor.visit(proc)
    cm = ConstraintMaker(proc_type_visitor.type_map)
    constraint = Constraint(())
    for pred in proc.preds:
        constraint = ConjunctionConstraint(constraint, cm.make_constraint(pred))
    constraint = ConjunctionConstraint(constraint, collect_path_constraints(s1, cm))
    args = [
        LoopIR.fnarg(name=var, type=arg_type, mem=DRAM, srcinfo=SrcInfo("", 0))
        for var, arg_type in get_free_variables(
            proc_type_visitor.type_map, [s1._node, s2._node]
        ).items()
    ]
    args = [arg for arg in args if not arg.type.is_numeric()] + [
        arg for arg in args if arg.type.is_numeric()
    ]
    for _ in range(TEST_CASE_BOUND):
        arg_vals1 = generate_args(args, constraint, cm)
        arg_vals2 = {
            key: val.copy() if isinstance(val, np.ndarray) else val
            for key, val in arg_vals1.items()
        }

        run_interpreter(
            LoopIR.proc(
                name=proc.name,
                args=args,
                preds=[],
                body=[s1._node, s2._node],
                instr=None,
                srcinfo=proc.srcinfo,
            ),
            arg_vals1,
        )
        run_interpreter(
            LoopIR.proc(
                name=proc.name,
                args=args,
                preds=[],
                body=[s2._node, s1._node],
                instr=None,
                srcinfo=proc.srcinfo,
            ),
            arg_vals2,
        )
        for x in arg_vals1:
            if not np.allclose(arg_vals1[x], arg_vals2[x]):
                raise SchedulingError("mismatch found")
