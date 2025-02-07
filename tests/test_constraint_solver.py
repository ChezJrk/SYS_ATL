from __future__ import annotations
from exo.core.prelude import Sym

from exo.rewrite.constraint_solver import ConstraintMaker
from exo.core.LoopIR import LoopIR
from exo import proc
from exo.rewrite.chexo import TypeVisitor


def test_make_constraint(golden):
    @proc
    def foo(a: size, b: size, c: size):
        assert ((a * 4 + b > c) or (a <= 3)) and (b < 5)
        pass

    foo_type = TypeVisitor({})
    foo_type.visit(foo._loopir_proc)
    assert (
        golden
        == ConstraintMaker(foo_type.type_map)
        .make_constraint(foo._loopir_proc.preds[0])
        .pretty_print()
    )


def test_solve(golden):
    @proc
    def foo(a: size, b: size, c: size):
        assert ((a * 4 + b > c) or (a <= 3)) and (b < 5)
        pass

    foo_type = TypeVisitor({})
    foo_type.visit(foo._loopir_proc)
    cm = ConstraintMaker(foo_type.type_map)
    constraint = cm.make_constraint(foo._loopir_proc.preds[0])
    assert golden == ", ".join(
        [
            f"{str(sym)} = {val}"
            for sym, val in cm.solve_constraint(constraint, 16, 13).items()
        ]
    )
