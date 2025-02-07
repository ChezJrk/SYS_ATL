from dataclasses import dataclass
from typing import Union, Optional

from exo.core.prelude import Sym
from ..core.LoopIR import LoopIR, T
import numpy as np


@dataclass
class Range:
    lower_bound: Optional[int]
    upper_bound: Optional[int]

    def intersect(self, other):
        def wrap_none(option):
            return [option] if option is not None else []

        lbs = [*wrap_none(self.lower_bound), *wrap_none(other.lower_bound)]
        ubs = [*wrap_none(self.upper_bound), *wrap_none(other.upper_bound)]
        return Range(
            None if len(lbs) == 0 else max(lbs),
            None if len(ubs) == 0 else min(ubs),
        )


def simplify_disjunction(ranges: tuple[Range]) -> tuple[Range]:
    bounds: list[tuple[Range, bool]] = []
    for r in ranges:
        if (
            r.lower_bound is None
            or r.upper_bound is None
            or r.lower_bound <= r.upper_bound
        ):
            bounds.append((r, False))
            bounds.append((r, True))

    def key(pair: tuple[Range, bool]) -> tuple[int, int, bool]:
        r, is_upper = pair
        if is_upper and r.upper_bound is None:
            return (1, 0, is_upper)
        if not is_upper and r.lower_bound is None:
            return (-1, 0, is_upper)
        return (0, r.upper_bound if is_upper else r.lower_bound, is_upper)

    bounds.sort(key=key)

    nest_depth = 0
    current_lower: Optional[int] = None
    new_ranges: list[Range] = []
    for r, is_upper in bounds:
        if nest_depth == 0:
            assert not is_upper
            current_lower = r.lower_bound
        nest_depth += -1 if is_upper else 1
        if nest_depth == 0:
            assert is_upper
            new_ranges.append(Range(current_lower, r.upper_bound))
    return tuple(new_ranges)


@dataclass
class ConstraintTerm:
    coefficient: int
    syms: tuple[Sym]

    def negate(self) -> "ConstraintTerm":
        return ConstraintTerm(-self.coefficient, self.syms)

    def multiply(self, other) -> "ConstraintTerm":
        return ConstraintTerm(
            self.coefficient * other.coefficient, self.syms + other.syms
        )

    def apply_assignments(
        self, assignments: dict[Sym, int], target_sym: Sym
    ) -> Optional[tuple[int, bool]]:
        is_const = True
        acc = self.coefficient
        for sym in self.syms:
            if sym == target_sym:
                is_const = False
            else:
                if sym not in assignments:
                    return None
                acc *= assignments[sym]
        return (acc, is_const)


@dataclass
class Constraint:
    terms: tuple[ConstraintTerm]

    def apply_assignments(
        self, assignments: dict[Sym, int], target_sym: Sym
    ) -> tuple[Range]:
        offset, scale = 0, 0
        for term in self.terms:
            assign_result = term.apply_assignments(assignments, target_sym)
            if assign_result is None:
                return (Range(None, None),)
            else:
                acc, is_const = assign_result
                if is_const:
                    offset += acc
                else:
                    scale += acc
        if scale == 0:
            if offset >= 0:
                return (Range(None, None),)
            else:
                return (Range(0, -1),)
        elif scale > 0:
            return (Range(-offset / scale, None),)
        else:
            return (Range(None, -offset / scale),)

    def collect_syms(self) -> frozenset[Sym]:
        return frozenset(sym for term in self.terms for sym in term.syms)

    def pretty_print(self) -> str:
        return (
            " + ".join(
                [
                    f"{' * '.join([str(term.coefficient)] + [repr(sym) for sym in term.syms])}"
                    for term in self.terms
                ]
            )
            + " >= 0"
        )


GenericConstraint = Union[Constraint, "ConjunctionConstraint", "DisjunctionConstraint"]


@dataclass
class ConjunctionConstraint:
    lhs: GenericConstraint
    rhs: GenericConstraint

    def apply_assignments(
        self, assignments: dict[Sym, int], target_sym: Sym
    ) -> tuple[Range]:
        lhs_ranges = self.lhs.apply_assignments(assignments, target_sym)
        rhs_ranges = self.rhs.apply_assignments(assignments, target_sym)
        return simplify_disjunction(
            tuple(
                lhs_range.intersect(rhs_range)
                for lhs_range in lhs_ranges
                for rhs_range in rhs_ranges
            )
        )

    def collect_syms(self) -> frozenset[Sym]:
        return self.lhs.collect_syms() | self.rhs.collect_syms()

    def pretty_print(self) -> str:
        return f"({self.lhs.pretty_print()}) and ({self.rhs.pretty_print()})"


@dataclass
class DisjunctionConstraint:
    lhs: Constraint
    rhs: Constraint

    def apply_assignments(
        self, assignments: dict[Sym, int], target_sym: Sym
    ) -> tuple[Range]:
        lhs_ranges = self.lhs.apply_assignments(assignments, target_sym)
        rhs_ranges = self.rhs.apply_assignments(assignments, target_sym)
        return simplify_disjunction(lhs_ranges + rhs_ranges)

    def collect_syms(self) -> frozenset[Sym]:
        return self.lhs.collect_syms() | self.rhs.collect_syms()

    def pretty_print(self) -> str:
        return f"({self.lhs.pretty_print()}) or ({self.rhs.pretty_print()})"


class ConstraintMaker:
    def __init__(self, type_map: dict[Sym, LoopIR.type]):
        self.nonneg_vars = set(
            sym
            for sym, sym_type in type_map.items()
            if isinstance(sym_type, (T.Size, T.Index))
        )
        self.bool_vars = set(
            sym for sym, sym_type in type_map.items() if isinstance(sym_type, (T.Bool))
        )
        self.stride_dummies: dict[tuple[Sym, int], Sym] = {}

    def make_constraint_terms(self, expr: LoopIR.expr) -> tuple[ConstraintTerm]:
        # expect that expr is int type
        if isinstance(expr, LoopIR.Read):
            assert (
                len(expr.idx) == 0
            ), "indexing not supported in assertions (yet, todo)"
            return (ConstraintTerm(1, (expr.name,)),)
        elif isinstance(expr, LoopIR.Const):
            return (ConstraintTerm(expr.val, ()),)
        elif isinstance(expr, LoopIR.USub):
            return tuple(term.negate() for term in self.make_constraint_terms(expr.arg))
        elif isinstance(expr, LoopIR.BinOp):
            # TODO: support mod and div using extra variables
            lhs_terms = self.make_constraint_terms(expr.lhs)
            rhs_terms = self.make_constraint_terms(expr.rhs)
            if expr.op == "+":
                return lhs_terms + rhs_terms
            elif expr.op == "-":
                return lhs_terms + tuple(term.negate() for term in rhs_terms)
            elif expr.op == "*":
                return tuple(
                    lhs_term.multiply(rhs_term)
                    for lhs_term in lhs_terms
                    for rhs_term in rhs_terms
                )
            else:
                assert False, f"unsupported op in assertion: {expr.op}"
        elif isinstance(expr, LoopIR.StrideExpr):
            if (expr.name, expr.dim) not in self.stride_dummies:
                new_sym = Sym("stride")
                self.stride_dummies[(expr.name, expr.dim)] = new_sym
                self.nonneg_vars.add(new_sym)
            dummy = self.stride_dummies[(expr.name, expr.dim)]
            return (ConstraintTerm(1, (dummy,)),)
        else:
            assert False, f"unsupported expr"

    def make_constraint(
        self,
        expr: LoopIR.expr,
    ) -> GenericConstraint:
        # expect that expr is bool type
        if isinstance(expr, LoopIR.BinOp):
            if expr.op == "and":
                return ConjunctionConstraint(
                    self.make_constraint(expr.lhs), self.make_constraint(expr.rhs)
                )
            elif expr.op == "or":
                return DisjunctionConstraint(
                    self.make_constraint(expr.lhs), self.make_constraint(expr.rhs)
                )
            elif expr.op == "<":
                return Constraint(
                    self.make_constraint_terms(expr.rhs)
                    + tuple(
                        term.negate() for term in self.make_constraint_terms(expr.lhs)
                    )
                    + (ConstraintTerm(-1, ()),)
                )
            elif expr.op == ">":
                return Constraint(
                    self.make_constraint_terms(expr.lhs)
                    + tuple(
                        term.negate() for term in self.make_constraint_terms(expr.rhs)
                    )
                    + (ConstraintTerm(-1, ()),)
                )
            elif expr.op == "<=":
                return Constraint(
                    self.make_constraint_terms(expr.rhs)
                    + tuple(
                        term.negate() for term in self.make_constraint_terms(expr.lhs)
                    )
                )
            elif expr.op == ">=":
                return Constraint(
                    self.make_constraint_terms(expr.lhs)
                    + tuple(
                        term.negate() for term in self.make_constraint_terms(expr.rhs)
                    )
                )
            elif expr.op == "==":
                return ConjunctionConstraint(
                    Constraint(
                        self.make_constraint_terms(expr.rhs)
                        + tuple(
                            term.negate()
                            for term in self.make_constraint_terms(expr.lhs)
                        )
                    ),
                    Constraint(
                        self.make_constraint_terms(expr.lhs)
                        + tuple(
                            term.negate()
                            for term in self.make_constraint_terms(expr.rhs)
                        )
                    ),
                )
            else:
                assert False, "boolean ops expected"
        elif isinstance(expr, LoopIR.Read):
            assert len(expr.idx) == 0, "cannot index into boolean"
            return ConjunctionConstraint(
                Constraint((ConstraintTerm(1, expr.name), ConstraintTerm(-1, ()))),
                Constraint((ConstraintTerm(-1, expr.name), ConstraintTerm(1, ()))),
            )
        else:
            assert False, "only boolean expected"

    def solve_constraint(
        self, constraint: GenericConstraint, bound: int, seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed=seed)
        assignments = {}
        syms = constraint.collect_syms()

        bounding_range = Range(-bound, bound)

        def solve_recursive() -> bool:
            sym_domains = [
                (
                    tuple(
                        sym_range.intersect(
                            Range(0, bound)
                            if sym in self.nonneg_vars
                            else (
                                Range(0, 1)
                                if sym in self.bool_vars
                                else Range(-bound, bound)
                            )
                        )
                        for sym_range in constraint.apply_assignments(assignments, sym)
                    ),
                    sym,
                )
                for sym in syms - assignments.keys()
            ]
            if len(sym_domains) == 0:
                return True
            else:

                def domain_size(sym_domain: tuple[Range]) -> int:
                    return sum(
                        sym_range.upper_bound - sym_range.lower_bound + 1
                        for sym_range in sym_domain
                    )

                sym_domains.sort(key=lambda sym_domain: domain_size(sym_domain[0]))
                sym_domain, sym = sym_domains[0]
                if len(sym_domain) == 0:
                    return False
                range_sizes = np.array(
                    [
                        sym_range.upper_bound - sym_range.lower_bound + 1
                        for sym_range in sym_domain
                    ]
                )
                chosen_range = np.random.choice(
                    sym_domain, p=range_sizes / np.sum(range_sizes)
                )
                assignments[sym] = np.random.randint(
                    chosen_range.lower_bound, chosen_range.upper_bound + 1
                )
                return solve_recursive()

        while not solve_recursive():
            assignments = {}
        return assignments
