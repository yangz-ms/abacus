import math

from calc.registry import register
from calc.helpers import _to_int
from calc.algebra import Polynomial
from calc.polytools import Calculator15, calc15
from calc.inequalities import (
    Interval, _merge_intervals, ALL_REALS, NO_SOLUTION,
    _find_inequality_op, _flip_op, _check_op,
    _parse_poly_from_expr, _solve_linear_inequality,
    _solve_abs_inequality, _detect_abs_inequality,
    solve_inequality as _solve_inequality_linear,
)


# ---------------------------------------------------------------------------
# Polynomial inequality solving (quadratic, cubic)
# ---------------------------------------------------------------------------

def _eval_poly_at(poly, x):
    """Evaluate polynomial at a point."""
    result = 0
    for i, c in enumerate(poly.coeffs):
        result += c * (x ** i)
    return result


def _solve_quadratic_inequality(poly, op):
    """Solve quadratic inequality by finding roots and testing intervals."""
    # Find roots of poly = 0
    roots = poly.solve()

    # Filter to real roots only
    real_roots = []
    for r in roots:
        if isinstance(r, complex):
            if abs(r.imag) < 1e-9:
                real_roots.append(_to_int(r.real))
            # Complex roots: polynomial doesn't cross zero at real axis
        else:
            real_roots.append(r)

    real_roots = sorted(real_roots)

    # Leading coefficient determines sign at +infinity
    leading = poly.coeffs[-1]
    if isinstance(leading, complex):
        raise Exception("Cannot solve inequality with complex coefficients")

    if not real_roots:
        # No real roots -- polynomial has constant sign
        test_val = _eval_poly_at(poly, 0)
        if _check_op(test_val, op):
            return ALL_REALS
        else:
            return NO_SOLUTION

    # Build test points for intervals
    intervals_to_test = []
    # (-inf, root0)
    if real_roots:
        test_point = real_roots[0] - 1
        intervals_to_test.append((float('-inf'), real_roots[0], False, test_point))
    # Between consecutive roots
    for i in range(len(real_roots) - 1):
        test_point = (real_roots[i] + real_roots[i+1]) / 2
        intervals_to_test.append((real_roots[i], real_roots[i+1], False, test_point))
    # (root_last, inf)
    if real_roots:
        test_point = real_roots[-1] + 1
        intervals_to_test.append((real_roots[-1], float('inf'), False, test_point))

    inclusive = op in ('<=', '>=')
    strict_op = '<' if op in ('<', '<=') else '>'

    result_intervals = []
    for lo, hi, _, test_point in intervals_to_test:
        val = _eval_poly_at(poly, test_point)
        if _check_op(val, strict_op):
            # Determine inclusivity at endpoints
            lo_inc = inclusive and lo != float('-inf') and lo in real_roots
            hi_inc = inclusive and hi != float('inf') and hi in real_roots
            result_intervals.append((lo, hi, lo_inc, hi_inc))

    # Also check if single root points are included (for <= or >=)
    if inclusive:
        for r in real_roots:
            # Check if this root is already covered by an interval endpoint
            already_covered = False
            for lo, hi, lo_inc, hi_inc in result_intervals:
                if (lo <= r <= hi):
                    already_covered = True
                    break
            if not already_covered:
                result_intervals.append((r, r, True, True))

    return Interval(_merge_intervals(result_intervals))


def _solve_polynomial_inequality(poly, op):
    """Solve poly <op> 0 for polynomial of degree <= 3."""
    poly._trim()
    deg = poly.degree()

    if deg == 0:
        val = poly.coeffs[0]
        if _check_op(val, op):
            return ALL_REALS
        return NO_SOLUTION

    if deg == 1:
        return _solve_linear_inequality(poly, op)

    if deg <= 3:
        return _solve_quadratic_inequality(poly, op)

    raise Exception(f"Cannot solve degree {deg} inequality")


def solve_polynomial_inequality(expression):
    """Solve a polynomial inequality expression. Returns interval notation string.

    This version handles linear, quadratic, and cubic polynomial inequalities,
    as well as absolute value inequalities.
    """
    expression = expression.strip()

    # Check for absolute value inequality
    abs_match = _detect_abs_inequality(expression)
    if abs_match:
        inner, op, rhs = abs_match
        # Use polynomial-level solver for abs value (handles quadratic inner exprs)
        return str(_solve_abs_inequality_poly(inner, op, rhs))

    # Find inequality operators
    ops = _find_inequality_op(expression)
    if not ops:
        raise Exception("No inequality operator found")

    # Compound inequality: a < expr < b
    if len(ops) == 2:
        pos1, op1, len1 = ops[0]
        pos2, op2, len2 = ops[1]

        left_str = expression[:pos1].strip()
        mid_str = expression[pos1+len1:pos2].strip()
        right_str = expression[pos2+len2:].strip()

        # Solve: left <op1> mid AND mid <op2> right
        left_poly = _parse_poly_from_expr(left_str)
        mid_poly = _parse_poly_from_expr(mid_str)
        right_poly = _parse_poly_from_expr(right_str)

        # left <op1> mid  =>  mid - left >/<op1_flipped> 0
        poly1 = mid_poly - left_poly
        flipped_op1 = _flip_op(op1)
        interval1 = _solve_polynomial_inequality(poly1, flipped_op1)

        # mid <op2> right  =>  mid - right <op2> 0
        poly2 = mid_poly - right_poly
        interval2 = _solve_polynomial_inequality(poly2, op2)

        return str(interval1.intersect(interval2))

    # Single inequality: left <op> right
    pos, op, length = ops[0]
    left_str = expression[:pos].strip()
    right_str = expression[pos+length:].strip()

    left_poly = _parse_poly_from_expr(left_str)
    right_poly = _parse_poly_from_expr(right_str)

    # Ensure variables match
    if not left_poly.is_constant() and not right_poly.is_constant():
        if left_poly.var != right_poly.var:
            right_poly.var = left_poly.var

    if left_poly.is_constant() and not right_poly.is_constant():
        left_poly.var = right_poly.var
    elif right_poly.is_constant() and not left_poly.is_constant():
        right_poly.var = left_poly.var

    poly = left_poly - right_poly
    return str(_solve_polynomial_inequality(poly, op))


def _solve_abs_inequality_poly(inner_expr, op, rhs_expr):
    """Solve |inner| <op> rhs using polynomial-level inequality solver.

    |expr| < a  =>  -a < expr < a
    |expr| > a  =>  expr < -a OR expr > a
    |expr| <= a =>  -a <= expr <= a
    |expr| >= a =>  expr <= -a OR expr >= a
    """
    inner_poly = _parse_poly_from_expr(inner_expr)
    rhs_poly = _parse_poly_from_expr(rhs_expr)

    if not rhs_poly.is_constant():
        raise Exception("Right side of absolute value inequality must be a constant")

    a = rhs_poly.constant_value()
    if isinstance(a, complex):
        raise Exception("Cannot solve inequality with complex bound")
    a = _to_int(a)

    if op in ('<', '<='):
        inclusive = (op == '<=')
        op_lower = '>=' if inclusive else '>'
        op_upper = '<=' if inclusive else '<'

        poly_lower = inner_poly + a
        interval_lower = _solve_polynomial_inequality(poly_lower, op_lower)

        poly_upper = inner_poly - a
        interval_upper = _solve_polynomial_inequality(poly_upper, op_upper)

        return interval_lower.intersect(interval_upper)

    elif op in ('>', '>='):
        inclusive = (op == '>=')
        op_lower = '<=' if inclusive else '<'
        op_upper = '>=' if inclusive else '>'

        poly_lower = inner_poly + a
        interval_lower = _solve_polynomial_inequality(poly_lower, op_lower)

        poly_upper = inner_poly - a
        interval_upper = _solve_polynomial_inequality(poly_upper, op_upper)

        return interval_lower.union(interval_upper)

    raise Exception(f"Unsupported operator: {op}")


# ---------------------------------------------------------------------------
# Calculator16 class - Polynomial Inequalities
# ---------------------------------------------------------------------------

class Calculator16(Calculator15):
    """Polynomial inequalities (quadratic, cubic).

    Inherits all tokenizer features including inequality operators from
    Calculator13 through the chain Calculator15 -> Calculator14 -> Calculator13.
    """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@register("calc16", description="Solve polynomial inequalities (quadratic, cubic)",
          short_desc="Polynomial Inequalities", group="solver",
          examples=["x^2-4<0", "x^2-4>=0", "x^3-x>0"],
          i18n={"zh": "\u591a\u9879\u5f0f\u4e0d\u7b49\u5f0f", "hi": "\u092c\u0939\u0941\u092a\u0926 \u0905\u0938\u092e\u093f\u0915\u093e\u090f\u0901", "es": "Desigualdades Polin\u00f3micas", "fr": "In\u00e9galit\u00e9s Polynomiales", "ar": "\u0645\u062a\u0628\u0627\u064a\u0646\u0627\u062a \u0643\u062b\u064a\u0631\u0627\u062a \u0627\u0644\u062d\u062f\u0648\u062f", "pt": "Desigualdades Polinomiais", "ru": "\u041f\u043e\u043b\u0438\u043d\u043e\u043c\u0438\u0430\u043b\u044c\u043d\u044b\u0435 \u043d\u0435\u0440\u0430\u0432\u0435\u043d\u0441\u0442\u0432\u0430", "ja": "\u591a\u9805\u5f0f\u4e0d\u7b49\u5f0f", "de": "Polynomische Ungleichungen"})
def calc16(expression):
    """Solve polynomial inequalities (quadratic, cubic)."""
    expression = expression.strip()

    # Check for inequality operators (at top level, not inside parens)
    ops = _find_inequality_op(expression)
    if ops:
        return solve_polynomial_inequality(expression)

    # No inequality operators found -- fall through to calc15
    return calc15(expression)
