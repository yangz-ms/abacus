import math

from calc.helpers import _to_int
from calc.algebra import Polynomial
from calc.systems import Calculator13, MultiPolynomial
from calc.polytools import Calculator14, calc14


# ---------------------------------------------------------------------------
# Interval class for solution output
# ---------------------------------------------------------------------------

class Interval:
    """Represents a solution interval or union of intervals."""

    def __init__(self, intervals):
        """intervals: list of (low, high, low_inclusive, high_inclusive) tuples."""
        self.intervals = sorted(intervals, key=lambda t: (t[0], t[1]))

    def __str__(self):
        if not self.intervals:
            return "no solution"
        parts = []
        for low, high, lo_inc, hi_inc in self.intervals:
            lo_bracket = '[' if lo_inc else '('
            hi_bracket = ']' if hi_inc else ')'
            lo_str = '-inf' if low == float('-inf') else _format_num(low)
            hi_str = 'inf' if high == float('inf') else _format_num(high)
            parts.append(f"{lo_bracket}{lo_str},{hi_str}{hi_bracket}")
        return ' U '.join(parts)

    def intersect(self, other):
        """Intersect two Interval objects."""
        result = []
        for a in self.intervals:
            for b in other.intervals:
                lo = max(a[0], b[0])
                hi = min(a[1], b[1])
                if lo > hi:
                    continue
                lo_inc = a[2] and b[2] if a[0] == b[0] else (a[2] if a[0] > b[0] else b[2])
                hi_inc = a[3] and b[3] if a[1] == b[1] else (a[3] if a[1] < b[1] else b[3])
                if lo == hi and not (lo_inc and hi_inc):
                    continue
                result.append((lo, hi, lo_inc, hi_inc))
        return Interval(result)

    def union(self, other):
        """Union two Interval objects."""
        combined = list(self.intervals) + list(other.intervals)
        return Interval(_merge_intervals(combined))


def _merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda t: (t[0], not t[2]))
    merged = [intervals[0]]
    for lo, hi, lo_inc, hi_inc in intervals[1:]:
        prev_lo, prev_hi, prev_lo_inc, prev_hi_inc = merged[-1]
        if lo < prev_hi or (lo == prev_hi and (lo_inc or prev_hi_inc)):
            new_hi = max(prev_hi, hi)
            if prev_hi == hi:
                new_hi_inc = prev_hi_inc or hi_inc
            elif hi > prev_hi:
                new_hi_inc = hi_inc
            else:
                new_hi_inc = prev_hi_inc
            new_lo_inc = prev_lo_inc or lo_inc if lo == prev_lo else prev_lo_inc
            merged[-1] = (prev_lo, new_hi, new_lo_inc, new_hi_inc)
        else:
            merged.append((lo, hi, lo_inc, hi_inc))
    return merged


ALL_REALS = Interval([(float('-inf'), float('inf'), False, False)])
NO_SOLUTION = Interval([])


def _format_num(x):
    """Format a number for interval display."""
    x = _to_int(x)
    if isinstance(x, float):
        if math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)
    return str(x)


# ---------------------------------------------------------------------------
# Inequality tokenizer and parser
# ---------------------------------------------------------------------------

def _find_inequality_op(expression):
    """Find the inequality operator(s) in an expression.
    Returns list of (position, operator, length) tuples.
    """
    ops = []
    i = 0
    depth = 0
    while i < len(expression):
        c = expression[i]
        if c == '(':
            depth += 1
            i += 1
        elif c == ')':
            depth -= 1
            i += 1
        elif depth == 0:
            if i + 1 < len(expression) and expression[i:i+2] in ('<=', '>='):
                ops.append((i, expression[i:i+2], 2))
                i += 2
            elif c in ('<', '>'):
                ops.append((i, c, 1))
                i += 1
            else:
                i += 1
        else:
            i += 1
    return ops


def _flip_op(op):
    """Flip an inequality operator."""
    return {'<': '>', '>': '<', '<=': '>=', '>=': '<='}[op]


def _op_is_strict(op):
    return op in ('<', '>')


def _parse_poly_from_expr(expr_str):
    """Parse an expression string into a Polynomial using Calculator13."""
    calc = Calculator13(expr_str)
    result = calc.Parse()
    var = calc._var_name or 'x'
    if isinstance(result, Polynomial):
        return result
    if isinstance(result, MultiPolynomial):
        if result.is_constant():
            return Polynomial([result.constant_value()], var=var)
        raise Exception("Multi-variable expressions not supported in inequalities")
    if isinstance(result, (int, float, complex)):
        return Polynomial([result], var=var)
    return Polynomial([result], var=var)


# ---------------------------------------------------------------------------
# Inequality solving
# ---------------------------------------------------------------------------

def _solve_linear_inequality(poly, op):
    """Solve a*x + b <op> 0 where poly = b + a*x."""
    a = poly.coeffs[1]
    b = poly.coeffs[0]
    # a*x + b <op> 0  =>  x <op'> -b/a  (flip if a < 0)
    if a == 0:
        # Degenerate: just check b <op> 0
        val = _to_int(b)
        if isinstance(val, complex):
            raise Exception("Cannot solve inequality with complex coefficients")
        if op == '<':
            return ALL_REALS if val < 0 else NO_SOLUTION
        elif op == '<=':
            return ALL_REALS if val <= 0 else NO_SOLUTION
        elif op == '>':
            return ALL_REALS if val > 0 else NO_SOLUTION
        elif op == '>=':
            return ALL_REALS if val >= 0 else NO_SOLUTION

    if isinstance(a, complex) or isinstance(b, complex):
        raise Exception("Cannot solve inequality with complex coefficients")

    boundary = -b / a
    boundary = _to_int(boundary)
    if isinstance(boundary, float) and math.isfinite(boundary) and boundary == int(boundary):
        boundary = int(boundary)

    actual_op = op
    if a < 0:
        actual_op = _flip_op(op)

    inclusive = actual_op in ('<=', '>=')
    if actual_op in ('<', '<='):
        return Interval([(float('-inf'), boundary, False, inclusive)])
    else:
        return Interval([(boundary, float('inf'), inclusive, False)])


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


def _check_op(val, op):
    """Check if val satisfies the inequality operator against 0."""
    val = float(val) if not isinstance(val, (int, float)) else val
    if op == '<':
        return val < 0
    elif op == '<=':
        return val <= 0
    elif op == '>':
        return val > 0
    elif op == '>=':
        return val >= 0
    return False


def _solve_polynomial_inequality(poly, op):
    """Solve poly <op> 0."""
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


def _solve_abs_inequality(inner_expr, op, rhs_expr):
    """Solve |inner| <op> rhs.

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
        # |expr| < a  =>  -a < expr AND expr < a
        inclusive = (op == '<=')
        op_lower = '>=' if inclusive else '>'
        op_upper = '<=' if inclusive else '<'

        # expr > -a  =>  expr - (-a) > 0  =>  expr + a > 0
        poly_lower = inner_poly + a
        interval_lower = _solve_polynomial_inequality(poly_lower, op_lower)

        # expr < a  =>  expr - a < 0
        poly_upper = inner_poly - a
        interval_upper = _solve_polynomial_inequality(poly_upper, op_upper)

        return interval_lower.intersect(interval_upper)

    elif op in ('>', '>='):
        # |expr| > a  =>  expr < -a OR expr > a
        inclusive = (op == '>=')
        op_lower = '<=' if inclusive else '<'
        op_upper = '>=' if inclusive else '>'

        # expr < -a  =>  expr + a < 0
        poly_lower = inner_poly + a
        interval_lower = _solve_polynomial_inequality(poly_lower, op_lower)

        # expr > a  =>  expr - a > 0
        poly_upper = inner_poly - a
        interval_upper = _solve_polynomial_inequality(poly_upper, op_upper)

        return interval_lower.union(interval_upper)

    raise Exception(f"Unsupported operator: {op}")


def _detect_abs_inequality(expression):
    """Detect pattern: abs(...) <op> expr.
    Returns (inner, op, rhs) or None.
    """
    stripped = expression.strip()
    if not stripped.startswith('abs('):
        return None

    # Find matching paren for abs(
    depth = 0
    i = 3  # start at '('
    while i < len(stripped):
        if stripped[i] == '(':
            depth += 1
        elif stripped[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1

    if i >= len(stripped):
        return None

    inner = stripped[4:i]
    rest = stripped[i+1:].strip()

    # rest should start with an inequality operator
    if rest.startswith('<='):
        return (inner, '<=', rest[2:].strip())
    elif rest.startswith('>='):
        return (inner, '>=', rest[2:].strip())
    elif rest.startswith('<'):
        return (inner, '<', rest[1:].strip())
    elif rest.startswith('>'):
        return (inner, '>', rest[1:].strip())

    return None


def solve_inequality(expression, symbolic=False):
    """Solve an inequality expression. Returns interval notation string."""
    expression = expression.strip()

    # Check for absolute value inequality
    abs_match = _detect_abs_inequality(expression)
    if abs_match:
        inner, op, rhs = abs_match
        result = _solve_abs_inequality(inner, op, rhs)
        return str(result)

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
        # Rewrite as: mid - left > 0 (or appropriate) AND right - mid > 0
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


# ---------------------------------------------------------------------------
# Conic section classifier
# ---------------------------------------------------------------------------

def _parse_conic_coefficients(expr_str):
    """Parse Ax^2 + Bxy + Cy^2 + Dx + Ey + F from expression string.

    Returns dict with keys 'A', 'C', 'D', 'E', 'F' (we don't handle Bxy term).
    Supports patterns like: x^2/9, 4*x^2, -y^2, 2*x, -3, etc.
    """
    A = 0  # coefficient of x^2
    C = 0  # coefficient of y^2
    D = 0  # coefficient of x
    E = 0  # coefficient of y
    F = 0  # constant

    tokens = _tokenize_conic(expr_str)
    i = 0
    sign = 1

    while i < len(tokens):
        tok = tokens[i]

        if tok == '+':
            sign = 1
            i += 1
            continue
        elif tok == '-':
            sign = -1
            i += 1
            continue

        # Parse one term: [number] [*] [var] [^2] [/number]
        coeff = None
        var = None
        power = 1

        # Leading number?
        if _is_number_str(tok):
            coeff = _parse_number(tok)
            i += 1
            # Check for * before variable
            if i < len(tokens) and tokens[i] == '*':
                i += 1
            # Check for / (number / number)
            if i < len(tokens) and tokens[i] == '/':
                i += 1
                if i < len(tokens) and _is_number_str(tokens[i]):
                    coeff = coeff / _parse_number(tokens[i])
                    i += 1
                    if i < len(tokens) and tokens[i] == '*':
                        i += 1
                else:
                    raise Exception("Expected number after /")

        # Variable?
        if i < len(tokens) and _is_var_token(tokens[i]):
            var = tokens[i]
            i += 1
            # ^2?
            if i < len(tokens) and tokens[i] == '^':
                i += 1
                if i < len(tokens) and tokens[i] == '2':
                    power = 2
                    i += 1
                else:
                    raise Exception("Only degree 2 supported in conic expressions")
            # /number after variable term? (e.g., x^2/9)
            if i < len(tokens) and tokens[i] == '/':
                i += 1
                if i < len(tokens) and _is_number_str(tokens[i]):
                    divisor = _parse_number(tokens[i])
                    i += 1
                    if coeff is None:
                        coeff = 1.0 / divisor
                    else:
                        coeff = coeff / divisor
                else:
                    raise Exception("Expected number after /")

        # Finalize coefficient
        if coeff is None:
            coeff = 1
        coeff = sign * coeff

        # Assign to the right bucket
        if var is None:
            F += coeff
        elif var == 'x' and power == 2:
            A += coeff
        elif var == 'y' and power == 2:
            C += coeff
        elif var == 'x' and power == 1:
            D += coeff
        elif var == 'y' and power == 1:
            E += coeff
        else:
            raise Exception(f"Unsupported term in conic expression")

        sign = 1  # reset for next term

    return {'A': _to_int(A), 'C': _to_int(C), 'D': _to_int(D), 'E': _to_int(E), 'F': _to_int(F)}


def _tokenize_conic(expr):
    """Tokenize a conic expression into a list of strings."""
    tokens = []
    i = 0
    expr = expr.strip()
    while i < len(expr):
        c = expr[i]
        if c == ' ':
            i += 1
            continue
        if c.isdigit() or c == '.':
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif c.isalpha():
            tokens.append(c)
            i += 1
        elif c in ('+', '-', '*', '/', '^', '(', ')'):
            tokens.append(c)
            i += 1
        else:
            i += 1
    return tokens


def _is_number_str(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _parse_number(s):
    if '.' in s:
        return float(s)
    return int(s)


def _is_var_token(s):
    return len(s) == 1 and s.isalpha() and s in ('x', 'y')


def _format_frac(num, denom=1):
    """Format a fraction nicely."""
    val = num / denom if denom != 1 else num
    val = _to_int(val)
    if isinstance(val, float) and math.isfinite(val) and val == int(val):
        return str(int(val))
    if isinstance(val, float):
        # Try to express as a nice fraction
        return str(val)
    return str(val)


def classify_conic(expr_str):
    """Classify a conic section and convert to standard form.

    Input: expression string like 'x^2+y^2-25' (set equal to 0).
    """
    coeffs = _parse_conic_coefficients(expr_str)
    A = coeffs['A']
    C = coeffs['C']
    D = coeffs['D']
    E = coeffs['E']
    F = coeffs['F']

    if A == 0 and C == 0:
        raise Exception("Not a conic section (no squared terms)")

    # Classification
    if A == C and A != 0:
        return _classify_circle(A, C, D, E, F)
    elif A != 0 and C != 0:
        if (A > 0 and C > 0) or (A < 0 and C < 0):
            return _classify_ellipse(A, C, D, E, F)
        else:
            return _classify_hyperbola(A, C, D, E, F)
    else:
        return _classify_parabola(A, C, D, E, F)


def _classify_circle(A, C, D, E, F):
    """Complete the square for circle: A(x^2 + D/A x) + A(y^2 + E/A y) + F = 0."""
    # Divide through by A
    d = D / A
    e = E / A
    f = F / A

    # x^2 + dx + y^2 + ey + f = 0
    # (x + d/2)^2 - (d/2)^2 + (y + e/2)^2 - (e/2)^2 + f = 0
    h = -d / 2
    k = -e / 2
    r_sq = (d/2)**2 + (e/2)**2 - f

    h = _to_int(h)
    k = _to_int(k)
    r_sq = _to_int(r_sq)

    if isinstance(r_sq, (int, float)) and r_sq < 0:
        return "Circle: no real solution (r^2 < 0)"
    if isinstance(r_sq, (int, float)) and r_sq == 0:
        return f"Circle: degenerate (point ({_format_num(h)},{_format_num(k)}))"

    r = math.sqrt(r_sq)
    r = _to_int(r)
    if isinstance(r, float) and math.isfinite(r) and abs(r - round(r)) < 1e-9:
        r = int(round(r))

    h_str = _format_num(h)
    k_str = _format_num(k)
    r_str = _format_num(r)

    # Build standard form string
    x_part = _std_form_part('x', h)
    y_part = _std_form_part('y', k)
    r_sq_str = _format_num(_to_int(r_sq))

    return f"Circle: {x_part}+{y_part}={r_sq_str}, center=({h_str},{k_str}), radius={r_str}"


def _classify_ellipse(A, C, D, E, F):
    """Complete the square for ellipse."""
    # A*x^2 + D*x + C*y^2 + E*y + F = 0
    # A(x^2 + D/A x) + C(y^2 + E/C y) + F = 0
    dx = D / A
    ey = E / C

    h = -dx / 2
    k = -ey / 2

    # A(x-h)^2 + C(y-k)^2 = A*(dx/2)^2 + C*(ey/2)^2 - F
    rhs = A * (dx/2)**2 + C * (ey/2)**2 - F

    if isinstance(rhs, (int, float)) and abs(rhs) < 1e-12:
        return f"Ellipse: degenerate (point ({_format_num(_to_int(h))},{_format_num(_to_int(k))}))"

    # (x-h)^2 / (rhs/A) + (y-k)^2 / (rhs/C) = 1
    a_sq = rhs / A
    b_sq = rhs / C

    a_sq = _to_int(a_sq)
    b_sq = _to_int(b_sq)
    h = _to_int(h)
    k = _to_int(k)

    if (isinstance(a_sq, (int, float)) and a_sq < 0) or (isinstance(b_sq, (int, float)) and b_sq < 0):
        return "Ellipse: no real solution"

    a = math.sqrt(a_sq)
    b = math.sqrt(b_sq)
    a = _to_int(a)
    b = _to_int(b)
    if isinstance(a, float) and math.isfinite(a) and abs(a - round(a)) < 1e-9:
        a = int(round(a))
    if isinstance(b, float) and math.isfinite(b) and abs(b - round(b)) < 1e-9:
        b = int(round(b))

    h_str = _format_num(h)
    k_str = _format_num(k)
    a_str = _format_num(a)
    b_str = _format_num(b)

    x_part = _std_form_part('x', h)
    y_part = _std_form_part('y', k)
    a_sq_str = _format_num(a_sq)
    b_sq_str = _format_num(b_sq)

    return f"Ellipse: {x_part}/{a_sq_str}+{y_part}/{b_sq_str}=1, center=({h_str},{k_str}), a={a_str}, b={b_str}"


def _classify_hyperbola(A, C, D, E, F):
    """Complete the square for hyperbola."""
    dx = D / A
    ey = E / C

    h = -dx / 2
    k = -ey / 2

    rhs = A * (dx/2)**2 + C * (ey/2)**2 - F

    if isinstance(rhs, (int, float)) and abs(rhs) < 1e-12:
        return "Hyperbola: degenerate"

    # Which squared term is positive?
    a_sq = rhs / A
    b_sq = rhs / C

    a_sq = _to_int(a_sq)
    b_sq = _to_int(b_sq)
    h = _to_int(h)
    k = _to_int(k)

    h_str = _format_num(h)
    k_str = _format_num(k)

    x_part = _std_form_part('x', h)
    y_part = _std_form_part('y', k)

    # a_sq should be positive for the positive term, b_sq negative (or vice versa)
    if isinstance(a_sq, (int, float)) and a_sq > 0:
        # (x-h)^2/a_sq - (y-k)^2/|b_sq| = 1
        neg_b_sq = -b_sq
        neg_b_sq = _to_int(neg_b_sq)
        a_val = math.sqrt(a_sq)
        b_val = math.sqrt(neg_b_sq)
        a_val = _to_int(a_val)
        b_val = _to_int(b_val)
        if isinstance(a_val, float) and math.isfinite(a_val) and abs(a_val - round(a_val)) < 1e-9:
            a_val = int(round(a_val))
        if isinstance(b_val, float) and math.isfinite(b_val) and abs(b_val - round(b_val)) < 1e-9:
            b_val = int(round(b_val))
        a_sq_str = _format_num(a_sq)
        b_sq_str = _format_num(neg_b_sq)
        return f"Hyperbola: {x_part}/{a_sq_str}-{y_part}/{b_sq_str}=1, center=({h_str},{k_str}), a={_format_num(a_val)}, b={_format_num(b_val)}"
    else:
        # a_sq < 0, b_sq > 0
        neg_a_sq = -a_sq
        neg_a_sq = _to_int(neg_a_sq)
        a_val = math.sqrt(b_sq)
        b_val = math.sqrt(neg_a_sq)
        a_val = _to_int(a_val)
        b_val = _to_int(b_val)
        if isinstance(a_val, float) and math.isfinite(a_val) and abs(a_val - round(a_val)) < 1e-9:
            a_val = int(round(a_val))
        if isinstance(b_val, float) and math.isfinite(b_val) and abs(b_val - round(b_val)) < 1e-9:
            b_val = int(round(b_val))
        a_sq_str = _format_num(b_sq)
        b_sq_str = _format_num(neg_a_sq)
        return f"Hyperbola: {y_part}/{a_sq_str}-{x_part}/{b_sq_str}=1, center=({h_str},{k_str}), a={_format_num(a_val)}, b={_format_num(b_val)}"


def _classify_parabola(A, C, D, E, F):
    """Complete the square for parabola. One of A, C is zero."""
    if A != 0 and C == 0:
        # A*x^2 + D*x + E*y + F = 0
        # A(x + D/(2A))^2 = -E*y - F + D^2/(4A)
        if E == 0:
            return "Parabola: degenerate (no y term)"
        h = -D / (2 * A)
        rhs_const = D**2 / (4 * A) - F
        # A(x-h)^2 = -E(y - rhs_const/E)  =>  but actually:
        # A(x-h)^2 = -E*y + rhs_const = -E(y - rhs_const/E)
        k = rhs_const / (-E) if E != 0 else 0
        # Negate: (x-h)^2 = (-E/A)(y - k)
        # Standard form: (x-h)^2 = 4p(y-k) where 4p = -E/A
        four_p = -E / A
        p = four_p / 4

        h = _to_int(h)
        k = _to_int(k)
        p = _to_int(p)
        if isinstance(h, float) and math.isfinite(h) and abs(h - round(h)) < 1e-9:
            h = int(round(h))
        if isinstance(k, float) and math.isfinite(k) and abs(k - round(k)) < 1e-9:
            k = int(round(k))
        if isinstance(p, float) and math.isfinite(p) and abs(p - round(p)) < 1e-9:
            p = int(round(p))

        h_str = _format_num(h)
        k_str = _format_num(k)
        p_str = _format_num(p)

        x_part = _std_form_part('x', h)
        four_p_str = _format_num(_to_int(four_p))

        direction = "up" if p > 0 else "down"
        return f"Parabola: {x_part}={four_p_str}*(y-{k_str}), vertex=({h_str},{k_str}), p={p_str}, opens {direction}"

    elif C != 0 and A == 0:
        # C*y^2 + E*y + D*x + F = 0
        if D == 0:
            return "Parabola: degenerate (no x term)"
        k = -E / (2 * C)
        rhs_const = E**2 / (4 * C) - F
        h = rhs_const / (-D) if D != 0 else 0
        four_p = -D / C
        p = four_p / 4

        h = _to_int(h)
        k = _to_int(k)
        p = _to_int(p)
        if isinstance(h, float) and math.isfinite(h) and abs(h - round(h)) < 1e-9:
            h = int(round(h))
        if isinstance(k, float) and math.isfinite(k) and abs(k - round(k)) < 1e-9:
            k = int(round(k))
        if isinstance(p, float) and math.isfinite(p) and abs(p - round(p)) < 1e-9:
            p = int(round(p))

        h_str = _format_num(h)
        k_str = _format_num(k)
        p_str = _format_num(p)

        y_part = _std_form_part('y', k)
        four_p_str = _format_num(_to_int(four_p))

        direction = "right" if p > 0 else "left"
        return f"Parabola: {y_part}={four_p_str}*(x-{h_str}), vertex=({h_str},{k_str}), p={p_str}, opens {direction}"

    raise Exception("Cannot classify parabola")


def _std_form_part(var, center):
    """Build (var-center)^2 string."""
    center = _to_int(center)
    if center == 0:
        return f"{var}^2"
    elif isinstance(center, (int, float)) and center > 0:
        return f"({var}-{_format_num(center)})^2"
    else:
        return f"({var}+{_format_num(-center)})^2"


# ---------------------------------------------------------------------------
# Calculator15 class
# ---------------------------------------------------------------------------

class Calculator15(Calculator14):
    """Inequalities and conic sections."""

    def __init__(self, expression, symbolic=False):
        # Add comparison operator tokens to the tokenizer
        self.symbolic = symbolic
        self.exp = []
        self.idx = 0
        self._var_names = set()
        self._var_name = None
        self._multi_mode = False
        i = 0
        while i < len(expression):
            c = expression[i]
            if (c >= '0' and c <= '9') or c == '.':
                j = i
                while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                    j += 1
                if j < len(expression) and expression[j] == '.':
                    j += 1
                    while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                        j += 1
                # Scientific notation
                if j < len(expression) and expression[j] in ('e', 'E'):
                    k = j + 1
                    if k < len(expression) and expression[k] in ('+', '-'):
                        k += 1
                    if k < len(expression) and expression[k] >= '0' and expression[k] <= '9':
                        j = k
                        while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                            j += 1
                num_str = expression[i:j]
                # Degree suffix: number followed by 'd'
                if j < len(expression) and expression[j] == 'd' and (j + 1 >= len(expression) or not expression[j + 1].isalpha()):
                    val = float(num_str) * math.pi / 180
                    self.exp.append(str(val))
                    j += 1
                else:
                    self.exp.append(num_str)
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('<', '>'):
                # Check for two-char operators <= >=
                if i + 1 < len(expression) and expression[i+1] == '=':
                    self.exp.append(c + '=')
                    i += 2
                else:
                    self.exp.append(c)
                    i += 1
            elif c in ('+', '-', '*', '/', '^', '(', ')', ',', '[', ']', '!', '%'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calc15(expression, symbolic=False):
    """Inequalities and conic sections."""
    expression = expression.strip()

    # Check for conic() function call
    if expression.startswith("conic(") and expression.endswith(")"):
        inner = expression[6:-1]
        return classify_conic(inner)

    # Check for inequality operators (at top level, not inside parens)
    ops = _find_inequality_op(expression)
    if ops:
        return solve_inequality(expression, symbolic)

    # Delegate everything else to calc14 (which handles factor, divpoly, complsq,
    # binom, equations, systems with ;, and falls through to expressions)
    return calc14(expression, symbolic=symbolic)
