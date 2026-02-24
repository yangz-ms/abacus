import math
import cmath
from fractions import Fraction

from calc.registry import register
from calc.helpers import _to_int, _clean_root, _unique_roots, _sort_roots, _format_solution
from calc.numtheory import _prime_factors, _format_factors
from calc.matrix import Matrix, _format_matrix_result
from calc.algebra import Polynomial
from calc.systems import Calculator13, calc13, MultiPolynomial


# ---------------------------------------------------------------------------
# Radical class: represents a + b*sqrt(n) with exact rational arithmetic
# ---------------------------------------------------------------------------

class Radical:
    """Represents a + b*sqrt(n) where a,b are Fraction and n is square-free integer.
    Used for exact symbolic arithmetic."""

    def __init__(self, rational=0, radical_coeff=0, radicand=0):
        """a + b*sqrt(n). If n=0 or b=0, this is just a rational number."""
        self.a = Fraction(rational)
        self.b = Fraction(radical_coeff)
        self.n = int(radicand)
        self._simplify()

    def _simplify(self):
        """Simplify: extract perfect square factors from n."""
        if self.b == 0 or self.n == 0:
            self.b = Fraction(0)
            self.n = 0
            return
        # Handle negative radicand by keeping sign
        sign = 1 if self.n > 0 else -1
        abs_n = abs(self.n)
        # Extract largest perfect square factor: sqrt(48) = 4*sqrt(3)
        factor = 1
        d = 2
        while d * d <= abs_n:
            while abs_n % (d * d) == 0:
                abs_n //= (d * d)
                factor *= d
            d += 1
        self.b *= factor
        self.n = abs_n * sign
        # sqrt(1) = 1, so fold into rational part
        if self.n == 1:
            self.a += self.b
            self.b = Fraction(0)
            self.n = 0

    def is_rational(self):
        return self.b == 0

    def __add__(self, other):
        if isinstance(other, (int, float, Fraction)):
            return Radical(self.a + Fraction(other), self.b, self.n)
        if not isinstance(other, Radical):
            return NotImplemented
        if other.is_rational():
            return Radical(self.a + other.a, self.b, self.n)
        if self.is_rational():
            return Radical(self.a + other.a, other.b, other.n)
        if self.n != other.n:
            raise ValueError("Cannot add radicals with different radicands")
        return Radical(self.a + other.a, self.b + other.b, self.n)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, Fraction)):
            return Radical(self.a - Fraction(other), self.b, self.n)
        if not isinstance(other, Radical):
            return NotImplemented
        if other.is_rational():
            return Radical(self.a - other.a, self.b, self.n)
        if self.is_rational():
            return Radical(self.a - other.a, -other.b, other.n)
        if self.n != other.n:
            raise ValueError("Cannot subtract radicals with different radicands")
        return Radical(self.a - other.a, self.b - other.b, self.n)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return Radical(-self.a, -self.b, self.n)

    def __mul__(self, other):
        if isinstance(other, (int, float, Fraction)):
            f = Fraction(other)
            return Radical(self.a * f, self.b * f, self.n)
        if not isinstance(other, Radical):
            return NotImplemented
        if other.is_rational():
            return self * other.a
        if self.is_rational():
            return other * self.a
        # (a1 + b1*sqrt(n)) * (a2 + b2*sqrt(n)) = (a1*a2 + b1*b2*n) + (a1*b2 + a2*b1)*sqrt(n)
        if self.n != other.n:
            raise ValueError("Cannot multiply radicals with different radicands")
        new_a = self.a * other.a + self.b * other.b * self.n
        new_b = self.a * other.b + self.b * other.a
        return Radical(new_a, new_b, self.n)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, Fraction)):
            f = Fraction(other)
            if f == 0:
                raise ZeroDivisionError("Division by zero")
            return Radical(self.a / f, self.b / f, self.n)
        if not isinstance(other, Radical):
            return NotImplemented
        if other.is_rational():
            return self / other.a
        # Rationalize denominator: multiply by conjugate
        # (a1+b1*sqrt(n)) / (a2+b2*sqrt(n)) = (a1+b1*sqrt(n))*(a2-b2*sqrt(n)) / (a2^2 - b2^2*n)
        if self.n != 0 and other.n != 0 and self.n != other.n:
            raise ValueError("Cannot divide radicals with different radicands")
        conjugate = Radical(other.a, -other.b, other.n)
        numer = self * conjugate
        denom = other.a * other.a - other.b * other.b * other.n
        if denom == 0:
            raise ZeroDivisionError("Division by zero (degenerate denominator)")
        return Radical(numer.a / denom, numer.b / denom, numer.n)

    def __eq__(self, other):
        if isinstance(other, (int, float, Fraction)):
            return self.is_rational() and self.a == Fraction(other)
        if isinstance(other, Radical):
            return self.a == other.a and self.b == other.b and self.n == other.n
        return NotImplemented

    def __float__(self):
        if self.is_rational():
            return float(self.a)
        return float(self.a) + float(self.b) * math.sqrt(float(self.n))

    def __str__(self):
        if self.is_rational():
            if self.a.denominator == 1:
                return str(self.a.numerator)
            return f"{self.a.numerator}/{self.a.denominator}"
        # Format b*sqrt(n)
        def _fmt_radical():
            if self.b == 1:
                return f"sqrt({self.n})"
            elif self.b == -1:
                return f"-sqrt({self.n})"
            elif self.b.denominator == 1:
                return f"{self.b.numerator}*sqrt({self.n})"
            else:
                return f"({self.b.numerator}/{self.b.denominator})*sqrt({self.n})"
        if self.a == 0:
            return _fmt_radical()
        # a + b*sqrt(n)
        a_str = str(self.a.numerator) if self.a.denominator == 1 else f"{self.a.numerator}/{self.a.denominator}"
        rad_str = _fmt_radical()
        if self.b > 0:
            return f"{a_str}+{rad_str}"
        return f"{a_str}{rad_str}"

    def __repr__(self):
        return f"Radical({self.a}, {self.b}, {self.n})"


# ---------------------------------------------------------------------------
# Polynomial factoring via Rational Root Theorem
# ---------------------------------------------------------------------------

def _divisors(n):
    """Return all positive divisors of integer n."""
    n = abs(n)
    if n == 0:
        return [1]
    divs = []
    for i in range(1, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs


def _eval_poly_fraction(coeffs, x):
    """Evaluate polynomial with Fraction coefficients at Fraction x using Horner's method."""
    result = Fraction(0)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _synthetic_div_fraction(coeffs, root):
    """Perform synthetic division of polynomial (coeffs low-to-high) by (x - root).
    root is a Fraction. Returns quotient coefficients (low-to-high)."""
    n = len(coeffs) - 1  # degree
    # Convert to high-to-low for synthetic division
    high_to_low = list(reversed(coeffs))
    result = [Fraction(0)] * n
    result[0] = Fraction(high_to_low[0])
    for i in range(1, n):
        result[i] = result[i - 1] * root + Fraction(high_to_low[i])
    # remainder = result[-1] * root + high_to_low[-1], should be 0
    # Convert back to low-to-high
    return list(reversed(result))


def _format_fraction(f):
    """Format a Fraction as a string."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _format_factor(root, var='x'):
    """Format (x - root) as a string, where root is a Fraction."""
    if root == 0:
        return var
    if root > 0:
        if root.denominator == 1:
            return f"({var}-{root.numerator})"
        return f"({var}-{root.numerator}/{root.denominator})"
    else:
        neg_root = -root
        if neg_root.denominator == 1:
            return f"({var}+{neg_root.numerator})"
        return f"({var}+{neg_root.numerator}/{neg_root.denominator})"


def _poly_to_str_fraction(coeffs, var='x'):
    """Convert Fraction coefficients (low-to-high) to a polynomial string."""
    p = Polynomial([float(c) if c.denominator != 1 else int(c.numerator) for c in coeffs], var=var)
    return str(p)


def factor_polynomial(poly):
    """Factor polynomial over rationals using the Rational Root Theorem.
    Returns a formatted string like '(x-2)*(x-3)' or '2*(x-1)*(x^2+x+1)'.
    poly: a Polynomial object."""
    if poly.degree() == 0:
        return str(poly)

    var = poly.var

    # Convert coefficients to Fraction for exact arithmetic
    frac_coeffs = [Fraction(c).limit_denominator(10**12) if isinstance(c, float)
                   else Fraction(c) for c in poly.coeffs]

    # Clear denominators to work with integer coefficients
    lcm_denom = 1
    for c in frac_coeffs:
        lcm_denom = lcm_denom * c.denominator // math.gcd(lcm_denom, c.denominator)
    int_coeffs = [c * lcm_denom for c in frac_coeffs]

    # Extract the leading coefficient
    leading = int_coeffs[-1]
    constant = int_coeffs[0]

    # Factor out GCD of all coefficients
    overall_gcd = abs(int(int_coeffs[0].numerator))
    for c in int_coeffs[1:]:
        overall_gcd = math.gcd(overall_gcd, abs(int(c.numerator)))
    if overall_gcd == 0:
        overall_gcd = 1

    working_coeffs = [c / overall_gcd for c in int_coeffs]
    scalar_factor = Fraction(overall_gcd, lcm_denom)

    # Make leading coefficient positive
    if working_coeffs[-1] < 0:
        working_coeffs = [-c for c in working_coeffs]
        scalar_factor = -scalar_factor

    # Find rational roots: test +-p/q where p | constant_term, q | leading_coeff
    roots_found = []
    current_coeffs = working_coeffs[:]

    while len(current_coeffs) > 1:
        constant_term = current_coeffs[0]
        leading_coeff = current_coeffs[-1]

        if constant_term == 0:
            # x=0 is a root
            roots_found.append(Fraction(0))
            # Divide out x (shift coefficients)
            current_coeffs = current_coeffs[1:]
            continue

        c_abs = abs(int(constant_term.numerator))
        l_abs = abs(int(leading_coeff.numerator))

        p_divs = _divisors(c_abs)
        q_divs = _divisors(l_abs)

        found = False
        for p in p_divs:
            for q in q_divs:
                for sign in [1, -1]:
                    candidate = Fraction(sign * p, q)
                    val = _eval_poly_fraction(current_coeffs, candidate)
                    if val == 0:
                        roots_found.append(candidate)
                        current_coeffs = _synthetic_div_fraction(current_coeffs, candidate)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break  # No more rational roots; remaining polynomial is irreducible over Q

    # Build the factored form
    factors = []

    # Scalar factor
    if scalar_factor != 1:
        if scalar_factor == -1:
            factors.append("-1")
        elif scalar_factor.denominator == 1:
            factors.append(str(scalar_factor.numerator))
        else:
            factors.append(f"({scalar_factor.numerator}/{scalar_factor.denominator})")

    # Linear factors from roots
    for root in roots_found:
        factors.append(_format_factor(root, var))

    # Remaining polynomial (irreducible quotient)
    if len(current_coeffs) > 1:
        remaining_str = _poly_to_str_fraction(current_coeffs, var)
        if len(roots_found) > 0:
            factors.append(f"({remaining_str})")
        else:
            # No roots found at all; polynomial is irreducible
            if scalar_factor == 1:
                return remaining_str
            factors.append(f"({remaining_str})")

    if not factors:
        return "0"

    return '*'.join(factors)


# ---------------------------------------------------------------------------
# Polynomial long division
# ---------------------------------------------------------------------------

def poly_divide(dividend, divisor):
    """Polynomial long division. Returns (quotient, remainder) as Polynomial objects.
    Both inputs are Polynomial objects."""
    if divisor.degree() == 0 and divisor.coeffs[0] == 0:
        raise Exception("Division by zero polynomial")

    var = dividend.var

    # Work with copies
    dividend_coeffs = list(dividend.coeffs)
    divisor_coeffs = list(divisor.coeffs)

    if dividend.degree() < divisor.degree():
        return Polynomial([0], var=var), Polynomial(dividend_coeffs, var=var)

    # Use Fraction for exact arithmetic
    d_coeffs = [Fraction(c).limit_denominator(10**12) if isinstance(c, float)
                else Fraction(c) for c in dividend_coeffs]
    s_coeffs = [Fraction(c).limit_denominator(10**12) if isinstance(c, float)
                else Fraction(c) for c in divisor_coeffs]

    quotient_len = len(d_coeffs) - len(s_coeffs) + 1
    q_coeffs = [Fraction(0)] * quotient_len

    remainder = list(d_coeffs)

    for i in range(quotient_len - 1, -1, -1):
        idx = i + len(s_coeffs) - 1
        coeff = remainder[idx] / s_coeffs[-1]
        q_coeffs[i] = coeff
        for j in range(len(s_coeffs)):
            remainder[i + j] -= coeff * s_coeffs[j]

    # Convert back to int/float
    def _frac_to_num(f):
        if f.denominator == 1:
            return int(f.numerator)
        return float(f)

    q_result = [_frac_to_num(c) for c in q_coeffs]
    r_result = [_frac_to_num(c) for c in remainder[:len(s_coeffs) - 1]]
    if not r_result:
        r_result = [0]

    return Polynomial(q_result, var=var), Polynomial(r_result, var=var)


# ---------------------------------------------------------------------------
# Completing the square
# ---------------------------------------------------------------------------

def complete_square(poly):
    """For ax^2+bx+c, return string like '(x+h)^2+k' or 'a*(x+h)^2+k'.
    poly: a Polynomial object of degree 2."""
    if poly.degree() != 2:
        raise Exception("Completing the square requires a degree-2 polynomial")

    var = poly.var
    a = Fraction(poly.coeffs[2]).limit_denominator(10**12) if isinstance(poly.coeffs[2], float) else Fraction(poly.coeffs[2])
    b = Fraction(poly.coeffs[1]).limit_denominator(10**12) if isinstance(poly.coeffs[1], float) else Fraction(poly.coeffs[1])
    c = Fraction(poly.coeffs[0]).limit_denominator(10**12) if isinstance(poly.coeffs[0], float) else Fraction(poly.coeffs[0])

    # a*x^2 + b*x + c = a*(x + b/(2a))^2 + (c - b^2/(4a))
    h = b / (2 * a)  # shift: (x + h)
    k = c - b * b / (4 * a)  # vertical offset

    def _fmt_frac(f):
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"

    # Build inner: (x+h) or (x-h)
    if h == 0:
        inner = f"{var}"
    elif h > 0:
        inner = f"{var}+{_fmt_frac(h)}"
    else:
        inner = f"{var}-{_fmt_frac(-h)}"

    squared = f"({inner})^2"

    # Build result with leading coefficient
    if a == 1:
        prefix = squared
    elif a == -1:
        prefix = f"-({inner})^2"
    else:
        prefix = f"{_fmt_frac(a)}*({inner})^2"

    # Add constant k
    if k == 0:
        return prefix
    elif k > 0:
        return f"{prefix}+{_fmt_frac(k)}"
    else:
        return f"{prefix}-{_fmt_frac(-k)}"


# ---------------------------------------------------------------------------
# Binomial expansion
# ---------------------------------------------------------------------------

def binom_expand(poly, n):
    """Expand (polynomial)^n using the existing Polynomial.__pow__."""
    if not isinstance(n, int) or n < 0:
        raise Exception("Binomial exponent must be a non-negative integer")
    return poly ** n


# ---------------------------------------------------------------------------
# Durand-Kerner method for higher-degree root finding
# ---------------------------------------------------------------------------

def durand_kerner(coeffs, max_iter=1000, tol=1e-12):
    """Find all roots of polynomial with given coefficients using Durand-Kerner method.
    coeffs: list of coefficients low-to-high [a_0, a_1, ..., a_n]
    Returns list of complex roots."""
    # Trim trailing zeros
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]

    n = len(coeffs) - 1  # degree
    if n <= 0:
        return []

    # Normalize so leading coefficient is 1 (monic)
    lead = complex(coeffs[-1])
    norm_coeffs = [complex(c) / lead for c in coeffs]

    def eval_poly(z):
        result = complex(0)
        zp = complex(1)
        for c in norm_coeffs:
            result += c * zp
            zp *= z
        return result

    # Initialize n distinct complex starting points on a circle
    # Use radius based on coefficients for better convergence
    radius = 1 + max(abs(complex(c)) for c in norm_coeffs[:-1])
    roots = []
    for k in range(n):
        angle = 2 * math.pi * k / n + 0.4  # offset to avoid symmetry issues
        roots.append(radius * cmath.exp(complex(0, angle)))

    for _ in range(max_iter):
        max_change = 0
        new_roots = list(roots)
        for i in range(n):
            # Compute product of (z_i - z_j) for j != i
            denom = complex(1)
            for j in range(n):
                if j != i:
                    diff = roots[i] - roots[j]
                    if abs(diff) < 1e-30:
                        diff = complex(1e-30, 1e-30)
                    denom *= diff
            delta = eval_poly(roots[i]) / denom
            new_roots[i] = roots[i] - delta
            max_change = max(max_change, abs(delta))
        roots = new_roots
        if max_change < tol:
            break

    return roots


# ---------------------------------------------------------------------------
# Parsing helpers for calc14 special functions
# ---------------------------------------------------------------------------

def _parse_poly_expr(expr_str):
    """Parse a polynomial expression string using Calculator13, returning a Polynomial."""
    calc = Calculator13(expr_str)
    result = calc.Parse()
    if isinstance(result, Polynomial):
        return result
    if isinstance(result, (int, float)):
        return Polynomial([result])
    raise Exception(f"Could not parse as polynomial: {expr_str}")


# ---------------------------------------------------------------------------
# Calculator14 class
# ---------------------------------------------------------------------------

class Calculator14(Calculator13):
    """Polynomial tools: factor, long division, completing the square,
    binomial expansion, higher-degree solving."""

    def __init__(self, expression, symbolic=False):
        self.symbolic = symbolic
        super().__init__(expression)


# ---------------------------------------------------------------------------
# calc14() entry function
# ---------------------------------------------------------------------------

@register("calc14", description="Polynomial factoring, long division, completing the square, binomial expansion, higher-degree equation solving",
          short_desc="Poly Tools", group="solver",
          examples=["factor(x^2-5*x+6)", "divpoly(x^3-1,x-1)", "complsq(x^2+6*x+5)", "binom(x+2,5)", "x^4-1=0"],
          i18n={"zh": "\u591a\u9879\u5f0f\u5de5\u5177", "hi": "\u092c\u0939\u0941\u092a\u0926 \u0909\u092a\u0915\u0930\u0923", "es": "Herr. Polinomios", "fr": "Outils Polyn\u00f4mes", "ar": "\u0623\u062f\u0648\u0627\u062a \u0643\u062b\u064a\u0631\u0627\u062a \u0627\u0644\u062d\u062f\u0648\u062f", "pt": "Ferr. Polin\u00f4mios", "ru": "\u041f\u043e\u043b\u0438\u043d\u043e\u043c\u044b", "ja": "\u591a\u9805\u5f0f\u30c4\u30fc\u30eb", "de": "Polynom-Werkzeuge"})
def calc14(expression, symbolic=False):
    """Polynomial tools: factor, long division, completing the square,
    binomial expansion, higher-degree equation solving."""

    expr = expression.strip()

    # Delegate semicolon systems to calc13
    if ';' in expression:
        return calc13(expression, symbolic=symbolic)

    # Handle special function calls: factor(...), divpoly(...), complsq(...), binom(...)
    if expr.startswith("factor(") and expr.endswith(")"):
        inner = expr[7:-1]
        poly = _parse_poly_expr(inner)
        # If the polynomial is a constant (degree 0), do numeric prime factoring
        if poly.is_constant():
            return _format_factors(_prime_factors(int(poly.constant_value())))
        return factor_polynomial(poly)

    if expr.startswith("divpoly(") and expr.endswith(")"):
        inner = expr[8:-1]
        # Split on comma, respecting parentheses
        parts = _split_args(inner)
        if len(parts) != 2:
            raise Exception("divpoly requires exactly 2 arguments: dividend, divisor")
        dividend = _parse_poly_expr(parts[0].strip())
        divisor = _parse_poly_expr(parts[1].strip())
        quotient, remainder = poly_divide(dividend, divisor)
        if remainder.degree() == 0 and remainder.coeffs[0] == 0:
            return str(quotient)
        return f"{quotient} R {remainder}"

    if expr.startswith("complsq(") and expr.endswith(")"):
        inner = expr[8:-1]
        poly = _parse_poly_expr(inner)
        return complete_square(poly)

    if expr.startswith("binom(") and expr.endswith(")"):
        inner = expr[6:-1]
        parts = _split_args(inner)
        if len(parts) != 2:
            raise Exception("binom requires exactly 2 arguments: expression, exponent")
        poly = _parse_poly_expr(parts[0].strip())
        n_val = int(parts[1].strip())
        result = binom_expand(poly, n_val)
        return str(result)

    # Handle equations with '='
    if '=' in expr:
        return _solve_equation(expr, symbolic)

    # Otherwise simplify
    calculator = Calculator14(expr, symbolic=symbolic)
    result = calculator.Parse()
    if isinstance(result, Matrix):
        return _format_matrix_result(result)
    if isinstance(result, list):
        return _format_matrix_result(result)
    if isinstance(result, Polynomial):
        if result.is_constant():
            return _format_solution(result.constant_value())
        return str(result)
    return _format_solution(result)


def _split_args(s):
    """Split a string by commas, respecting parentheses."""
    parts = []
    depth = 0
    current = []
    for c in s:
        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
    parts.append(''.join(current))
    return parts


def _solve_equation(expression, symbolic=False):
    """Solve an equation, using Durand-Kerner for degree > 3."""
    sides = expression.split('=')
    if len(sides) != 2:
        raise Exception("Only one '=' sign allowed")
    left_expr, right_expr = sides

    calc_left = Calculator14(left_expr, symbolic=symbolic)
    left = calc_left.Parse()

    calc_right = Calculator14(right_expr, symbolic=symbolic)
    right = calc_right.Parse()

    var = calc_left._var_name or calc_right._var_name or 'x'

    if not isinstance(left, Polynomial):
        left = Polynomial([left], var=var)
    if not isinstance(right, Polynomial):
        right = Polynomial([right], var=var)

    if left.is_constant() and not right.is_constant():
        left.var = right.var
    elif right.is_constant() and not left.is_constant():
        right.var = left.var

    poly = left - right
    poly.var = var

    deg = poly.degree()

    # For degree <= 3, use the existing exact solver
    if deg <= 3:
        solutions = poly.solve()
    else:
        # Use Durand-Kerner for higher degrees
        raw_roots = durand_kerner(poly.coeffs)
        solutions = []
        for r in raw_roots:
            r = _clean_root(r)
            solutions.append(r)
        solutions = _unique_roots(solutions)
        solutions = _sort_roots(solutions)

    if not solutions:
        raise Exception("No solution")

    parts = []
    for sol in solutions:
        parts.append(f"{var}={_format_solution(sol)}")
    return '; '.join(parts)
