import math
from fractions import Fraction

from calc.registry import register
from calc.algebra import Polynomial
from calc.polytools import poly_divide, _parse_poly_expr, _split_args
from calc.polyineq import Calculator16, calc16


# ---------------------------------------------------------------------------
# Polynomial GCD via Euclidean algorithm
# ---------------------------------------------------------------------------

def poly_gcd(a, b):
    """Compute the GCD of two polynomials using the Euclidean algorithm.

    Uses poly_divide for polynomial long division.  Returns a monic
    polynomial (leading coefficient = 1).
    """
    # Ensure both are Polynomial objects
    if not isinstance(a, Polynomial):
        a = Polynomial([a])
    if not isinstance(b, Polynomial):
        b = Polynomial([b])

    # Use the same variable
    var = a.var if a.degree() > 0 else b.var

    while b.degree() > 0 or (b.degree() == 0 and b.coeffs[0] != 0):
        _, remainder = poly_divide(a, b)
        a = b
        b = remainder

    # Make monic (leading coefficient = 1)
    if a.degree() >= 0 and a.coeffs[-1] != 0:
        lead = a.coeffs[-1]
        new_coeffs = []
        for c in a.coeffs:
            if isinstance(c, int) and isinstance(lead, int):
                new_coeffs.append(Fraction(c, lead))
            else:
                new_coeffs.append(c / lead)
        a = Polynomial(new_coeffs, var=var)

    a.var = var
    return a


# ---------------------------------------------------------------------------
# RationalExpression class
# ---------------------------------------------------------------------------

class RationalExpression:
    """Represents a rational expression: numerator / denominator,
    where both are Polynomial objects."""

    def __init__(self, numerator, denominator=None):
        if denominator is None:
            denominator = Polynomial([1])
        if not isinstance(numerator, Polynomial):
            numerator = Polynomial([numerator])
        if not isinstance(denominator, Polynomial):
            denominator = Polynomial([denominator])

        # Ensure matching variable names
        if numerator.degree() > 0:
            var = numerator.var
        elif denominator.degree() > 0:
            var = denominator.var
        else:
            var = numerator.var
        numerator.var = var
        denominator.var = var

        # Check for zero denominator
        if denominator.degree() == 0 and denominator.coeffs[0] == 0:
            raise Exception("Division by zero polynomial")

        self.numerator = numerator
        self.denominator = denominator

    def simplify(self):
        """Simplify by dividing numerator and denominator by their GCD."""
        gcd = poly_gcd(self.numerator, self.denominator)

        if gcd.degree() == 0 and gcd.coeffs[0] == 0:
            return RationalExpression(self.numerator, self.denominator)

        # Divide both by polynomial GCD
        if gcd.degree() > 0:
            new_num, _ = poly_divide(self.numerator, gcd)
            new_den, _ = poly_divide(self.denominator, gcd)
        else:
            new_num = self.numerator
            new_den = self.denominator

        # Also simplify by scalar GCD of all integer coefficients
        result = RationalExpression(new_num, new_den)
        result = result._simplify_scalar()
        return result._normalize_sign()

    def _simplify_scalar(self):
        """Factor out common integer/fraction factor from all coefficients
        of both numerator and denominator."""
        all_coeffs = list(self.numerator.coeffs) + list(self.denominator.coeffs)

        # Convert all to Fraction for uniform handling
        fracs = []
        for c in all_coeffs:
            if isinstance(c, Fraction):
                fracs.append(c)
            elif isinstance(c, (int, float)):
                fracs.append(Fraction(c).limit_denominator(10**12))
            else:
                return self  # Complex or other type, skip scalar simplification

        # Filter out zeros
        nonzero = [abs(f) for f in fracs if f != 0]
        if not nonzero:
            return self

        # Find GCD of all numerators and LCM of all denominators
        from math import gcd as _gcd
        nums = [f.numerator for f in nonzero]
        dens = [f.denominator for f in nonzero]

        g_num = abs(nums[0])
        for n in nums[1:]:
            g_num = _gcd(g_num, abs(n))

        g_den = dens[0]
        for d in dens[1:]:
            g_den = g_den * d // _gcd(g_den, d)

        scalar_gcd = Fraction(g_num, g_den)
        if scalar_gcd <= 0 or scalar_gcd == 1:
            return self

        var = self.numerator.var
        new_num_coeffs = []
        for c in self.numerator.coeffs:
            f = Fraction(c).limit_denominator(10**12) if not isinstance(c, Fraction) else c
            new_num_coeffs.append(f / scalar_gcd)
        new_den_coeffs = []
        for c in self.denominator.coeffs:
            f = Fraction(c).limit_denominator(10**12) if not isinstance(c, Fraction) else c
            new_den_coeffs.append(f / scalar_gcd)

        return RationalExpression(
            Polynomial(new_num_coeffs, var=var),
            Polynomial(new_den_coeffs, var=var)
        )

    def _normalize_sign(self):
        """Ensure the leading coefficient of the denominator is positive."""
        den_lead = self.denominator.coeffs[-1]
        if isinstance(den_lead, Fraction):
            den_lead_neg = den_lead < 0
        else:
            den_lead_neg = den_lead < 0

        if den_lead_neg:
            new_num = Polynomial([-c for c in self.numerator.coeffs],
                                 var=self.numerator.var)
            new_den = Polynomial([-c for c in self.denominator.coeffs],
                                 var=self.denominator.var)
            return RationalExpression(new_num, new_den)
        return RationalExpression(self.numerator, self.denominator)

    def __add__(self, other):
        """Add two rational expressions: a/b + c/d = (a*d + c*b) / (b*d)."""
        if not isinstance(other, RationalExpression):
            other = RationalExpression(other)
        new_num = self.numerator * other.denominator + other.numerator * self.denominator
        new_den = self.denominator * other.denominator
        return RationalExpression(new_num, new_den).simplify()

    def __sub__(self, other):
        """Subtract two rational expressions: a/b - c/d = (a*d - c*b) / (b*d)."""
        if not isinstance(other, RationalExpression):
            other = RationalExpression(other)
        new_num = self.numerator * other.denominator - other.numerator * self.denominator
        new_den = self.denominator * other.denominator
        return RationalExpression(new_num, new_den).simplify()

    def __mul__(self, other):
        """Multiply two rational expressions: (a/b) * (c/d) = (a*c) / (b*d)."""
        if not isinstance(other, RationalExpression):
            other = RationalExpression(other)
        new_num = self.numerator * other.numerator
        new_den = self.denominator * other.denominator
        return RationalExpression(new_num, new_den).simplify()

    def __truediv__(self, other):
        """Divide two rational expressions: (a/b) / (c/d) = (a*d) / (b*c)."""
        if not isinstance(other, RationalExpression):
            other = RationalExpression(other)
        if other.numerator.degree() == 0 and other.numerator.coeffs[0] == 0:
            raise Exception("Division by zero")
        new_num = self.numerator * other.denominator
        new_den = self.denominator * other.numerator
        return RationalExpression(new_num, new_den).simplify()

    def __eq__(self, other):
        if not isinstance(other, RationalExpression):
            return NotImplemented
        # Compare simplified forms
        a = self.simplify()
        b = other.simplify()
        return str(a) == str(b)

    def __str__(self):
        # Simplify for display
        num = self.numerator
        den = self.denominator

        # If denominator is 1, just show numerator
        if den.degree() == 0:
            den_val = den.coeffs[0]
            if isinstance(den_val, Fraction):
                is_one = (den_val == Fraction(1))
            else:
                is_one = (den_val == 1)
            if is_one:
                return str(num)

        num_str = str(num)
        den_str = str(den)

        # Wrap in parentheses if polynomial has multiple terms
        if num.degree() > 0 or (isinstance(num.coeffs[0], (int, float)) and num.coeffs[0] < 0):
            if '+' in num_str or (num_str.count('-') > (1 if num_str.startswith('-') else 0)):
                num_str = f"({num_str})"
            elif num.degree() > 0:
                num_str = f"({num_str})"
        if den.degree() > 0:
            den_str = f"({den_str})"

        return f"{num_str}/{den_str}"

    def __repr__(self):
        return f"RationalExpression({self.numerator}, {self.denominator})"


# ---------------------------------------------------------------------------
# Calculator17 class - Rational Expressions
# ---------------------------------------------------------------------------

class Calculator17(Calculator16):
    """Rational expressions: simplify, add, subtract, multiply, divide
    polynomial fractions.

    Inherits all features from Calculator16 (polynomial inequalities).
    """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@register("calc17",
          description="Rational expressions: simplify, add, subtract, multiply, divide polynomial fractions",
          short_desc="Rational Expressions",
          group="solver",
          examples=["simplify(x^2-1,x+1)", "radd(1,x+1,1,x-1)", "rmul(x,x+1,x+1,x-1)"],
          i18n={"zh": "\u6709\u7406\u8868\u8fbe\u5f0f",
                "hi": "\u092a\u0930\u093f\u092e\u0947\u092f \u0935\u094d\u092f\u0902\u091c\u0915",
                "es": "Expresiones Racionales",
                "fr": "Expressions Rationnelles",
                "ar": "\u0627\u0644\u062a\u0639\u0628\u064a\u0631\u0627\u062a \u0627\u0644\u0646\u0633\u0628\u064a\u0629",
                "pt": "Express\u00f5es Racionais",
                "ru": "\u0420\u0430\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u0438\u044f",
                "ja": "\u6709\u7406\u5f0f",
                "de": "Rationale Ausdr\u00fccke"})
def calc17(expression):
    """Rational expressions: simplify, add, subtract, multiply, divide
    polynomial fractions."""

    expr = expression.strip()

    # --- simplify(num_expr, den_expr) ---
    if expr.startswith("simplify(") and expr.endswith(")"):
        inner = expr[9:-1]
        parts = _split_args(inner)
        if len(parts) != 2:
            raise Exception("simplify requires exactly 2 arguments: numerator, denominator")
        num_poly = _parse_poly_expr(parts[0].strip())
        den_poly = _parse_poly_expr(parts[1].strip())
        rat = RationalExpression(num_poly, den_poly).simplify()
        return str(rat)

    # --- radd(num1, den1, num2, den2) ---
    if expr.startswith("radd(") and expr.endswith(")"):
        inner = expr[5:-1]
        parts = _split_args(inner)
        if len(parts) != 4:
            raise Exception("radd requires exactly 4 arguments: num1, den1, num2, den2")
        n1 = _parse_poly_expr(parts[0].strip())
        d1 = _parse_poly_expr(parts[1].strip())
        n2 = _parse_poly_expr(parts[2].strip())
        d2 = _parse_poly_expr(parts[3].strip())
        r1 = RationalExpression(n1, d1)
        r2 = RationalExpression(n2, d2)
        result = r1 + r2
        return str(result)

    # --- rsub(num1, den1, num2, den2) ---
    if expr.startswith("rsub(") and expr.endswith(")"):
        inner = expr[5:-1]
        parts = _split_args(inner)
        if len(parts) != 4:
            raise Exception("rsub requires exactly 4 arguments: num1, den1, num2, den2")
        n1 = _parse_poly_expr(parts[0].strip())
        d1 = _parse_poly_expr(parts[1].strip())
        n2 = _parse_poly_expr(parts[2].strip())
        d2 = _parse_poly_expr(parts[3].strip())
        r1 = RationalExpression(n1, d1)
        r2 = RationalExpression(n2, d2)
        result = r1 - r2
        return str(result)

    # --- rmul(num1, den1, num2, den2) ---
    if expr.startswith("rmul(") and expr.endswith(")"):
        inner = expr[5:-1]
        parts = _split_args(inner)
        if len(parts) != 4:
            raise Exception("rmul requires exactly 4 arguments: num1, den1, num2, den2")
        n1 = _parse_poly_expr(parts[0].strip())
        d1 = _parse_poly_expr(parts[1].strip())
        n2 = _parse_poly_expr(parts[2].strip())
        d2 = _parse_poly_expr(parts[3].strip())
        r1 = RationalExpression(n1, d1)
        r2 = RationalExpression(n2, d2)
        result = r1 * r2
        return str(result)

    # --- rdiv(num1, den1, num2, den2) ---
    if expr.startswith("rdiv(") and expr.endswith(")"):
        inner = expr[5:-1]
        parts = _split_args(inner)
        if len(parts) != 4:
            raise Exception("rdiv requires exactly 4 arguments: num1, den1, num2, den2")
        n1 = _parse_poly_expr(parts[0].strip())
        d1 = _parse_poly_expr(parts[1].strip())
        n2 = _parse_poly_expr(parts[2].strip())
        d2 = _parse_poly_expr(parts[3].strip())
        r1 = RationalExpression(n1, d1)
        r2 = RationalExpression(n2, d2)
        result = r1 / r2
        return str(result)

    # Fall through to calc16
    return calc16(expression)
