import math
import cmath

from calc.registry import register
from calc.core import format_complex
from calc.helpers import _to_int, _fmt_num, _cbrt, _clean_root, _unique_roots, _sort_roots, _format_solution
from calc.numtheory import Calculator11


def _unwrap_poly(val):
    """Unwrap a Polynomial constant to a plain number."""
    if isinstance(val, Polynomial) and val.is_constant():
        return val.constant_value()
    return val


class Polynomial:
    def __init__(self, coeffs=None, var='x'):
        if coeffs is None:
            coeffs = [0]
        self.coeffs = [_to_int(c) for c in coeffs]
        self.var = var
        self._trim()

    def _trim(self):
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()

    def degree(self):
        if len(self.coeffs) == 1 and self.coeffs[0] == 0:
            return 0
        return len(self.coeffs) - 1

    def is_constant(self):
        return all(c == 0 for c in self.coeffs[1:])

    def constant_value(self):
        if not self.is_constant():
            raise Exception("Polynomial is not a constant")
        return self.coeffs[0]

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = Polynomial([other], var=self.var)
        if not isinstance(other, Polynomial):
            return NotImplemented
        var = self.var if self.degree() > 0 else other.var
        n = max(len(self.coeffs), len(other.coeffs))
        coeffs = [0] * n
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            coeffs[i] = a + b
        return Polynomial(coeffs, var=var)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return Polynomial([-c for c in self.coeffs], var=self.var)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = Polynomial([other], var=self.var)
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            other = Polynomial([other], var=self.var)
        if not isinstance(other, Polynomial):
            return NotImplemented
        var = self.var if self.degree() > 0 else other.var
        n = len(self.coeffs) + len(other.coeffs) - 1
        coeffs = [0] * n
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                coeffs[i + j] += a * b
        return Polynomial(coeffs, var=var)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            if not other.is_constant():
                raise Exception("Can only divide polynomial by a scalar")
            other = other.constant_value()
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise Exception("Division by zero")
            if isinstance(other, int):
                from fractions import Fraction as _Frac
                new_coeffs = []
                for c in self.coeffs:
                    if isinstance(c, (int, _Frac)):
                        new_coeffs.append(_Frac(c, other))
                    else:
                        new_coeffs.append(c / other)
                return Polynomial(new_coeffs, var=self.var)
            return Polynomial([c / other for c in self.coeffs], var=self.var)
        return NotImplemented

    def __rtruediv__(self, other):
        if not self.is_constant():
            raise Exception("Can only divide by a constant polynomial")
        val = self.constant_value()
        if val == 0:
            raise Exception("Division by zero")
        if isinstance(other, (int, float, complex)):
            from fractions import Fraction as _Frac
            if isinstance(other, (int, _Frac)) and isinstance(val, (int, _Frac)):
                return Polynomial([_Frac(other) / _Frac(val)], var=self.var)
            return Polynomial([other / val], var=self.var)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, Polynomial):
            if not other.is_constant():
                raise Exception("Exponent must be a constant")
            other = other.constant_value()
        if isinstance(other, (int, float, complex)):
            if self.is_constant():
                return Polynomial([pow(self.constant_value(), other)], var=self.var)
            if isinstance(other, complex) or other != int(other) or other < 0:
                raise Exception("Polynomial exponent must be a non-negative integer")
            exp = int(other)
            result = Polynomial([1], var=self.var)
            for _ in range(exp):
                result = result * self
            return result
        return NotImplemented

    def __rpow__(self, other):
        if not self.is_constant():
            raise Exception("Exponent must be a constant polynomial")
        return Polynomial([pow(other, self.constant_value())], var=self.var)

    def __str__(self):
        if self.is_constant():
            return _fmt_num(self.coeffs[0])

        parts = []
        for i in range(len(self.coeffs) - 1, -1, -1):
            c = self.coeffs[i]
            if c == 0:
                continue
            if i == 0:
                term = _fmt_num(c)
                if not parts:
                    parts.append(term)
                else:
                    if isinstance(c, complex):
                        parts.append('+' + term)
                    elif c > 0:
                        parts.append('+' + term)
                    else:
                        parts.append(term)
            else:
                var_part = self.var if i == 1 else f"{self.var}^{i}"
                if c == 1:
                    coeff_str = ''
                elif c == -1:
                    coeff_str = '-'
                else:
                    coeff_str = _fmt_num(c) + '*'

                term = coeff_str + var_part

                if not parts:
                    parts.append(term)
                else:
                    if isinstance(c, (int, float)) and not isinstance(c, complex):
                        if c > 0:
                            if c == 1:
                                parts.append('+' + var_part)
                            else:
                                parts.append('+' + term)
                        else:
                            parts.append(term)
                    else:
                        parts.append('+' + term)

        if not parts:
            return '0'
        return ''.join(parts)

    def solve(self):
        self._trim()
        deg = self.degree()

        if deg == 0:
            if self.coeffs[0] == 0:
                raise Exception("Infinite solutions")
            else:
                return []  # No solution (nonzero constant = 0)

        if deg == 1:
            sol = -self.coeffs[0] / self.coeffs[1]
            return [_to_int(sol)]

        if deg == 2:
            a = self.coeffs[2]
            b = self.coeffs[1]
            c = self.coeffs[0]
            disc = b * b - 4 * a * c
            disc = _to_int(disc)

            if isinstance(disc, (int, float)) and not isinstance(disc, complex) and disc >= 0:
                sqrt_disc = math.sqrt(disc)
                x1 = (-b - sqrt_disc) / (2 * a)
                x2 = (-b + sqrt_disc) / (2 * a)
                x1 = _to_int(x1)
                x2 = _to_int(x2)
                if x1 == x2:
                    return [x1]
                solutions = sorted([x1, x2], key=lambda v: (v.real if isinstance(v, complex) else v))
                return solutions
            else:
                sqrt_disc = cmath.sqrt(disc)
                x1 = (-b - sqrt_disc) / (2 * a)
                x2 = (-b + sqrt_disc) / (2 * a)
                x1 = _to_int(x1)
                x2 = _to_int(x2)
                solutions = sorted([x1, x2], key=lambda v: (v.real, v.imag) if isinstance(v, complex) else (v, 0))
                return solutions

        if deg == 3:
            a = self.coeffs[3]
            b = self.coeffs[2]
            c = self.coeffs[1]
            d = self.coeffs[0]

            shift = b / (3 * a)
            p = (3 * a * c - b * b) / (3 * a * a)
            q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)

            disc = -(4 * p * p * p + 27 * q * q)

            eps = 1e-9

            if abs(p) < eps and abs(q) < eps:
                root = -shift
                root = _clean_root(root, eps)
                return [root]

            if abs(disc) < eps:
                if abs(q) < eps:
                    root = -shift
                    root = _clean_root(root, eps)
                    return [root]
                else:
                    root1 = 3 * q / p - shift
                    root2 = -3 * q / (2 * p) - shift
                    root1 = _clean_root(root1, eps)
                    root2 = _clean_root(root2, eps)
                    roots = sorted(set([root1, root2]), key=lambda v: v.real if isinstance(v, complex) else v)
                    return roots

            inner = complex(q * q / 4 + p * p * p / 27)
            sqrt_inner = cmath.sqrt(inner)

            u_base = -q / 2 + sqrt_inner
            v_base = -q / 2 - sqrt_inner

            u = _cbrt(u_base)
            v = _cbrt(v_base)

            omega = complex(-0.5, math.sqrt(3) / 2)
            omega2 = complex(-0.5, -math.sqrt(3) / 2)

            t1 = u + v
            t2 = u * omega + v * omega2
            t3 = u * omega2 + v * omega

            raw_roots = [t1 - shift, t2 - shift, t3 - shift]

            solutions = []
            for r in raw_roots:
                r = _clean_root(r, eps)
                solutions.append(r)

            solutions = _unique_roots(solutions, eps)
            solutions = _sort_roots(solutions)

            return solutions

        raise Exception(f"Cannot solve degree {deg} polynomial")


class Calculator12(Calculator11):
    _var_name = None

    def _is_variable(self, name):
        if len(name) != 1:
            return False
        if name in self.CONSTANTS:
            return False
        if name in self.FUNCTIONS:
            return False
        if not name.isalpha():
            return False
        return True

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
            return result
        elif next is not None and next[0].isalpha():
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                _all_funcs = set(self.FUNCTIONS) | set(self.MULTI_FUNCTIONS)
                if name not in _all_funcs:
                    raise Exception(f"Unknown function '{name}'")
                self.PopNextToken()
                args = self._parse_function_args()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
                if name in self.MULTI_FUNCTIONS:
                    # Unwrap Polynomial constants for numeric multi-arg functions
                    unwrapped = []
                    for a in args:
                        if isinstance(a, Polynomial) and a.is_constant():
                            unwrapped.append(a.constant_value())
                        else:
                            unwrapped.append(a)
                    result = self.MULTI_FUNCTIONS[name](unwrapped)
                    return Polynomial([result], var=self._var_name or 'x')
                arg = args[0]
                if isinstance(arg, Polynomial):
                    if arg.is_constant():
                        val = arg.constant_value()
                        result = self.FUNCTIONS[name](val)
                        return Polynomial([result], var=arg.var)
                    else:
                        raise Exception(f"Cannot apply function '{name}' to polynomial with variable")
                else:
                    return self.FUNCTIONS[name](arg)
            elif self._is_variable(name):
                if self._var_name is None:
                    self._var_name = name
                elif self._var_name != name:
                    raise Exception(f"Multiple variables not supported: '{self._var_name}' and '{name}'")
                return Polynomial([0, 1], var=name)
            elif name in self.CONSTANTS:
                return Polynomial([self.CONSTANTS[name]])
            else:
                raise Exception(f"Unknown identifier '{name}'")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next or 'e' in next or 'E' in next:
                    val = float(next)
                else:
                    val = int(next)
                return Polynomial([val])
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")

    def Postfix(self):
        result = self.Value()
        while self.PeekNextToken() == "!":
            self.PopNextToken()
            if isinstance(result, Polynomial) and result.is_constant():
                result = Polynomial([math.factorial(int(result.constant_value()))], var=result.var)
            else:
                result = math.factorial(int(result))
        return result

    def Power(self):
        result = self.Postfix()
        next = self.PeekNextToken()
        if next == "^":
            self.PopNextToken()
            exponent = self.Power()
            if isinstance(result, Polynomial) and isinstance(exponent, Polynomial):
                result = result ** exponent
            elif isinstance(result, Polynomial):
                result = result ** exponent
            elif isinstance(exponent, Polynomial):
                result = result ** exponent
            else:
                result = pow(result, exponent)
        return result

    def Product(self):
        result = self.Power()
        next = self.PeekNextToken()
        while next == "*" or next == "/" or next == "%":
            self.PopNextToken()
            right = self.Power()
            if next == "*":
                result = result * right
            elif next == "/":
                result = result / right
            elif next == "%":
                a = result.constant_value() if isinstance(result, Polynomial) and result.is_constant() else result
                b = right.constant_value() if isinstance(right, Polynomial) and right.is_constant() else right
                mod_result = int(a) % int(b)
                result = Polynomial([mod_result], var=getattr(result, 'var', 'x') if isinstance(result, Polynomial) else 'x')
            next = self.PeekNextToken()
        return result

    def Sum(self):
        result = self.Product()
        next = self.PeekNextToken()
        while next == "+" or next == "-":
            self.PopNextToken()
            right = self.Product()
            if next == "+":
                result = result + right
            elif next == "-":
                result = result - right
            next = self.PeekNextToken()
        return result


@register("calc12", description="Algebra: simplify expressions and solve equations",
          short_desc="Algebra & Equations", group="solver",
          examples=["(x+1)*(x-1)", "(x+1)^2", "x^2-5*x+6=0", "x^3-6*x^2+11*x-6=0"],
          i18n={"zh": "\u4ee3\u6570\u4e0e\u65b9\u7a0b", "hi": "\u092c\u0940\u091c\u0917\u0923\u093f\u0924 \u0914\u0930 \u0938\u092e\u0940\u0915\u0930\u0923", "es": "\u00c1lgebra y Ecuaciones", "fr": "Alg\u00e8bre et \u00c9quations", "ar": "\u0627\u0644\u062c\u0628\u0631 \u0648\u0627\u0644\u0645\u0639\u0627\u062f\u0644\u0627\u062a", "pt": "\u00c1lgebra e Equa\u00e7\u00f5es", "ru": "\u0410\u043b\u0433\u0435\u0431\u0440\u0430 \u0438 \u0443\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u044f", "ja": "\u4ee3\u6570\u3068\u65b9\u7a0b\u5f0f", "de": "Algebra und Gleichungen"})
def calc12(expression):
    '''
    Extends calc11 to support single-variable algebra and equation solving.
    Without '=': simplify expression (eg. 2*x+3*x -> 5*x)
    With '=': solve equation (eg. x^2=1 -> x=-1; x=1)
    Supports linear, quadratic, and cubic equations.
    '''
    if '=' in expression:
        sides = expression.split('=')
        if len(sides) != 2:
            raise Exception("Only one '=' sign allowed")
        left_expr, right_expr = sides

        calc_left = Calculator12(left_expr)
        left = calc_left.Parse()
        var_left = calc_left._var_name

        calc_right = Calculator12(right_expr)
        right = calc_right.Parse()
        var_right = calc_right._var_name

        var = var_left or var_right or 'x'

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
        solutions = poly.solve()

        if not solutions:
            raise Exception("No solution")

        parts = []
        for sol in solutions:
            parts.append(f"{var}={_format_solution(sol)}")
        return '; '.join(parts)
    else:
        calculator = Calculator12(expression)
        result = calculator.Parse()
        if isinstance(result, Polynomial):
            if result.is_constant():
                val = result.constant_value()
                return _format_solution(val)
            return str(result)
        return _format_solution(result)
