import math

from calc.registry import register
from calc.helpers import _to_int, _format_solution
from calc.algebra import Polynomial, Calculator12, _unwrap_poly
from calc.inequalities import Calculator13


class MultiPolynomial:
    def __init__(self, terms=None):
        if terms is None:
            terms = {'': 0}
        self.terms = {}
        for k, v in terms.items():
            v = _to_int(v)
            if v != 0 or k == '':
                self.terms[k] = v
        if '' not in self.terms:
            self.terms[''] = 0

    def is_constant(self):
        return all(v == 0 for k, v in self.terms.items() if k != '')

    def constant_value(self):
        if not self.is_constant():
            raise Exception("MultiPolynomial is not a constant")
        return self.terms.get('', 0)

    def variables(self):
        return sorted(k for k, v in self.terms.items() if k != '' and v != 0)

    def get_coeff(self, var):
        return self.terms.get(var, 0)

    def _clean(self):
        to_remove = [k for k, v in self.terms.items() if k != '' and v == 0]
        for k in to_remove:
            del self.terms[k]

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = _multi_from_const(other)
        if isinstance(other, Polynomial):
            if other.is_constant():
                other = _multi_from_const(other.constant_value())
            else:
                raise Exception("Cannot add non-constant Polynomial to MultiPolynomial")
        if not isinstance(other, MultiPolynomial):
            return NotImplemented
        result = dict(self.terms)
        for k, v in other.terms.items():
            result[k] = result.get(k, 0) + v
        mp = MultiPolynomial(result)
        mp._clean()
        return mp

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return MultiPolynomial({k: -v for k, v in self.terms.items()})

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = _multi_from_const(other)
        if isinstance(other, Polynomial):
            if other.is_constant():
                other = _multi_from_const(other.constant_value())
            else:
                raise Exception("Cannot subtract non-constant Polynomial from MultiPolynomial")
        if not isinstance(other, MultiPolynomial):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return MultiPolynomial({k: v * other for k, v in self.terms.items()})
        if isinstance(other, Polynomial):
            if other.is_constant():
                return self * other.constant_value()
            else:
                raise Exception("Cannot multiply MultiPolynomial by non-constant Polynomial")
        if isinstance(other, MultiPolynomial):
            if other.is_constant():
                return self * other.constant_value()
            if self.is_constant():
                return other * self.constant_value()
            raise Exception("Cannot multiply two non-constant MultiPolynomials (nonlinear)")
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        if isinstance(other, Polynomial):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise Exception("Division by zero")
            return MultiPolynomial({k: v / other for k, v in self.terms.items()})
        if isinstance(other, Polynomial):
            if other.is_constant():
                return self / other.constant_value()
            raise Exception("Can only divide MultiPolynomial by a scalar")
        if isinstance(other, MultiPolynomial):
            if other.is_constant():
                return self / other.constant_value()
            raise Exception("Can only divide MultiPolynomial by a scalar")
        return NotImplemented

    def __rtruediv__(self, other):
        if not self.is_constant():
            raise Exception("Can only divide by a constant MultiPolynomial")
        val = self.constant_value()
        if val == 0:
            raise Exception("Division by zero")
        if isinstance(other, (int, float, complex)):
            return _multi_from_const(other / val)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MultiPolynomial):
            if not other.is_constant():
                raise Exception("Exponent must be a constant")
            other = other.constant_value()
        if isinstance(other, Polynomial):
            if not other.is_constant():
                raise Exception("Exponent must be a constant")
            other = other.constant_value()
        if isinstance(other, (int, float, complex)):
            if self.is_constant():
                return _multi_from_const(pow(self.constant_value(), other))
            # For non-constant, only support integer exponents
            if isinstance(other, complex) or other != int(other) or other < 0:
                raise Exception("MultiPolynomial exponent must be a non-negative integer")
            exp = int(other)
            if exp == 0:
                return _multi_from_const(1)
            result = self
            for _ in range(exp - 1):
                result = result * self
            return result
        return NotImplemented

    def __rpow__(self, other):
        if not self.is_constant():
            raise Exception("Exponent must be a constant MultiPolynomial")
        return _multi_from_const(pow(other, self.constant_value()))

    def __str__(self):
        if self.is_constant():
            return _format_solution(self.terms.get('', 0))

        parts = []
        var_keys = sorted(k for k in self.terms if k != '' and self.terms[k] != 0)

        for var in var_keys:
            c = _to_int(self.terms[var])
            if c == 0:
                continue
            if c == 1:
                term = var
            elif c == -1:
                term = '-' + var
            else:
                term = _format_solution(c) + '*' + var

            if not parts:
                parts.append(term)
            else:
                if isinstance(c, (int, float)) and not isinstance(c, complex) and c > 0:
                    if c == 1:
                        parts.append('+' + var)
                    else:
                        parts.append('+' + _format_solution(c) + '*' + var)
                else:
                    if isinstance(c, (int, float)) and not isinstance(c, complex) and c < 0:
                        parts.append(term)
                    else:
                        parts.append('+' + term)

        const = _to_int(self.terms.get('', 0))
        if const != 0:
            const_str = _format_solution(const)
            if not parts:
                parts.append(const_str)
            else:
                if isinstance(const, (int, float)) and not isinstance(const, complex) and const > 0:
                    parts.append('+' + const_str)
                else:
                    parts.append(const_str)

        if not parts:
            return '0'
        return ''.join(parts)


def _multi_from_var(var):
    return MultiPolynomial({'': 0, var: 1})

def _multi_from_const(value):
    return MultiPolynomial({'': _to_int(value)})


def solve_linear_system(equations, variables):
    n = len(variables)
    m = len(equations)

    if m < n:
        raise Exception("Underdetermined system: not enough equations")

    # Build augmented matrix
    matrix = []
    for eq in equations:
        row = []
        for var in variables:
            row.append(float(eq.get_coeff(var)))
        row.append(-float(eq.terms.get('', 0)))
        matrix.append(row)

    # Gaussian elimination with partial pivoting
    for col in range(n):
        max_val = abs(matrix[col][col])
        max_row = col
        for row in range(col + 1, m):
            if abs(matrix[row][col]) > max_val:
                max_val = abs(matrix[row][col])
                max_row = row

        if max_val < 1e-12:
            raise Exception("System has no unique solution (singular matrix)")

        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]

        for row in range(col + 1, m):
            factor = matrix[row][col] / matrix[col][col]
            for j in range(col, n + 1):
                matrix[row][j] -= factor * matrix[col][j]

    # Back substitution
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(matrix[i][i]) < 1e-12:
            raise Exception("System has no unique solution (singular matrix)")
        s = matrix[i][n]
        for j in range(i + 1, n):
            s -= matrix[i][j] * solution[j]
        solution[i] = s / matrix[i][i]

    result = {}
    for i, var in enumerate(variables):
        val = solution[i]
        if abs(val - round(val)) < 1e-9:
            val = int(round(val))
        else:
            val = _to_int(val)
        result[var] = val

    return result


class Calculator14(Calculator13):
    """Multi-variable linear equation systems."""
    _var_names = None
    _multi_mode = False

    def __init__(self, expression):
        self._var_names = set()
        self._multi_mode = False
        super().__init__(expression)

    def _promote_to_multi(self, value):
        if isinstance(value, MultiPolynomial):
            return value
        if isinstance(value, Polynomial):
            if value.is_constant():
                return _multi_from_const(value.constant_value())
            if value.degree() > 1:
                raise Exception("Multi-variable mode only supports linear expressions per variable")
            terms = {'': _to_int(value.coeffs[0])}
            if len(value.coeffs) > 1:
                terms[value.var] = _to_int(value.coeffs[1])
            return MultiPolynomial(terms)
        if isinstance(value, (int, float, complex)):
            return _multi_from_const(value)
        return value

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
                    # Unwrap Polynomial/MultiPolynomial constants
                    unwrapped = []
                    for a in args:
                        if isinstance(a, MultiPolynomial) and a.is_constant():
                            unwrapped.append(a.constant_value())
                        elif isinstance(a, Polynomial) and a.is_constant():
                            unwrapped.append(a.constant_value())
                        else:
                            unwrapped.append(a)
                    result = self.MULTI_FUNCTIONS[name](unwrapped)
                    if self._multi_mode:
                        return _multi_from_const(result)
                    return Polynomial([result], var=self._var_name or 'x')
                arg = args[0]
                if isinstance(arg, MultiPolynomial):
                    if arg.is_constant():
                        val = arg.constant_value()
                        return _multi_from_const(self.FUNCTIONS[name](val))
                    else:
                        raise Exception(f"Cannot apply function '{name}' to expression with variables")
                elif isinstance(arg, Polynomial):
                    if arg.is_constant():
                        val = arg.constant_value()
                        if self._multi_mode:
                            return _multi_from_const(self.FUNCTIONS[name](val))
                        result = self.FUNCTIONS[name](val)
                        return Polynomial([result], var=arg.var)
                    else:
                        raise Exception(f"Cannot apply function '{name}' to polynomial with variable")
                else:
                    result = self.FUNCTIONS[name](arg)
                    if self._multi_mode:
                        return _multi_from_const(result)
                    return result
            elif self._is_variable(name):
                self._var_names.add(name)
                if len(self._var_names) > 1 and not self._multi_mode:
                    self._multi_mode = True
                if self._multi_mode:
                    return _multi_from_var(name)
                else:
                    if self._var_name is None:
                        self._var_name = name
                    elif self._var_name != name:
                        pass
                    return Polynomial([0, 1], var=name)
            elif name in self.CONSTANTS:
                if self._multi_mode:
                    return _multi_from_const(self.CONSTANTS[name])
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
                if self._multi_mode:
                    return _multi_from_const(val)
                return Polynomial([val])
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")

    def Product(self):
        result = self.Power()
        next = self.PeekNextToken()
        while next == "*" or next == "/" or next == "%":
            self.PopNextToken()
            right = self.Power()
            if self._multi_mode:
                result = self._promote_to_multi(result)
                right = self._promote_to_multi(right)
            if next == "*":
                result = result * right
            elif next == "/":
                result = result / right
            elif next == "%":
                a = result.constant_value() if isinstance(result, (Polynomial, MultiPolynomial)) and result.is_constant() else result
                b = right.constant_value() if isinstance(right, (Polynomial, MultiPolynomial)) and right.is_constant() else right
                mod_result = int(a) % int(b)
                if self._multi_mode:
                    result = _multi_from_const(mod_result)
                else:
                    result = Polynomial([mod_result], var=self._var_name or 'x')
            next = self.PeekNextToken()
        return result

    def Sum(self):
        result = self.Product()
        next = self.PeekNextToken()
        while next == "+" or next == "-":
            self.PopNextToken()
            right = self.Product()
            if self._multi_mode:
                result = self._promote_to_multi(result)
                right = self._promote_to_multi(right)
            if next == "+":
                result = result + right
            elif next == "-":
                result = result - right
            next = self.PeekNextToken()
        return result

    def Postfix(self):
        result = self.Value()
        while self.PeekNextToken() == "!":
            self.PopNextToken()
            if isinstance(result, (Polynomial, MultiPolynomial)):
                if result.is_constant():
                    val = math.factorial(int(result.constant_value()))
                    if self._multi_mode:
                        return _multi_from_const(val)
                    return Polynomial([val], var=self._var_name or 'x')
            result = math.factorial(int(result))
        return result

    def Power(self):
        result = self.Postfix()
        next = self.PeekNextToken()
        if next == "^":
            self.PopNextToken()
            exponent = self.Power()
            if self._multi_mode:
                result = self._promote_to_multi(result)
                exponent = self._promote_to_multi(exponent)
                result = result ** exponent
            else:
                result = result ** exponent
        return result


def _promote_to_multi_poly(value):
    # Convert Polynomial or scalar to MultiPolynomial
    if isinstance(value, MultiPolynomial):
        return value
    if isinstance(value, Polynomial):
        if value.is_constant():
            return _multi_from_const(value.constant_value())
        else:
            terms = {'': _to_int(value.coeffs[0])}
            if len(value.coeffs) > 1:
                terms[value.var] = _to_int(value.coeffs[1])
            return MultiPolynomial(terms)
    if isinstance(value, (int, float, complex)):
        return _multi_from_const(value)
    return value


@register("calc14", description="Multi-variable linear equation systems",
          short_desc="Linear Systems", group="solver",
          examples=["x+y=2; x-y=0", "x+y+z=6; x-y=0; x+z=4"],
          i18n={"zh": "\u7ebf\u6027\u65b9\u7a0b\u7ec4", "hi": "\u0930\u0948\u0916\u093f\u0915 \u0938\u092e\u0940\u0915\u0930\u0923 \u0928\u093f\u0915\u093e\u092f", "es": "Sistemas Lineales", "fr": "Syst\u00e8mes Lin\u00e9aires", "ar": "\u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062e\u0637\u064a\u0629", "pt": "Sistemas Lineares", "ru": "\u041b\u0438\u043d\u0435\u0439\u043d\u044b\u0435 \u0441\u0438\u0441\u0442\u0435\u043c\u044b", "ja": "\u9023\u7acb\u4e00\u6b21\u65b9\u7a0b\u5f0f", "de": "Lineare Systeme"})
def calc14(expression):
    '''
    Extends calc13 to support multi-variable linear algebra.
    Multiple equations separated by ';' are solved as a system.
    eg. x+y=2; x-y=0 -> x=1; y=1
    Supports up to 3 variables.
    '''
    # Check for inequality operators first (before '=' check, since <= and >= contain '=')
    from calc.inequalities import _find_inequality_op
    ineq_ops = _find_inequality_op(expression)
    if ineq_ops:
        from calc.inequalities import calc13
        return calc13(expression)

    if ';' in expression:
        equation_strs = [s.strip() for s in expression.split(';')]
        all_equations = []
        all_vars = set()
        for eq_str in equation_strs:
            if '=' not in eq_str:
                raise Exception(f"Expected equation with '=' in system, got: {eq_str}")
            sides = eq_str.split('=')
            if len(sides) != 2:
                raise Exception("Only one '=' sign allowed per equation")
            left_str, right_str = sides

            calc_left = Calculator14(left_str)
            left = calc_left.Parse()
            calc_right = Calculator14(right_str)
            right = calc_right.Parse()

            for c in [calc_left, calc_right]:
                all_vars.update(c._var_names)

            left = _promote_to_multi_poly(left)
            right = _promote_to_multi_poly(right)

            eq = left - right
            all_equations.append(eq)

        variables = sorted(all_vars)
        solution = solve_linear_system(all_equations, variables)
        parts = []
        for var in variables:
            parts.append(f"{var}={_format_solution(solution[var])}")
        return '; '.join(parts)

    elif '=' in expression:
        sides = expression.split('=')
        if len(sides) != 2:
            raise Exception("Only one '=' sign allowed")
        left_expr, right_expr = sides

        calc_left = Calculator14(left_expr)
        left = calc_left.Parse()
        var_left = calc_left._var_name
        vars_left = calc_left._var_names

        calc_right = Calculator14(right_expr)
        right = calc_right.Parse()
        var_right = calc_right._var_name
        vars_right = calc_right._var_names

        all_vars = vars_left | vars_right

        if len(all_vars) <= 1:
            var = var_left or var_right or 'x'
            if not isinstance(left, Polynomial):
                if isinstance(left, MultiPolynomial):
                    if left.is_constant():
                        left = Polynomial([left.constant_value()], var=var)
                    else:
                        v = left.variables()[0]
                        left = Polynomial([left.terms.get('', 0), left.get_coeff(v)], var=v)
                else:
                    left = Polynomial([left], var=var)
            if not isinstance(right, Polynomial):
                if isinstance(right, MultiPolynomial):
                    if right.is_constant():
                        right = Polynomial([right.constant_value()], var=var)
                    else:
                        v = right.variables()[0]
                        right = Polynomial([right.terms.get('', 0), right.get_coeff(v)], var=v)
                else:
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
            left = _promote_to_multi_poly(left)
            right = _promote_to_multi_poly(right)

            eq = left - right
            variables = sorted(all_vars)
            solution = solve_linear_system([eq], variables)
            parts = []
            for var in variables:
                parts.append(f"{var}={_format_solution(solution[var])}")
            return '; '.join(parts)
    else:
        # No ';' or '=': check for inequality operators, otherwise parse with
        # Calculator14 for multi-variable simplification support
        from calc.inequalities import calc13, _find_inequality_op
        ops = _find_inequality_op(expression)
        if ops:
            return calc13(expression)
        calculator = Calculator14(expression)
        result = calculator.Parse()
        if isinstance(result, MultiPolynomial):
            if result.is_constant():
                return _format_solution(result.constant_value())
            return str(result)
        if isinstance(result, Polynomial):
            if result.is_constant():
                val = result.constant_value()
                return _format_solution(val)
            return str(result)
        return _format_solution(result)
