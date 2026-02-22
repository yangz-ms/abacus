import math
import cmath

def calc(expression):
    result = 0
    op = '+'
    current = 0
    for c in expression+'\uffff':
        if c >= '0' and c <= '9':
            # accumulate current number
            current = current * 10 + int(c)
        elif c == '+' or c == '-' or c == '\uffff':
            # start next operator, calculate current result
            if op == '+':
                result += current
            elif op == '-':
                result -= current
            current = 0
            op = c
        elif c != ' ':
            raise Exception(f"Invalid character '{c}'")
    return str(result)

def calc2(expression):
    result = 0
    result2 = 0
    op = '+'
    current = 0
    for c in expression+'\uffff':
        if c >= '0' and c <= '9':
            # accumulate current number
            current = current * 10 + int(c)
        elif c == '+' or c == '-' or c == '*' or c == '/' or c == '\uffff':
            # start next operator, calculate current result
            if op == '+':
                result += result2
                result2 = current
            elif op == '-':
                result += result2
                result2 = -current
            elif op == '*':
                result2 *= current
            elif op == '/':
                result2 /= current
            current = 0
            op = c
        elif c != ' ':
            raise Exception(f"Invalid character '{c}'")

    return str(result+result2)

class Calculator3:
    exp = None
    idx = 0

    def __init__(self, expression):
        self.exp = []
        current = ""
        for c in expression:
            if c >= '0' and c <= '9':
                current += c
            else:
                if current != "":
                    self.exp.append(current)
                    current = ""
            if c == '+' or c == '-' or c == '*' or c == '/' or c == '^' or c == '(' or c == ')':
                self.exp.append(str(c))
        if current != "":
            self.exp.append(current)
        self.idx = 0

    def PeekNextToken(self):
        if self.idx >= len(self.exp):
            return None
        return self.exp[self.idx]

    def PopNextToken(self):
        if self.idx >= len(self.exp):
            return None
        result = self.exp[self.idx]
        self.idx += 1
        return result

    def Expr(self):
        return self.Sum()

    def Parse(self):
        result = self.Expr()
        if self.PeekNextToken() is not None:
            raise Exception(f"Unexpected token {self.PeekNextToken()}")
        return result
    
    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            if not next.isdigit():
                raise Exception(f"Unexpected token {next}")
            result = int(next)
        return result

    def Power(self):
        result = self.Value()
        next = self.PeekNextToken()
        if next == "^":
            next = self.PopNextToken()
            nextResult = self.Power()
            result = pow(result, nextResult)
        return result

    def Product(self):
        result = self.Power()
        next = self.PeekNextToken()
        while next == "*" or next == "/":
            next = self.PopNextToken()
            nextResult = self.Power()
            if next == "*":
                result *= nextResult
            elif next == "/":
                result /= nextResult
            next = self.PeekNextToken()
        return result

    def Sum(self):
        result = self.Product()
        next = self.PeekNextToken()
        while next == "+" or next == "-":
            next = self.PopNextToken()
            nextResult = self.Product()
            if next == "+":
                result += nextResult
            elif next == "-":
                result -= nextResult
            next = self.PeekNextToken()
        return result


def calc3(expression):
    '''
    Use the following grammar
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← [0-9]+ / '(' Expr ')'
    '''

    calculator = Calculator3(expression)
    return str(calculator.Parse())

class Calculator4(Calculator3):
    def __init__(self, expression):
        self.exp = []
        self.idx = 0
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
                if j < len(expression) and expression[j] in ('e', 'E'):
                    j += 1
                    if j < len(expression) and expression[j] in ('+', '-'):
                        j += 1
                    while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                        j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next or 'e' in next or 'E' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result


def calc4(expression):
    '''
    Extends calc3 to support scientific notation and decimals.
    Use the following grammar:
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← Number / '(' Expr ')'
    Number  ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    '''
    calculator = Calculator4(expression)
    return str(calculator.Parse())


class Calculator5(Calculator4):
    CONSTANTS = {
        'pi': math.pi,
        'e':  math.e,
    }

    def __init__(self, expression):
        self.exp = []
        self.idx = 0
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
                # only consume e/E as scientific notation if followed by digit or sign+digit
                if j < len(expression) and expression[j] in ('e', 'E'):
                    k = j + 1
                    if k < len(expression) and expression[k] in ('+', '-'):
                        k += 1
                    if k < len(expression) and expression[k] >= '0' and expression[k] <= '9':
                        j = k
                        while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                            j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            if next[0].isalpha():
                if next not in self.CONSTANTS:
                    raise Exception(f"Unknown constant '{next}'")
                result = self.CONSTANTS[next]
            else:
                try:
                    if '.' in next or 'e' in next or 'E' in next:
                        result = float(next)
                    else:
                        result = int(next)
                except (ValueError, TypeError):
                    raise Exception(f"Unexpected token {next}")
        return result


def calc5(expression):
    '''
    Extends calc4 to support named constants (pi, e).
    Use the following grammar:
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← Number / Constant / '(' Expr ')'
    Number  ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant ← 'pi' / 'e'
    '''
    calculator = Calculator5(expression)
    return str(calculator.Parse())


class Calculator6(Calculator5):
    CONSTANTS = {**Calculator5.CONSTANTS, 'i': complex(0, 1)}


def format_complex(value):
    def fmt_num(x):
        if isinstance(x, float) and math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)

    if not isinstance(value, complex):
        return fmt_num(value)

    real = round(value.real, 15)
    imag = round(value.imag, 15)

    if imag == 0:
        return fmt_num(real)
    if real == 0:
        if imag == 1:
            return 'i'
        elif imag == -1:
            return '-i'
        return f"{fmt_num(imag)}i"
    if imag == 1:
        return f"{fmt_num(real)}+i"
    elif imag == -1:
        return f"{fmt_num(real)}-i"
    elif imag > 0:
        return f"{fmt_num(real)}+{fmt_num(imag)}i"
    else:
        return f"{fmt_num(real)}{fmt_num(imag)}i"


def calc6(expression):
    '''
    Extends calc5 to support imaginary unit i and complex numbers.
    Use the following grammar:
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← Number / Constant / '(' Expr ')'
    Number  ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant ← 'pi' / 'e' / 'i'
    '''
    calculator = Calculator6(expression)
    return format_complex(calculator.Parse())


class Calculator7(Calculator6):
    FUNCTIONS = {
        'sin':   cmath.sin,
        'cos':   cmath.cos,
        'tan':   cmath.tan,
        'asin':  cmath.asin,
        'acos':  cmath.acos,
        'atan':  cmath.atan,
        'sinh':  cmath.sinh,
        'cosh':  cmath.cosh,
        'tanh':  cmath.tanh,
        'exp':   cmath.exp,
        'ln':    cmath.log,
        'log':   cmath.log10,
        'sqrt':  cmath.sqrt,
        'abs':   abs,
    }

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            self.PopNextToken()
            result = self.Expr()
            closing = self.PopNextToken()
            if closing != ")":
                raise Exception(f"Invalid token {closing}")
        elif next is not None and next[0].isalpha():
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                if name not in self.FUNCTIONS:
                    raise Exception(f"Unknown function '{name}'")
                self.PopNextToken()  # consume '('
                arg = self.Expr()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
                result = self.FUNCTIONS[name](arg)
            else:
                if name not in self.CONSTANTS:
                    raise Exception(f"Unknown constant '{name}'")
                result = self.CONSTANTS[name]
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next or 'e' in next or 'E' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result


def calc7(expression):
    '''
    Extends calc6 to support common math functions.
    Use the following grammar:
    Expr     ← Sum
    Sum      ← Product (('+' / '-') Product)*
    Product  ← Power (('*' / '/') Power)*
    Power    ← Value ('^' Power)?
    Value    ← Function '(' Expr ')' / Constant / Number / '(' Expr ')'
    Number   ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant ← 'pi' / 'e' / 'i'
    Function ← 'sin' / 'cos' / 'tan' / 'asin' / 'acos' / 'atan' /
               'sinh' / 'cosh' / 'tanh' / 'exp' / 'ln' / 'log' /
               'sqrt' / 'abs'
    '''
    calculator = Calculator7(expression)
    return format_complex(calculator.Parse())


class Polynomial:
    """Represents a polynomial: coeffs[i] is the coefficient of x^i."""

    def __init__(self, coeffs=None, var='x'):
        if coeffs is None:
            coeffs = [0]
        # Normalize coefficients: convert to int when possible
        self.coeffs = [self._to_int(c) for c in coeffs]
        self.var = var
        self._trim()

    @staticmethod
    def _to_int(value):
        """Convert float to int if it represents a whole number."""
        if isinstance(value, complex):
            if value.imag == 0:
                value = value.real
            else:
                return value
        if isinstance(value, float) and value == int(value) and math.isfinite(value):
            return int(value)
        return value

    def _trim(self):
        """Remove trailing zero coefficients, keep at least one."""
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()

    @property
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
        var = self.var if self.degree > 0 else other.var
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
        var = self.var if self.degree > 0 else other.var
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
            return Polynomial([c / other for c in self.coeffs], var=self.var)
        return NotImplemented

    def __rtruediv__(self, other):
        if not self.is_constant():
            raise Exception("Can only divide by a constant polynomial")
        val = self.constant_value()
        if val == 0:
            raise Exception("Division by zero")
        if isinstance(other, (int, float, complex)):
            return Polynomial([other / val], var=self.var)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, Polynomial):
            if not other.is_constant():
                raise Exception("Exponent must be a constant")
            other = other.constant_value()
        if isinstance(other, (int, float, complex)):
            # If base is constant, use numeric pow
            if self.is_constant():
                return Polynomial([pow(self.constant_value(), other)], var=self.var)
            # For polynomials, only integer exponents >= 0
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
        # If constant, format the number
        if self.is_constant():
            return self._fmt_num(self.coeffs[0])

        parts = []
        for i in range(len(self.coeffs) - 1, -1, -1):
            c = self.coeffs[i]
            if c == 0:
                continue
            if i == 0:
                # Constant term
                term = self._fmt_num(c)
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
                # Variable term
                var_part = self.var if i == 1 else f"{self.var}^{i}"
                if c == 1:
                    coeff_str = ''
                elif c == -1:
                    coeff_str = '-'
                else:
                    coeff_str = self._fmt_num(c) + '*'

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

    @staticmethod
    def _fmt_num(x):
        x = Polynomial._to_int(x)
        if isinstance(x, complex):
            return format_complex(x)
        if isinstance(x, float):
            if math.isfinite(x) and x == int(x):
                return str(int(x))
            return str(x)
        return str(x)

    def solve(self):
        """Solve polynomial = 0. Returns sorted list of solutions."""
        self._trim()
        deg = self.degree

        if deg == 0:
            if self.coeffs[0] == 0:
                raise Exception("Infinite solutions")
            else:
                return []  # No solution (nonzero constant = 0)

        if deg == 1:
            # a1*x + a0 = 0 => x = -a0/a1
            sol = -self.coeffs[0] / self.coeffs[1]
            return [self._to_int(sol)]

        if deg == 2:
            a = self.coeffs[2]
            b = self.coeffs[1]
            c = self.coeffs[0]
            disc = b * b - 4 * a * c
            disc = self._to_int(disc)

            if isinstance(disc, (int, float)) and not isinstance(disc, complex) and disc >= 0:
                sqrt_disc = math.sqrt(disc)
                x1 = (-b - sqrt_disc) / (2 * a)
                x2 = (-b + sqrt_disc) / (2 * a)
                x1 = self._to_int(x1)
                x2 = self._to_int(x2)
                if x1 == x2:
                    return [x1]
                solutions = sorted([x1, x2], key=lambda v: (v.real if isinstance(v, complex) else v))
                return solutions
            else:
                # Complex roots
                sqrt_disc = cmath.sqrt(disc)
                x1 = (-b - sqrt_disc) / (2 * a)
                x2 = (-b + sqrt_disc) / (2 * a)
                x1 = self._to_int(x1)
                x2 = self._to_int(x2)
                # Sort by real part, then imaginary part
                solutions = sorted([x1, x2], key=lambda v: (v.real, v.imag) if isinstance(v, complex) else (v, 0))
                return solutions

        raise Exception(f"Cannot solve degree {deg} polynomial")


class Calculator8(Calculator7):
    _var_name = None

    def _is_variable(self, name):
        """Check if a name is a single-letter variable (not a constant or function)."""
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
        next_tok = self.PeekNextToken()
        if next_tok == "(":
            self.PopNextToken()
            result = self.Expr()
            closing = self.PopNextToken()
            if closing != ")":
                raise Exception(f"Invalid token {closing}")
            return result
        elif next_tok is not None and next_tok[0].isalpha():
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                # Function call
                if name not in self.FUNCTIONS:
                    raise Exception(f"Unknown function '{name}'")
                self.PopNextToken()  # consume '('
                arg = self.Expr()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
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
                # Single-letter variable
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
            tok = self.PopNextToken()
            if tok is None:
                raise Exception("Unexpected end")
            try:
                if '.' in tok or 'e' in tok or 'E' in tok:
                    val = float(tok)
                else:
                    val = int(tok)
                return Polynomial([val])
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {tok}")

    def Power(self):
        result = self.Value()
        next_tok = self.PeekNextToken()
        if next_tok == "^":
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


def calc8(expression):
    '''
    Extends calc7 to support single-variable algebra and equation solving.
    - Without '=': simplify expression and return string
    - With '=': solve equation and return solutions
    '''
    if '=' in expression:
        sides = expression.split('=')
        if len(sides) != 2:
            raise Exception("Only one '=' sign allowed")
        left_expr, right_expr = sides

        calc_left = Calculator8(left_expr)
        left = calc_left.Parse()
        var_left = calc_left._var_name

        calc_right = Calculator8(right_expr)
        right = calc_right.Parse()
        var_right = calc_right._var_name

        # Determine variable name
        var = var_left or var_right or 'x'

        # Ensure both are Polynomials
        if not isinstance(left, Polynomial):
            left = Polynomial([left], var=var)
        if not isinstance(right, Polynomial):
            right = Polynomial([right], var=var)

        # Set var on constant polynomials
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
        calculator = Calculator8(expression)
        result = calculator.Parse()
        if isinstance(result, Polynomial):
            if result.is_constant():
                val = result.constant_value()
                return _format_solution(val)
            return str(result)
        return _format_solution(result)


def _format_solution(value):
    """Format a numeric value for output."""
    value = Polynomial._to_int(value)
    if isinstance(value, complex):
        return format_complex(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return str(int(value))
        return str(value)
    return str(value)


def test(expression, expected, op = calc, exception = None):
    caught = None
    try:
        result = op(expression)
    except Exception as e:
        caught = e
    
    if exception is not None:
        if type(caught) == type(exception):
            print(f"✅ Testing {expression}, got expected exception {caught}")
        else:
            print(f"❌ Testing {expression}, expected exception {exception}, got {caught}")
    else:
        if caught is not None:
            print(f"❌ Testing {expression}, got exception {caught}")
        else:
            if result == expected:
                print(f"✅ Testing {expression}, got expected {expected}")
            else:
                print(f"❌ Testing {expression}, expected {expected}, result {result}")

if __name__ == '__main__':
    test("1+2+3", "6", calc)
    test("123+456 - 789", "-210", calc)
    test("123-456", "-333", calc)

    test("1+2+3", "6", calc2)
    test("123+456 - 789", "-210", calc2)
    test("123-456", "-333", calc2)
    test("1*2*3", "6", calc2)
    test("123+456*789", "359907", calc2)
    test("1+2*3-4", "3", calc2)
    test("1+2*3-5/4", "5.75", calc2)
    test("1*2*3*4*5/6", "20.0", calc2)

    test("1+2+3", "6", calc3)
    test("123+456 - 789", "-210", calc3)
    test("123-456", "-333", calc3)
    test("1*2*3", "6", calc3)
    test("123+456*789", "359907", calc3)
    test("1+2*3-4", "3", calc3)
    test("1+2*3-5/4", "5.75", calc3)
    test("1*2*3*4*5/6", "20.0", calc3)
    test("1+2*(3-4)", "-1", calc3)
    test("(3^5+2)/(7*7)", "5.0", calc3)
    test("1**2", "", calc3, Exception())
    test("", "", calc3, Exception())

    test("1+2+3", "6", calc4)
    test("1+2*3-4", "3", calc4)
    test("1e2", "100.0", calc4)
    test("1.5e3", "1500.0", calc4)
    test("2.5e-3", "0.0025", calc4)
    test("1.5E+3", "1500.0", calc4)
    test("1e2+1.5e2", "250.0", calc4)
    test("1.5e3*2", "3000.0", calc4)
    test("(1e2+1.5e2)*2e1", "5000.0", calc4)
    test("1.5e3/1.5e2", "10.0", calc4)
    test("", "", calc4, Exception())

    test("pi", "3.141592653589793", calc5)
    test("e", "2.718281828459045", calc5)
    test("2*pi", "6.283185307179586", calc5)
    test("pi+e", "5.859874482048838", calc5)
    test("e^2", "7.3890560989306495", calc5)
    test("1e2*pi", "314.1592653589793", calc5)
    test("1+2", "3", calc5)
    test("foo", "", calc5, Exception())

    test("i", "i", calc6)
    test("i*i", "-1", calc6)
    test("i^2", "-1", calc6)
    test("1+i", "1+i", calc6)
    test("1-i", "1-i", calc6)
    test("(1+i)*2", "2+2i", calc6)
    test("(1+i)*(1-i)", "2", calc6)
    test("(1+i)/(1-i)", "i", calc6)
    test("2+3*i", "2+3i", calc6)
    test("e^(i*pi)", "-1", calc6)
    test("1+2", "3", calc6)

    test("sin(0)", "0", calc7)
    test("cos(0)", "1", calc7)
    test("sin(pi/2)", "1", calc7)
    test("tan(pi/4)", "1", calc7)
    test("sinh(0)", "0", calc7)
    test("cosh(0)", "1", calc7)
    test("exp(1)", "2.718281828459045", calc7)
    test("ln(1)", "0", calc7)
    test("log(100)", "2", calc7)
    test("sqrt(4)", "2", calc7)
    test("sqrt(0-1)", "i", calc7)
    test("abs(3+4*i)", "5", calc7)
    test("e^(i*pi)", "-1", calc7)
    test("1+2", "3", calc7)

    # calc8: Simplification tests
    test("x", "x", calc8)
    test("2*x+3*x", "5*x", calc8)
    test("x*x", "x^2", calc8)
    test("2+3", "5", calc8)
    test("x+1-1", "x", calc8)
    test("(x+1)*(x-1)", "x^2-1", calc8)
    test("3*x^2+2*x+1", "3*x^2+2*x+1", calc8)

    # calc8: Linear equation tests
    test("2*x=4", "x=2", calc8)
    test("x+1=3", "x=2", calc8)
    test("3*x+2=x+10", "x=4", calc8)
    test("x=5", "x=5", calc8)
    test("2*(x+1)=6", "x=2", calc8)

    # calc8: Quadratic equation tests
    test("x^2=1", "x=-1; x=1", calc8)
    test("x^2+2*x+1=0", "x=-1", calc8)
    test("x^2-5*x+6=0", "x=2; x=3", calc8)
