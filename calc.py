import math
import cmath

def calc1(expression):
    '''Evaluate addition and subtraction of integers.'''
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
    '''Evaluate addition, subtraction, multiplication, and division with operator precedence.'''
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
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        elif next is not None and next[0].isalpha():
            next = self.PopNextToken()
            if self.PeekNextToken() == "(":
                if next not in self.FUNCTIONS:
                    raise Exception(f"Unknown function '{next}'")
                func = self.FUNCTIONS[next]
                next = self.PopNextToken()
                result = self.Expr()
                next = self.PopNextToken()
                if next != ")":
                    raise Exception(f"Invalid token {next}")
                result = func(result)
            else:
                if next not in self.CONSTANTS:
                    raise Exception(f"Unknown constant '{next}'")
                result = self.CONSTANTS[next]
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


def _to_int(value):
    if isinstance(value, complex):
        if value.imag == 0:
            value = value.real
        else:
            return value
    if isinstance(value, float) and value == int(value) and math.isfinite(value):
        return int(value)
    return value

def _fmt_num(x):
    x = _to_int(x)
    if isinstance(x, complex):
        return format_complex(x)
    if isinstance(x, float):
        if math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)
    return str(x)

def _cbrt(z):
    z = complex(z)
    if z == 0:
        return complex(0)
    r = abs(z)
    theta = cmath.phase(z)
    return (r ** (1.0 / 3.0)) * cmath.exp(complex(0, theta / 3.0))

def _clean_root(r, eps=1e-9):
    if isinstance(r, complex):
        real = r.real
        imag = r.imag
        if abs(real) < eps:
            real = 0.0
        if abs(imag) < eps:
            imag = 0.0
        if abs(real - round(real)) < eps:
            real = round(real)
        if abs(imag - round(imag)) < eps:
            imag = round(imag)
        if imag == 0:
            return _to_int(real)
        return _to_int(complex(real, imag))
    else:
        if abs(r) < eps:
            return 0
        if abs(r - round(r)) < eps:
            return int(round(r))
        return _to_int(r)

def _unique_roots(roots, eps=1e-9):
    unique = []
    for r in roots:
        is_dup = False
        for u in unique:
            if isinstance(r, complex) or isinstance(u, complex):
                rc = complex(r) if not isinstance(r, complex) else r
                uc = complex(u) if not isinstance(u, complex) else u
                if abs(rc - uc) < eps:
                    is_dup = True
                    break
            else:
                if abs(r - u) < eps:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(r)
    return unique

def _sort_roots(roots):
    real_roots = []
    complex_roots = []
    for r in roots:
        if isinstance(r, complex):
            complex_roots.append(r)
        else:
            real_roots.append(r)
    real_roots.sort()
    complex_roots.sort(key=lambda v: (v.real, v.imag))
    return real_roots + complex_roots

def _format_solution(value):
    value = _to_int(value)
    if isinstance(value, complex):
        return format_complex(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return str(int(value))
        return str(value)
    return str(value)


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


class Calculator8(Calculator7):
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
                if name not in self.FUNCTIONS:
                    raise Exception(f"Unknown function '{name}'")
                self.PopNextToken()
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

    def Power(self):
        result = self.Value()
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


def calc8(expression):
    '''
    Extends calc7 to support single-variable algebra and equation solving.
    Without '=': simplify expression (eg. 2*x+3*x -> 5*x)
    With '=': solve equation (eg. x^2=1 -> x=-1; x=1)
    Supports linear, quadratic, and cubic equations.
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
        calculator = Calculator8(expression)
        result = calculator.Parse()
        if isinstance(result, Polynomial):
            if result.is_constant():
                val = result.constant_value()
                return _format_solution(val)
            return str(result)
        return _format_solution(result)


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


class Calculator9(Calculator8):
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
                if name not in self.FUNCTIONS:
                    raise Exception(f"Unknown function '{name}'")
                self.PopNextToken()
                arg = self.Expr()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
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
        while next == "*" or next == "/":
            self.PopNextToken()
            right = self.Power()
            if self._multi_mode:
                result = self._promote_to_multi(result)
                right = self._promote_to_multi(right)
            if next == "*":
                result = result * right
            elif next == "/":
                result = result / right
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

    def Power(self):
        result = self.Value()
        next = self.PeekNextToken()
        if next == "^":
            self.PopNextToken()
            exponent = self.Power()
            if self._multi_mode:
                result = self._promote_to_multi(result)
                exponent = self._promote_to_multi(exponent)
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


def calc9(expression):
    '''
    Extends calc8 to support multi-variable linear algebra.
    Multiple equations separated by ';' are solved as a system.
    eg. x+y=2; x-y=0 -> x=1; y=1
    Supports up to 3 variables.
    '''
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

            calc_left = Calculator9(left_str)
            left = calc_left.Parse()
            calc_right = Calculator9(right_str)
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

        calc_left = Calculator9(left_expr)
        left = calc_left.Parse()
        var_left = calc_left._var_name
        vars_left = calc_left._var_names

        calc_right = Calculator9(right_expr)
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
        calculator = Calculator9(expression)
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
