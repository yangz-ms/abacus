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

    @staticmethod
    def _cbrt(z):
        """Compute the principal cube root of a complex number."""
        z = complex(z)
        if z == 0:
            return complex(0)
        r = abs(z)
        theta = cmath.phase(z)
        return (r ** (1.0 / 3.0)) * cmath.exp(complex(0, theta / 3.0))

    @staticmethod
    def _clean_root(r, eps=1e-9):
        """Round near-integer and near-zero parts of a root."""
        if isinstance(r, complex):
            real = r.real
            imag = r.imag
            # Clean near-zero
            if abs(real) < eps:
                real = 0.0
            if abs(imag) < eps:
                imag = 0.0
            # Clean near-integer
            if abs(real - round(real)) < eps:
                real = round(real)
            if abs(imag - round(imag)) < eps:
                imag = round(imag)
            if imag == 0:
                return Polynomial._to_int(real)
            return Polynomial._to_int(complex(real, imag))
        else:
            if abs(r) < eps:
                return 0
            if abs(r - round(r)) < eps:
                return int(round(r))
            return Polynomial._to_int(r)

    @staticmethod
    def _unique_roots(roots, eps=1e-9):
        """Remove duplicate roots (within epsilon tolerance)."""
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

    @staticmethod
    def _sort_roots(roots):
        """Sort roots: real first (ascending), then complex by real part then imag part."""
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

        if deg == 3:
            a = self.coeffs[3]
            b = self.coeffs[2]
            c = self.coeffs[1]
            d = self.coeffs[0]

            # Depressed cubic substitution: x = t - b/(3a)
            # Converts ax^3 + bx^2 + cx + d = 0 to t^3 + pt + q = 0
            shift = b / (3 * a)
            p = (3 * a * c - b * b) / (3 * a * a)
            q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)

            # Cardano's discriminant
            disc = -(4 * p * p * p + 27 * q * q)

            eps = 1e-9

            if abs(p) < eps and abs(q) < eps:
                # Triple root: t = 0, so x = -shift
                root = -shift
                root = self._clean_root(root, eps)
                return [root]

            if abs(disc) < eps:
                # Double root case
                if abs(q) < eps:
                    root = -shift
                    root = self._clean_root(root, eps)
                    return [root]
                else:
                    root1 = 3 * q / p - shift
                    root2 = -3 * q / (2 * p) - shift
                    root1 = self._clean_root(root1, eps)
                    root2 = self._clean_root(root2, eps)
                    roots = sorted(set([root1, root2]), key=lambda v: v.real if isinstance(v, complex) else v)
                    return roots

            # General Cardano's formula using complex arithmetic
            # t^3 + pt + q = 0
            inner = complex(q * q / 4 + p * p * p / 27)
            sqrt_inner = cmath.sqrt(inner)

            u_base = -q / 2 + sqrt_inner
            v_base = -q / 2 - sqrt_inner

            # Cube roots
            u = self._cbrt(u_base)
            v = self._cbrt(v_base)

            # The three cube roots of unity
            omega = complex(-0.5, math.sqrt(3) / 2)
            omega2 = complex(-0.5, -math.sqrt(3) / 2)

            # Choose v for each u such that u*v = -p/3
            # Three roots of depressed cubic
            t1 = u + v
            t2 = u * omega + v * omega2
            t3 = u * omega2 + v * omega

            # Convert back: x = t - shift
            raw_roots = [t1 - shift, t2 - shift, t3 - shift]

            # Clean up roots
            solutions = []
            for r in raw_roots:
                r = self._clean_root(r, eps)
                solutions.append(r)

            # Remove duplicates
            solutions = self._unique_roots(solutions, eps)

            # Sort: real roots first (ascending), then complex by real part then imag part
            solutions = self._sort_roots(solutions)

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


class MultiPolynomial:
    """Represents a linear expression in multiple variables.
    Stored as a dict: {'': constant, 'x': coeff_x, 'y': coeff_y, ...}
    Only supports degree 1 per variable (linear terms only for multi-variable).
    """

    def __init__(self, terms=None):
        if terms is None:
            terms = {'': 0}
        self.terms = {}
        for k, v in terms.items():
            v = Polynomial._to_int(v)
            if v != 0 or k == '':
                self.terms[k] = v
        if '' not in self.terms:
            self.terms[''] = 0

    @staticmethod
    def from_variable(var):
        """Create a MultiPolynomial representing a single variable."""
        return MultiPolynomial({'': 0, var: 1})

    @staticmethod
    def from_constant(value):
        """Create a MultiPolynomial representing a constant."""
        return MultiPolynomial({'': Polynomial._to_int(value)})

    def is_constant(self):
        """Check if this is a constant (no variable terms)."""
        return all(v == 0 for k, v in self.terms.items() if k != '')

    def constant_value(self):
        """Get the constant value (only valid if is_constant())."""
        if not self.is_constant():
            raise Exception("MultiPolynomial is not a constant")
        return self.terms.get('', 0)

    def variables(self):
        """Return sorted list of variables with non-zero coefficients."""
        return sorted(k for k, v in self.terms.items() if k != '' and v != 0)

    def get_coeff(self, var):
        """Get the coefficient of a variable."""
        return self.terms.get(var, 0)

    def _clean(self):
        """Remove zero-coefficient variable terms."""
        to_remove = [k for k, v in self.terms.items() if k != '' and v == 0]
        for k in to_remove:
            del self.terms[k]

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = MultiPolynomial.from_constant(other)
        if isinstance(other, Polynomial):
            if other.is_constant():
                other = MultiPolynomial.from_constant(other.constant_value())
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
            other = MultiPolynomial.from_constant(other)
        if isinstance(other, Polynomial):
            if other.is_constant():
                other = MultiPolynomial.from_constant(other.constant_value())
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
            return MultiPolynomial.from_constant(other / val)
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
                return MultiPolynomial.from_constant(pow(self.constant_value(), other))
            # For non-constant, only support integer exponents
            if isinstance(other, complex) or other != int(other) or other < 0:
                raise Exception("MultiPolynomial exponent must be a non-negative integer")
            exp = int(other)
            if exp == 0:
                return MultiPolynomial.from_constant(1)
            result = self
            for _ in range(exp - 1):
                result = result * self
            return result
        return NotImplemented

    def __rpow__(self, other):
        if not self.is_constant():
            raise Exception("Exponent must be a constant MultiPolynomial")
        return MultiPolynomial.from_constant(pow(other, self.constant_value()))

    def __str__(self):
        if self.is_constant():
            return _format_solution(self.terms.get('', 0))

        parts = []
        # Sort variables alphabetically
        var_keys = sorted(k for k in self.terms if k != '' and self.terms[k] != 0)

        for var in var_keys:
            c = Polynomial._to_int(self.terms[var])
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

        # Constant term
        const = Polynomial._to_int(self.terms.get('', 0))
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


def solve_linear_system(equations, variables):
    """Solve a system of linear equations using Gaussian elimination.

    Args:
        equations: list of MultiPolynomial (each represents equation = 0)
        variables: list of variable names

    Returns:
        dict mapping variable name to solution value
    """
    n = len(variables)
    m = len(equations)

    if m < n:
        raise Exception("Underdetermined system: not enough equations")

    # Build augmented matrix [A | b] where A*vars = -constant
    # Each equation: c_x1 * x1 + c_x2 * x2 + ... + constant = 0
    # So: c_x1 * x1 + c_x2 * x2 + ... = -constant
    matrix = []
    for eq in equations:
        row = []
        for var in variables:
            row.append(float(eq.get_coeff(var)))
        row.append(-float(eq.terms.get('', 0)))
        matrix.append(row)

    # Gaussian elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(matrix[col][col])
        max_row = col
        for row in range(col + 1, m):
            if abs(matrix[row][col]) > max_val:
                max_val = abs(matrix[row][col])
                max_row = row

        if max_val < 1e-12:
            raise Exception("System has no unique solution (singular matrix)")

        # Swap rows
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]

        # Eliminate below
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

    # Clean up solutions: convert to int if close to integer
    result = {}
    for i, var in enumerate(variables):
        val = solution[i]
        if abs(val - round(val)) < 1e-9:
            val = int(round(val))
        else:
            val = Polynomial._to_int(val)
        result[var] = val

    return result


class Calculator9(Calculator8):
    """Extends Calculator8 to support multi-variable linear algebra."""
    _var_names = None  # Track all variable names encountered
    _multi_mode = False  # Whether we've entered multi-variable mode

    def __init__(self, expression):
        self._var_names = set()
        self._multi_mode = False
        super().__init__(expression)

    def _promote_to_multi(self, value):
        """Convert a value to MultiPolynomial if needed."""
        if isinstance(value, MultiPolynomial):
            return value
        if isinstance(value, Polynomial):
            if value.is_constant():
                return MultiPolynomial.from_constant(value.constant_value())
            # Convert single-var polynomial to multi
            if value.degree > 1:
                raise Exception("Multi-variable mode only supports linear expressions per variable")
            terms = {'': Polynomial._to_int(value.coeffs[0])}
            if len(value.coeffs) > 1:
                terms[value.var] = Polynomial._to_int(value.coeffs[1])
            return MultiPolynomial(terms)
        if isinstance(value, (int, float, complex)):
            return MultiPolynomial.from_constant(value)
        return value

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
                # Functions only work on constants
                if isinstance(arg, MultiPolynomial):
                    if arg.is_constant():
                        val = arg.constant_value()
                        return MultiPolynomial.from_constant(self.FUNCTIONS[name](val))
                    else:
                        raise Exception(f"Cannot apply function '{name}' to expression with variables")
                elif isinstance(arg, Polynomial):
                    if arg.is_constant():
                        val = arg.constant_value()
                        if self._multi_mode:
                            return MultiPolynomial.from_constant(self.FUNCTIONS[name](val))
                        result = self.FUNCTIONS[name](val)
                        return Polynomial([result], var=arg.var)
                    else:
                        raise Exception(f"Cannot apply function '{name}' to polynomial with variable")
                else:
                    result = self.FUNCTIONS[name](arg)
                    if self._multi_mode:
                        return MultiPolynomial.from_constant(result)
                    return result
            elif self._is_variable(name):
                # Single-letter variable
                self._var_names.add(name)
                if len(self._var_names) > 1 and not self._multi_mode:
                    self._multi_mode = True
                if self._multi_mode:
                    return MultiPolynomial.from_variable(name)
                else:
                    # Single variable mode, use Polynomial
                    if self._var_name is None:
                        self._var_name = name
                    elif self._var_name != name:
                        # This shouldn't happen because we check _var_names above
                        pass
                    return Polynomial([0, 1], var=name)
            elif name in self.CONSTANTS:
                if self._multi_mode:
                    return MultiPolynomial.from_constant(self.CONSTANTS[name])
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
                if self._multi_mode:
                    return MultiPolynomial.from_constant(val)
                return Polynomial([val])
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {tok}")

    def Product(self):
        result = self.Power()
        next_tok = self.PeekNextToken()
        while next_tok == "*" or next_tok == "/":
            self.PopNextToken()
            right = self.Power()
            # Promote if needed for multi-mode
            if self._multi_mode:
                result = self._promote_to_multi(result)
                right = self._promote_to_multi(right)
            if next_tok == "*":
                result = result * right
            elif next_tok == "/":
                result = result / right
            next_tok = self.PeekNextToken()
        return result

    def Sum(self):
        result = self.Product()
        next_tok = self.PeekNextToken()
        while next_tok == "+" or next_tok == "-":
            self.PopNextToken()
            right = self.Product()
            # Promote if needed for multi-mode
            if self._multi_mode:
                result = self._promote_to_multi(result)
                right = self._promote_to_multi(right)
            if next_tok == "+":
                result = result + right
            elif next_tok == "-":
                result = result - right
            next_tok = self.PeekNextToken()
        return result

    def Power(self):
        result = self.Value()
        next_tok = self.PeekNextToken()
        if next_tok == "^":
            self.PopNextToken()
            exponent = self.Power()
            # Promote if needed for multi-mode
            if self._multi_mode:
                result = self._promote_to_multi(result)
                exponent = self._promote_to_multi(exponent)
            result = result ** exponent
        return result


def calc9(expression):
    '''
    Extends calc8 to support multi-variable linear algebra and systems of linear equations.
    - Multiple equations separated by ';' are solved as a system
    - Single-variable expressions fall back to calc8 behavior
    - Multi-variable expressions are simplified or solved
    '''
    # Check for system of equations (multiple ';'-separated parts that contain '=')
    if ';' in expression:
        equation_strs = [s.strip() for s in expression.split(';')]
        # Parse each equation
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

            # Collect variables
            for c in [calc_left, calc_right]:
                all_vars.update(c._var_names)

            # Promote to MultiPolynomial
            if isinstance(left, Polynomial):
                if left.is_constant():
                    left = MultiPolynomial.from_constant(left.constant_value())
                else:
                    terms = {'': Polynomial._to_int(left.coeffs[0])}
                    if len(left.coeffs) > 1:
                        terms[left.var] = Polynomial._to_int(left.coeffs[1])
                    left = MultiPolynomial(terms)
            elif isinstance(left, (int, float, complex)):
                left = MultiPolynomial.from_constant(left)
            if isinstance(right, Polynomial):
                if right.is_constant():
                    right = MultiPolynomial.from_constant(right.constant_value())
                else:
                    terms = {'': Polynomial._to_int(right.coeffs[0])}
                    if len(right.coeffs) > 1:
                        terms[right.var] = Polynomial._to_int(right.coeffs[1])
                    right = MultiPolynomial(terms)
            elif isinstance(right, (int, float, complex)):
                right = MultiPolynomial.from_constant(right)

            eq = left - right
            all_equations.append(eq)

        variables = sorted(all_vars)
        solution = solve_linear_system(all_equations, variables)
        parts = []
        for var in variables:
            parts.append(f"{var}={_format_solution(solution[var])}")
        return '; '.join(parts)

    elif '=' in expression:
        # Single equation
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
            # Single variable - use Polynomial solve (same as calc8)
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
            # Multi-variable equation - promote and solve as system
            if isinstance(left, Polynomial):
                if left.is_constant():
                    left = MultiPolynomial.from_constant(left.constant_value())
                else:
                    terms = {'': Polynomial._to_int(left.coeffs[0])}
                    if len(left.coeffs) > 1:
                        terms[left.var] = Polynomial._to_int(left.coeffs[1])
                    left = MultiPolynomial(terms)
            elif isinstance(left, (int, float, complex)):
                left = MultiPolynomial.from_constant(left)
            if isinstance(right, Polynomial):
                if right.is_constant():
                    right = MultiPolynomial.from_constant(right.constant_value())
                else:
                    terms = {'': Polynomial._to_int(right.coeffs[0])}
                    if len(right.coeffs) > 1:
                        terms[right.var] = Polynomial._to_int(right.coeffs[1])
                    right = MultiPolynomial(terms)
            elif isinstance(right, (int, float, complex)):
                right = MultiPolynomial.from_constant(right)

            eq = left - right
            variables = sorted(all_vars)
            solution = solve_linear_system([eq], variables)
            parts = []
            for var in variables:
                parts.append(f"{var}={_format_solution(solution[var])}")
            return '; '.join(parts)
    else:
        # No equation - simplify
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

    # calc8: More simplification tests
    test("0*x", "0", calc8)
    test("1*x", "x", calc8)
    test("x+x+x", "3*x", calc8)
    test("x^2+x^2", "2*x^2", calc8)
    test("(x+1)^2", "x^2+2*x+1", calc8)
    test("x*0", "0", calc8)
    test("x-x", "0", calc8)
    test("x^3", "x^3", calc8)
    test("x*x*x", "x^3", calc8)
    test("(x+1)*(x+1)", "x^2+2*x+1", calc8)
    test("2*x*3", "6*x", calc8)
    test("x^2-x^2", "0", calc8)
    test("(x+1)*(x+2)", "x^2+3*x+2", calc8)

    # calc8: More linear equation tests
    test("5*x=0", "x=0", calc8)
    test("x/2=3", "x=6", calc8)
    test("10-x=3", "x=7", calc8)
    test("x+x=8", "x=4", calc8)
    test("3*x-1=2*x+4", "x=5", calc8)

    # calc8: Quadratic equation tests
    test("x^2=1", "x=-1; x=1", calc8)
    test("x^2+2*x+1=0", "x=-1", calc8)
    test("x^2-5*x+6=0", "x=2; x=3", calc8)
    test("x^2=4", "x=-2; x=2", calc8)
    test("x^2-1=0", "x=-1; x=1", calc8)
    test("x^2+1=0", "x=-i; x=i", calc8)
    test("2*x^2-8=0", "x=-2; x=2", calc8)
    test("x^2-3*x=0", "x=0; x=3", calc8)
    test("x^2-4*x+4=0", "x=2", calc8)
    test("x^2+4*x+4=0", "x=-2", calc8)
    test("x^2-2*x-3=0", "x=-1; x=3", calc8)

    # calc8: Cubic equation tests
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc8)
    test("x^3-1=0", "x=1; x=-0.5-0.866025403784439i; x=-0.5+0.866025403784439i", calc8)
    test("x^3=0", "x=0", calc8)
    test("x^3-3*x^2+3*x-1=0", "x=1", calc8)
    test("x^3+x^2-x-1=0", "x=-1; x=1", calc8)
    test("x^3=8", "x=2; x=-1-1.732050807568877i; x=-1+1.732050807568877i", calc8)
    test("x^3+1=0", "x=-1; x=0.5-0.866025403784439i; x=0.5+0.866025403784439i", calc8)
    test("x^3-x=0", "x=-1; x=0; x=1", calc8)  # x(x^2-1) = x(x-1)(x+1)

    # calc9 single-variable (should work same as calc8)
    test("2*x+3*x", "5*x", calc9)
    test("2*x=4", "x=2", calc9)
    test("x^2=1", "x=-1; x=1", calc9)
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc9)
    test("2+3", "5", calc9)

    # Multi-variable simplification
    test("x+y+x", "2*x+y", calc9)
    test("3*x+2*y-x", "2*x+2*y", calc9)
    test("x+y-x-y", "0", calc9)
    test("2*x+3*y+x-y", "3*x+2*y", calc9)

    # Two-variable linear systems
    test("x+y=2; x-y=0", "x=1; y=1", calc9)
    test("2*x+3*y=7; x-y=1", "x=2; y=1", calc9)
    test("x+y=10; 2*x+y=15", "x=5; y=5", calc9)
    test("x+2*y=5; 3*x-y=1", "x=1; y=2", calc9)
    test("x=3; x+y=5", "x=3; y=2", calc9)

    # Three-variable linear systems
    test("x+y+z=6; x-y=0; x+z=4", "x=2; y=2; z=2", calc9)
    test("x+y+z=3; x+y-z=1; x-y+z=1", "x=1; y=1; z=1", calc9)
    test("x+y+z=10; x-y+z=4; x+y-z=2", "x=3; y=3; z=4", calc9)
    test("2*x+y-z=1; x+y+z=6; x-y+2*z=5", "x=1; y=2; z=3", calc9)
