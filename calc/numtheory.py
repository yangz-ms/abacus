import math
import cmath
from fractions import Fraction
from functools import reduce

from calc.core import Calculator5, format_complex
from calc.registry import register
from calc.helpers import _to_int, _fmt_num


def _prime_factors(n):
    """Return prime factorization as list of (prime, exponent) tuples."""
    n = int(abs(n))
    factors = []
    d = 2
    while d * d <= n:
        exp = 0
        while n % d == 0:
            n //= d
            exp += 1
        if exp > 0:
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def _format_factors(factors):
    """Format factor list as string like 2^2*3*5."""
    parts = []
    for prime, exp in factors:
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    return '*'.join(parts) if parts else '1'


def _isprime(n):
    """Return 1 if n is prime, 0 otherwise."""
    n = int(abs(n))
    if n < 2:
        return 0
    if n < 4:
        return 1
    if n % 2 == 0 or n % 3 == 0:
        return 0
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return 0
        d += 6
    return 1


# ---------------------------------------------------------------------------
# Calculator6: GCD, LCM, primes, modulo, floor, ceil, rounding
# ---------------------------------------------------------------------------

class Calculator6(Calculator5):
    """Number theory: gcd, lcm, modulo, prime factorization, rounding."""

    MULTI_FUNCTIONS = {
        'gcd': lambda args: reduce(math.gcd, [int(a) for a in args]),
        'lcm': lambda args: reduce(math.lcm, [int(a) for a in args]),
        'round': lambda args: round(args[0], int(args[1])) if len(args) > 1 else round(args[0]),
    }

    FUNCTIONS = {
        **Calculator5.FUNCTIONS,
        'floor': math.floor,
        'ceil': math.ceil,
        'isprime': lambda x: _isprime(x),
        'factor': lambda x: _format_factors(_prime_factors(x)),
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
                self.exp.append(expression[i:j])
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')', '%', ','):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def _parse_function_args(self):
        """Parse comma-separated arguments inside function call parentheses."""
        args = [self.Expr()]
        while self.PeekNextToken() == ",":
            self.PopNextToken()  # consume ','
            args.append(self.Expr())
        return args

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
                if name in self.MULTI_FUNCTIONS:
                    self.PopNextToken()  # consume '('
                    args = self._parse_function_args()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = self.MULTI_FUNCTIONS[name](args)
                elif name in self.FUNCTIONS:
                    func = self.FUNCTIONS[name]
                    self.PopNextToken()  # consume '('
                    arg = self.Expr()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = func(arg)
                    if isinstance(result, complex) and result.imag == 0:
                        result = result.real
                else:
                    raise Exception(f"Unknown function '{name}'")
            elif hasattr(self, 'CONSTANTS') and name in self.CONSTANTS:
                result = self.CONSTANTS[name]
            else:
                raise Exception(f"Unknown identifier '{name}'")
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

    def Product(self):
        result = self.Power()
        next = self.PeekNextToken()
        while next == "*" or next == "/" or next == "%":
            self.PopNextToken()
            nextResult = self.Power()
            if next == "*":
                result *= nextResult
            elif next == "/":
                if isinstance(result, (int, Fraction)) and isinstance(nextResult, (int, Fraction)):
                    result = Fraction(result) / Fraction(nextResult)
                else:
                    result /= nextResult
            elif next == "%":
                result = int(result) % int(nextResult)
            next = self.PeekNextToken()
        return result


@register("calc6", description="GCD, LCM, primes, modulo, and rounding",
          short_desc="Number Theory", group="expression",
          examples=["gcd(12,8)", "17%3", "factor(60)", "floor(3.7)"],
          i18n={"zh": "数论", "hi": "संख्या सिद्धांत", "es": "Teoría de Números", "fr": "Théorie des Nombres", "ar": "نظرية الأعداد", "pt": "Teoria dos Números", "ru": "Теория чисел", "ja": "数論", "de": "Zahlentheorie"})
def calc6(expression):
    """GCD, LCM, primes, modulo, and rounding."""
    calculator = Calculator6(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# ---------------------------------------------------------------------------
# Calculator7: Pi, e, Scientific Notation
# ---------------------------------------------------------------------------

class Calculator7(Calculator6):
    """Named constants (pi, e) and scientific notation."""

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
                # Scientific notation: only consume e/E if followed by digit or sign+digit
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
            elif c in ('+', '-', '*', '/', '^', '(', ')', '%', ','):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

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
                if name in self.MULTI_FUNCTIONS:
                    self.PopNextToken()  # consume '('
                    args = self._parse_function_args()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = self.MULTI_FUNCTIONS[name](args)
                elif name in self.FUNCTIONS:
                    func = self.FUNCTIONS[name]
                    self.PopNextToken()  # consume '('
                    arg = self.Expr()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = func(arg)
                    if isinstance(result, complex) and result.imag == 0:
                        result = result.real
                else:
                    raise Exception(f"Unknown function '{name}'")
            elif name in self.CONSTANTS:
                result = self.CONSTANTS[name]
            else:
                raise Exception(f"Unknown identifier '{name}'")
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


@register("calc7", description="Named constants (pi, e) and scientific notation",
          short_desc="Pi, e & Sci. Notation", group="expression",
          examples=["2*pi", "e^2", "1.5e3*2"],
          i18n={"zh": "常量与科学计数法", "hi": "स्थिरांक और वैज्ञानिक संकेतन", "es": "Constantes y Notación Científica", "fr": "Constantes et Notation Scientifique", "ar": "الثوابت والترميز العلمي", "pt": "Constantes e Notação Científica", "ru": "Константы и научная запись", "ja": "定数と科学表記法", "de": "Konstanten und Wissenschaftliche Notation"})
def calc7(expression):
    """Named constants (pi, e) and scientific notation."""
    calculator = Calculator7(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# ---------------------------------------------------------------------------
# Calculator8: Basic Trig + Degree Mode
# ---------------------------------------------------------------------------

class Calculator8(Calculator7):
    """Basic trigonometry (sin, cos, tan) with degree mode."""

    FUNCTIONS = {
        **Calculator7.FUNCTIONS,
        'sin':  cmath.sin,
        'cos':  cmath.cos,
        'tan':  cmath.tan,
        'asin': cmath.asin,
        'acos': cmath.acos,
        'atan': cmath.atan,
        'sinh': cmath.sinh,
        'cosh': cmath.cosh,
        'tanh': cmath.tanh,
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
                # Scientific notation: only consume e/E if followed by digit or sign+digit
                if j < len(expression) and expression[j] in ('e', 'E'):
                    k = j + 1
                    if k < len(expression) and expression[k] in ('+', '-'):
                        k += 1
                    if k < len(expression) and expression[k] >= '0' and expression[k] <= '9':
                        j = k
                        while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                            j += 1
                num_str = expression[i:j]
                # Degree mode: number followed by 'd' (not part of longer identifier)
                if j < len(expression) and expression[j] == 'd':
                    if j + 1 >= len(expression) or not expression[j + 1].isalpha():
                        j += 1  # consume the 'd'
                        val = float(num_str) * math.pi / 180
                        self.exp.append(str(val))
                        i = j
                        continue
                self.exp.append(num_str)
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')', '%', ','):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")


@register("calc8", description="Basic trigonometry (sin, cos, tan) with degree mode",
          short_desc="Basic Trig", group="expression",
          examples=["sin(pi/2)", "cos(180d)", "tan(45d)"],
          i18n={"zh": "基本三角函数", "hi": "बुनियादी त्रिकोणमिति", "es": "Trigonometría Básica", "fr": "Trigonométrie de Base", "ar": "حساب المثلثات الأساسي", "pt": "Trigonometria Básica", "ru": "Базовая тригонометрия", "ja": "基本三角関数", "de": "Grundlegende Trigonometrie"})
def calc8(expression):
    """Basic trigonometry (sin, cos, tan) with degree mode."""
    calculator = Calculator8(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# ---------------------------------------------------------------------------
# Calculator9: Exponential & Log Functions
# ---------------------------------------------------------------------------

class Calculator9(Calculator8):
    """Exponential and logarithmic functions (exp, ln, log, logb)."""

    FUNCTIONS = {
        **Calculator8.FUNCTIONS,
        'exp': cmath.exp,
        'ln':  cmath.log,
        'log': cmath.log10,
    }

    MULTI_FUNCTIONS = {
        **Calculator8.MULTI_FUNCTIONS,
        'logb': lambda args: cmath.log(args[1]) / cmath.log(args[0]),
    }


@register("calc9", description="Exponential and logarithmic functions (exp, ln, log, logb)",
          short_desc="Exp & Log", group="expression",
          examples=["exp(1)", "ln(1)", "log(100)", "logb(2,8)"],
          i18n={"zh": "指数与对数", "hi": "घातीय और लघुगणक", "es": "Exponencial y Logaritmos", "fr": "Exponentielle et Logarithmes", "ar": "الأسية واللوغاريتمية", "pt": "Exponencial e Logaritmos", "ru": "Экспонента и логарифмы", "ja": "指数と対数", "de": "Exponential und Logarithmen"})
def calc9(expression):
    """Exponential and logarithmic functions (exp, ln, log, logb)."""
    calculator = Calculator9(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# ---------------------------------------------------------------------------
# Calculator10: Factorial & Combinatorics
# ---------------------------------------------------------------------------

class Calculator10(Calculator9):
    """Factorial and combinatorics (n!, C, P)."""

    MULTI_FUNCTIONS = {
        **Calculator9.MULTI_FUNCTIONS,
        'C': lambda args: math.comb(int(args[0]), int(args[1])),
        'P': lambda args: math.perm(int(args[0]), int(args[1])),
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
                # Scientific notation: only consume e/E if followed by digit or sign+digit
                if j < len(expression) and expression[j] in ('e', 'E'):
                    k = j + 1
                    if k < len(expression) and expression[k] in ('+', '-'):
                        k += 1
                    if k < len(expression) and expression[k] >= '0' and expression[k] <= '9':
                        j = k
                        while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                            j += 1
                num_str = expression[i:j]
                # Degree mode: number followed by 'd' (not part of longer identifier)
                if j < len(expression) and expression[j] == 'd':
                    if j + 1 >= len(expression) or not expression[j + 1].isalpha():
                        j += 1  # consume the 'd'
                        val = float(num_str) * math.pi / 180
                        self.exp.append(str(val))
                        i = j
                        continue
                self.exp.append(num_str)
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')', '!', '%', ','):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def Value(self):
        next = self.PeekNextToken()
        # Special handling for C/P since they are uppercase and could conflict
        if next is not None and next[0].isalpha() and next in self.MULTI_FUNCTIONS:
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                self.PopNextToken()  # consume '('
                args = self._parse_function_args()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
                return self.MULTI_FUNCTIONS[name](args)
            elif name in self.CONSTANTS:
                return self.CONSTANTS[name]
            else:
                raise Exception(f"Unknown identifier '{name}'")
        return super().Value()

    def Postfix(self):
        result = self.Value()
        while self.PeekNextToken() == "!":
            self.PopNextToken()
            result = math.factorial(int(result))
        return result

    def Power(self):
        result = self.Postfix()
        next = self.PeekNextToken()
        if next == "^":
            self.PopNextToken()
            nextResult = self.Power()
            result = pow(result, nextResult)
        return result


@register("calc10", description="Factorial and combinatorics (n!, C, P)",
          short_desc="Factorial & Combinatorics", group="expression",
          examples=["5!", "C(10,3)", "P(5,2)"],
          i18n={"zh": "阶乘与组合", "hi": "क्रमगुणित और संयोजन", "es": "Factorial y Combinatoria", "fr": "Factorielle et Combinatoire", "ar": "العاملي والتوافيق", "pt": "Fatorial e Combinatória", "ru": "Факториал и комбинаторика", "ja": "階乗と組合せ", "de": "Fakultät und Kombinatorik"})
def calc10(expression):
    """Factorial and combinatorics (n!, C, P)."""
    calculator = Calculator10(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# ---------------------------------------------------------------------------
# Calculator11: Complex Numbers
# ---------------------------------------------------------------------------

class Calculator11(Calculator10):
    """Complex numbers (imaginary unit i)."""

    CONSTANTS = {
        **Calculator10.CONSTANTS,
        'i': complex(0, 1),
    }

    FUNCTIONS = {
        **Calculator10.FUNCTIONS,
        'abs': abs,
    }


@register("calc11", description="Complex numbers (imaginary unit i)",
          short_desc="Complex Numbers", group="expression",
          examples=["i^2", "(1+i)*(1-i)", "(1+i)/(1-i)", "e^(i*pi)"],
          i18n={"zh": "复数运算", "hi": "सम्मिश्र संख्याएँ", "es": "Números Complejos", "fr": "Nombres Complexes", "ar": "الأعداد المركبة", "pt": "Números Complexos", "ru": "Комплексные числа", "ja": "複素数", "de": "Komplexe Zahlen"})
def calc11(expression):
    """Complex numbers (imaginary unit i)."""
    calculator = Calculator11(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)
