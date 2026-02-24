import math
import cmath
from fractions import Fraction
from functools import reduce

from calc.core import Calculator7, format_complex
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


class Calculator8(Calculator7):
    """Number theory: gcd, lcm, factorial, modulo, prime factorization, rounding."""

    MULTI_FUNCTIONS = {
        'gcd': lambda args: reduce(math.gcd, [int(a) for a in args]),
        'lcm': lambda args: reduce(math.lcm, [int(a) for a in args]),
        'round': lambda args: round(args[0], int(args[1])) if len(args) > 1 else round(args[0]),
    }

    FUNCTIONS = {
        **Calculator7.FUNCTIONS,
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
            elif c in ('+', '-', '*', '/', '^', '(', ')', '!', '%', ','):
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
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        elif next is not None and next[0].isalpha():
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                if name in self.MULTI_FUNCTIONS:
                    self.PopNextToken()  # consume '('
                    args = [self.Expr()]
                    while self.PeekNextToken() == ",":
                        self.PopNextToken()  # consume ','
                        args.append(self.Expr())
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = self.MULTI_FUNCTIONS[name](args)
                elif name in self.FUNCTIONS:
                    self.PopNextToken()  # consume '('
                    arg = self.Expr()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = self.FUNCTIONS[name](arg)
                else:
                    raise Exception(f"Unknown function '{name}'")
            elif name in self.CONSTANTS:
                result = self.CONSTANTS[name]
            else:
                raise Exception(f"Unknown constant '{name}'")
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


@register("calc8", description="Number theory: GCD, LCM, factorial, prime factorization, modulo, rounding",
          short_desc="Number Theory", group="expression",
          examples=["gcd(12,8)", "5!", "factor(60)", "17%3", "floor(3.7)"],
          i18n={"zh": "\u6570\u8bba", "hi": "\u0938\u0902\u0916\u094d\u092f\u093e \u0938\u093f\u0926\u094d\u0927\u093e\u0902\u0924", "es": "Teor\u00eda de N\u00fameros", "fr": "Th\u00e9orie des Nombres", "ar": "\u0646\u0638\u0631\u064a\u0629 \u0627\u0644\u0623\u0639\u062f\u0627\u062f", "pt": "Teoria dos N\u00fameros", "ru": "\u0422\u0435\u043e\u0440\u0438\u044f \u0447\u0438\u0441\u0435\u043b", "ja": "\u6570\u8ad6", "de": "Zahlentheorie"})
def calc8(expression):
    """Number theory: gcd, lcm, factorial, modulo, prime factorization, rounding."""
    calculator = Calculator8(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# --- Calculator9: Combinatorics (permutations, combinations) ---

class Calculator9(Calculator8):
    """Combinatorics: permutations and combinations."""

    MULTI_FUNCTIONS = {
        **Calculator8.MULTI_FUNCTIONS,
        'C': lambda args: math.comb(int(args[0]), int(args[1])),
        'P': lambda args: math.perm(int(args[0]), int(args[1])),
    }

    def Value(self):
        next = self.PeekNextToken()
        if next is not None and next[0].isalpha() and next in self.MULTI_FUNCTIONS:
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                self.PopNextToken()  # consume '('
                args = [self.Expr()]
                while self.PeekNextToken() == ",":
                    self.PopNextToken()  # consume ','
                    args.append(self.Expr())
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
                return self.MULTI_FUNCTIONS[name](args)
            elif name in self.CONSTANTS:
                return self.CONSTANTS[name]
            else:
                raise Exception(f"Unknown identifier '{name}'")
        return super().Value()


@register("calc9", description="Combinatorics: permutations and combinations",
          short_desc="Combinatorics", group="expression",
          examples=["C(10,3)", "P(5,2)", "C(52,5)"],
          i18n={"zh": "\u7ec4\u5408\u6570\u5b66", "hi": "\u0938\u0902\u092f\u094b\u091c\u0928", "es": "Combinatoria", "fr": "Combinatoire", "ar": "\u0627\u0644\u062a\u0648\u0627\u0641\u064a\u0642", "pt": "Combinat\u00f3ria", "ru": "\u041a\u043e\u043c\u0431\u0438\u043d\u0430\u0442\u043e\u0440\u0438\u043a\u0430", "ja": "\u7d44\u5408\u305b", "de": "Kombinatorik"})
def calc9(expression):
    """Combinatorics: permutations and combinations."""
    calculator = Calculator9(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)


# --- Calculator10: Extended Trig & Logs ---

class Calculator10(Calculator9):
    """Extended trig (degree mode, sec/csc/cot), arbitrary-base log, polar/rect conversion."""

    MULTI_FUNCTIONS = {
        **Calculator9.MULTI_FUNCTIONS,
        'logb': lambda args: cmath.log(args[1]) / cmath.log(args[0]),
        'polar': None,
        'rect': None,
    }

    FUNCTIONS = {
        **Calculator8.FUNCTIONS,
        'sec': lambda x: 1 / cmath.cos(x),
        'csc': lambda x: 1 / cmath.sin(x),
        'cot': lambda x: cmath.cos(x) / cmath.sin(x),
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
        if next is not None and next[0].isalpha():
            if next in self.MULTI_FUNCTIONS:
                name = self.PopNextToken()
                if self.PeekNextToken() == "(":
                    self.PopNextToken()  # consume '('
                    args = [self.Expr()]
                    while self.PeekNextToken() == ",":
                        self.PopNextToken()  # consume ','
                        args.append(self.Expr())
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    handler = self.MULTI_FUNCTIONS[name]
                    if name == 'polar':
                        return self._polar(args[0], args[1])
                    elif name == 'rect':
                        return self._rect(args[0], args[1])
                    else:
                        return handler(args)
                elif name in self.CONSTANTS:
                    return self.CONSTANTS[name]
                else:
                    raise Exception(f"Unknown identifier '{name}'")
        return Calculator9.Value(self)

    def _polar(self, x, y):
        r = abs(complex(x, y))
        theta = math.atan2(float(y), float(x))
        r_str = _fmt_num(_to_int(r))
        theta_str = _fmt_num(_to_int(round(theta, 10)))
        return f"({r_str}, {theta_str})"

    def _rect(self, r, theta):
        x = float(r) * math.cos(float(theta))
        y = float(r) * math.sin(float(theta))
        x_str = _fmt_num(_to_int(round(x, 10)))
        y_str = _fmt_num(_to_int(round(y, 10)))
        return f"({x_str}, {y_str})"


@register("calc10", description="Extended trig (degree mode), reciprocal trig, arbitrary-base logarithm, polar/rectangular conversion",
          short_desc="Trig & Logs", group="expression",
          examples=["sin(90d)", "sec(pi/4)", "logb(2,8)", "polar(3,4)"],
          i18n={"zh": "\u4e09\u89d2\u4e0e\u5bf9\u6570", "hi": "\u0924\u094d\u0930\u093f\u0915\u094b\u0923\u092e\u093f\u0924\u093f \u0914\u0930 \u0932\u0949\u0917", "es": "Trig y Logs", "fr": "Trigo et Logs", "ar": "\u062d\u0633\u0627\u0628 \u0627\u0644\u0645\u062b\u0644\u062b\u0627\u062a", "pt": "Trig e Logs", "ru": "\u0422\u0440\u0438\u0433. \u0438 \u043b\u043e\u0433.", "ja": "\u4e09\u89d2\u3068\u5bfe\u6570", "de": "Trig. und Log."})
def calc10(expression):
    """Extended trig, arbitrary-base log, coordinate conversion."""
    calculator = Calculator10(expression)
    result = calculator.Parse()
    if isinstance(result, str):
        return result
    return format_complex(result)
